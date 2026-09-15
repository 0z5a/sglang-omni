# SPDX-License-Identifier: Apache-2.0
"""Compare Omni's sampler with OpenBMB's on real checkpoint/adapter weights.

The estimators are shared to isolate the ODE/inference-recipe contract. This
does not validate Omni LoRA loading, model serving, or perceptual audio quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM
from sglang_omni.models.voxcpm2.sampling import VoxCPM2Sampling


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--recipes", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    recipes = [VoxCPM2Sampling(**row) for row in json.loads(args.recipes.read_text())]
    if not recipes:
        parser.error("--recipes must contain at least one inference recipe")
    if not torch.cuda.is_available():
        raise RuntimeError("checkpoint sampler validation requires CUDA")

    from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model

    lora_config = None
    adapter_files = {}
    metadata = {}
    loaded = []
    if args.adapter is not None:
        metadata = json.loads((args.adapter / "lora_config.json").read_text())
        lora_config = LoRAConfig(**metadata["lora_config"])
        # Do not accept arbitrary pickle checkpoints as a validation fixture.
        for name in ("lora_config.json", "lora_weights.safetensors"):
            adapter_files[name] = file_hash(args.adapter / name)
    model = VoxCPM2Model.from_local(
        str(args.checkpoint), optimize=False, device="cuda:0", lora_config=lora_config
    ).eval()
    if args.adapter is not None:
        loaded, skipped = model.load_lora_weights(str(args.adapter))
        missing = set(model.get_lora_state_dict()) - set(loaded)
        if not loaded or skipped or missing:
            raise RuntimeError(
                f"adapter loading incomplete: loaded={len(loaded)}, "
                f"skipped={skipped}, missing={sorted(missing)}"
            )
    reference = model.feat_decoder
    sampler = UnifiedCFM(
        reference.in_channels,
        CfmConfig(solver=reference.solver),
        reference.estimator,
        mean_mode=reference.mean_mode,
    ).eval()
    config = json.loads((args.checkpoint / "config.json").read_text())
    dtype = next(reference.estimator.parameters()).dtype
    torch.manual_seed(args.seed)
    mu = torch.randn(
        1, config["dit_config"]["hidden_dim"] * 2, device="cuda", dtype=dtype
    )
    cond = torch.randn(
        1, config["feat_dim"], config["patch_size"], device="cuda", dtype=dtype
    )
    torch.cuda.reset_peak_memory_stats()
    results = []
    for recipe in recipes:
        kwargs = {
            "mu": mu,
            "cond": cond,
            "patch_size": config["patch_size"],
            "n_timesteps": recipe.inference_timesteps,
            "cfg_value": recipe.cfg_value,
            "sway_sampling_coef": recipe.sway_sampling_coef,
            "use_cfg_zero_star": recipe.use_cfg_zero_star,
        }
        torch.manual_seed(args.seed)
        expected = reference(**kwargs)
        torch.manual_seed(args.seed)
        actual = sampler(**kwargs)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        adapter_effect = None
        if args.adapter is not None:
            model.set_lora_enabled(False)
            try:
                torch.manual_seed(args.seed)
                without_adapter = sampler(**kwargs)
            finally:
                model.set_lora_enabled(True)
            adapter_effect = float((actual - without_adapter).abs().max())
            if adapter_effect == 0:
                raise RuntimeError("loaded adapter does not affect the sampled patch")
        durations = {}
        for name, implementation in (("upstream", reference), ("omni", sampler)):
            implementation(**kwargs)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.repeats):
                implementation(**kwargs)
            torch.cuda.synchronize()
            durations[name] = (time.perf_counter() - start) * 1000 / args.repeats
        results.append(
            {
                "recipe": asdict(recipe),
                "max_abs_diff": float((actual - expected).abs().max()),
                "adapter_max_abs_effect": adapter_effect,
                "sampler_ms": durations,
            }
        )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    report = {
        "status": "passed",
        "scope": "sampler parity with shared estimator",
        "omni_revision": revision,
        "checkpoint": str(args.checkpoint),
        "checkpoint_config_sha256": file_hash(args.checkpoint / "config.json"),
        "checkpoint_weights_sha256": file_hash(args.checkpoint / "model.safetensors"),
        "adapter": str(args.adapter) if args.adapter else None,
        "adapter_files_sha256": adapter_files,
        "adapter_loaded_parameters": len(loaded),
        "adapter_model_metadata": metadata.get("meanflow_model"),
        "effective_mean_mode": reference.mean_mode,
        "sampler_source_sha256": file_hash(
            Path(__file__).parents[1] / "sglang_omni/models/voxcpm2/components/cfm.py"
        ),
        "distilled_quality_validated": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dtype": str(dtype),
        "seed": args.seed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
