"""Full text-to-waveform parity using upstream AR/VAE and Omni's sampler.

This validates the sampler in a complete audio generation loop, not the Omni
HTTP server or dynamic adapter loader. Audio metrics are smoke checks, not MOS.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from voxcpm.model.voxcpm2 import LoRAConfig, VoxCPM2Model

from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM


class ConfiguredSampler(torch.nn.Module):
    def __init__(self, sampler, zero_star):
        super().__init__()
        self.sampler = sampler
        self.zero_star = zero_star
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return self.sampler(
            **kwargs, sway_sampling_coef=1.0, use_cfg_zero_star=self.zero_star
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--adapter", required=True, type=Path)
    p.add_argument("--output-dir", required=True, type=Path)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((a.adapter / "lora_config.json").read_text())
    model = VoxCPM2Model.from_local(
        str(a.checkpoint),
        optimize=False,
        device="cuda:0",
        lora_config=LoRAConfig(**metadata["lora_config"]),
    ).eval()
    loaded, skipped = model.load_lora_weights(str(a.adapter))
    assert loaded and not skipped and set(loaded) == set(model.get_lora_state_dict())
    upstream = model.feat_decoder
    omni = UnifiedCFM(
        upstream.in_channels,
        CfmConfig(solver=upstream.solver),
        upstream.estimator,
        mean_mode=upstream.mean_mode,
    ).eval()
    report = {
        "scope": "full upstream AR/VAE text-to-waveform with Omni sampler",
        "native_omni_http_e2e": False,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "adapter_loaded_parameters": len(loaded),
        "effective_mean_mode": upstream.mean_mode,
        "adapter_meanflow_metadata": metadata.get("meanflow_model"),
        "seed": 42,
        "sample_rate": (
            model.audio_vae.config.sample_rate
            if hasattr(model.audio_vae, "config")
            else 48000
        ),
        "cases": [],
        "passed": False,
        "perceptual_quality_validated": False,
    }
    config = json.loads((a.checkpoint / "config.json").read_text())
    sample_rate = int(config["audio_vae_config"].get("out_sample_rate", 48000))
    report["sample_rate"] = sample_rate
    prompts = {
        "en": "The little bird flew over the quiet river.",
        "zh": "今天阳光很好，我们一起去公园散步。",
    }
    try:
        # Warm up the full AR/DiT/VAE loop before comparing repeated waveforms.
        model.feat_decoder = ConfiguredSampler(upstream, True)
        model.set_lora_enabled(False)
        model.generate(
            target_text=prompts["en"],
            inference_timesteps=10,
            cfg_value=2.0,
            seed=42,
            max_len=120,
            retry_badcase=False,
        )
        report["full_pipeline_warmup"] = True
        for recipe, steps, cfg, zero, adapter in (
            ("base10", 10, 2.0, True, False),
            ("base4", 4, 2.45, False, False),
            ("adapter4", 4, 2.45, False, True),
        ):
            for language, prompt in prompts.items():
                row = {
                    "recipe": recipe,
                    "steps": steps,
                    "cfg": cfg,
                    "zero_star": zero,
                    "adapter": adapter,
                    "prompt": prompt,
                    "language": language,
                    "implementations": {},
                }
                outputs = []
                for label, sampler in (("upstream", upstream), ("omni", omni)):
                    wrapped = ConfiguredSampler(sampler, zero)
                    model.feat_decoder = wrapped
                    model.set_lora_enabled(adapter)
                    torch.cuda.reset_peak_memory_stats()
                    start = time.monotonic()
                    wav = model.generate(
                        target_text=prompt,
                        inference_timesteps=steps,
                        cfg_value=cfg,
                        seed=42,
                        max_len=120,
                        retry_badcase=False,
                    )
                    torch.cuda.synchronize()
                    array = (
                        wav.detach().float().cpu().numpy().reshape(-1)
                        if isinstance(wav, torch.Tensor)
                        else np.asarray(wav).reshape(-1)
                    )
                    file = a.output_dir / f"{recipe}-{language}-{label}.wav"
                    sf.write(file, array, sample_rate, subtype="FLOAT")
                    duration = len(array) / sample_rate
                    peak, rms = float(np.abs(array).max()), float(
                        np.sqrt(np.mean(array**2))
                    )
                    row["implementations"][label] = {
                        "audio": str(file),
                        "elapsed_s": time.monotonic() - start,
                        "duration_s": duration,
                        "samples": len(array),
                        "peak": peak,
                        "rms": rms,
                        "finite": bool(np.isfinite(array).all()),
                        "sampler_calls": wrapped.calls,
                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
                    }
                    outputs.append(array)
                row["exact_waveform_parity"] = np.array_equal(*outputs)
                row["max_abs_difference"] = (
                    float(np.max(np.abs(outputs[0] - outputs[1])))
                    if outputs[0].shape == outputs[1].shape
                    else None
                )
                row["passed"] = row["exact_waveform_parity"] and all(
                    v["finite"]
                    and v["duration_s"] > 0.2
                    and v["rms"] > 1e-5
                    and v["sampler_calls"] > 0
                    for v in row["implementations"].values()
                )
                report["cases"].append(row)
                (a.output_dir / "report.json").write_text(
                    json.dumps(report, indent=2, ensure_ascii=False)
                )
                print(recipe, language, row["passed"], flush=True)
        report["passed"] = all(row["passed"] for row in report["cases"])
    except Exception as exc:
        report["error"] = repr(exc)
        raise
    finally:
        (a.output_dir / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )


if __name__ == "__main__":
    main()
