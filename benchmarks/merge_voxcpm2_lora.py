# SPDX-License-Identifier: Apache-2.0
"""Build a separate static-merged checkpoint for native few-step validation.

This is an offline merge, not an Omni dynamic adapter loader. Preserve the base
model's sampler/mean-mode config; adapter training metadata is copied as evidence.
"""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base", type=Path, required=True)
p.add_argument("--adapter", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
assert not a.output.exists(), "Use a new output directory to preserve prior evidence"
metadata = json.loads((a.adapter / "lora_config.json").read_text())
cfg = metadata["lora_config"]
scale = cfg["alpha"] / cfg["r"]
weights = load_file(str(a.base / "model.safetensors"))
adapter = load_file(str(a.adapter / "lora_weights.safetensors"))
consumed, merged = set(), []
for key in sorted(adapter):
    if not key.endswith(".lora_A"):
        continue
    stem = key.removesuffix(".lora_A")
    b_key, w_key = stem + ".lora_B", stem + ".weight"
    left, right = adapter[b_key].float(), adapter[key].float()
    weight = weights[w_key]
    assert right.shape[0] == left.shape[1] == cfg["r"]
    delta = left @ right * scale
    assert delta.shape == weight.shape and torch.isfinite(delta).all()
    updated = (weight.float() + delta).to(weight.dtype)
    assert torch.isfinite(updated).all()
    weights[w_key] = updated
    merged.append(
        {
            "weight": w_key,
            "delta_max": delta.abs().max().item(),
            "changed": bool(torch.any(weight != updated)),
        }
    )
    consumed.update((key, b_key))
assert consumed == set(adapter), f"Unconsumed adapter keys: {set(adapter) - consumed}"
assert merged and any(x["changed"] for x in merged)
a.output.mkdir(parents=True)
for source in a.base.resolve().iterdir():
    if source.name not in ("model.safetensors", ".cache"):
        (a.output / source.name).symlink_to(source)
save_file(weights, str(a.output / "model.safetensors"))
report = {
    "scope": "offline static merge",
    "base": str(a.base),
    "adapter": str(a.adapter),
    "rank": cfg["r"],
    "alpha": cfg["alpha"],
    "adapter_keys": len(consumed),
    "merged_matrices": merged,
    "adapter_metadata": metadata,
    "adapter_sha256": hashlib.sha256(
        (a.adapter / "lora_weights.safetensors").read_bytes()
    ).hexdigest(),
}
(a.output / "static_merge_evidence.json").write_text(json.dumps(report, indent=2))
print(a.output, "adapter keys", len(consumed), "matrices", len(merged))
