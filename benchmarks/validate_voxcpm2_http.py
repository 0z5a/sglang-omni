# SPDX-License-Identifier: Apache-2.0
"""Send native speech requests and save reproducible waveform smoke results.

Start an Omni VoxCPM2 server first. This checks transport and audio validity;
use an independent recognizer or a quality benchmark to evaluate intelligibility.
"""

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--base-url", default="http://127.0.0.1:8000")
p.add_argument("--model-label", default="base")
p.add_argument("--output-dir", required=True)
p.add_argument("--only-four", action="store_true")
a = p.parse_args()
root = Path(a.output_dir)
root.mkdir(parents=True, exist_ok=True)
report = {
    "passed": False,
    "cases": [],
    "scope": "Native Omni /v1/audio/speech waveform smoke check",
    "model_label": a.model_label,
}
recipes = [(4, 2.45, False)] if a.only_four else [(10, 2.0, True), (4, 2.45, False)]
try:
    for steps, cfg, zero in recipes:
        for language, text in [
            ("en", "Hello, this is a test of speech generation."),
            ("zh", "你好，这是一个语音生成测试。"),
        ]:
            payload = {
                "input": text,
                "response_format": "wav",
                "seed": 42,
                "stage_params": {
                    "tts_engine": {
                        "inference_timesteps": steps,
                        "cfg_value": cfg,
                        "sway_sampling_coef": 1.0,
                        "use_cfg_zero_star": zero,
                        "max_len": 120,
                    }
                },
            }
            start = time.monotonic()
            r = requests.post(
                f"{a.base_url.rstrip('/')}/v1/audio/speech", json=payload, timeout=240
            )
            row = {
                "recipe": payload,
                "http_status": r.status_code,
                "elapsed_s": time.monotonic() - start,
            }
            report["cases"].append(row)
            if r.status_code != 200:
                row["error"] = r.text[:8000]
            r.raise_for_status()
            wave, sr = sf.read(io.BytesIO(r.content), dtype="float32")
            path = root / f"{a.model_label}-{steps}-{language}.wav"
            path.write_bytes(r.content)
            row.update(
                path=str(path),
                sample_rate=sr,
                duration_s=len(wave) / sr,
                finite=bool(np.isfinite(wave).all()),
                rms=float(np.sqrt(np.mean(wave**2))),
            )
            row["passed"] = (
                sr == 48000
                and 0.5 < len(wave) / sr < 19.2
                and row["finite"]
                and row["rms"] > 0.001
            )
            assert row["passed"], row
            print(json.dumps(row, ensure_ascii=False), flush=True)
    report["passed"] = True
finally:
    (root / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
