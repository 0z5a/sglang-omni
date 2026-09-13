#!/usr/bin/env python3
"""Cookbook zero-shot clone probe against one Fun-CosyVoice3 server."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import requests

port = int(sys.argv[1])
out = Path(sys.argv[2])
url = f"http://127.0.0.1:{port}/v1/audio/speech"
payload = {
    "model": "FunAudioLLM/Fun-CosyVoice3-0.5B-2512",
    "input": "Get the trust fund to the bank early.",
    "ref_audio": (
        "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/"
        "resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
    ),
    "ref_text": (
        "We asked over twenty different people, and they all said it was his."
    ),
}
t0 = time.perf_counter()
resp = requests.post(url, json=payload, timeout=180)
elapsed = time.perf_counter() - t0
out.write_bytes(resp.content)
print(
    f"port={port} status={resp.status_code} bytes={len(resp.content)} "
    f"elapsed_s={elapsed:.3f} content_type={resp.headers.get('content-type')}"
)
resp.raise_for_status()
if len(resp.content) < 1000:
    raise SystemExit(f"probe response too small: {len(resp.content)} bytes")
