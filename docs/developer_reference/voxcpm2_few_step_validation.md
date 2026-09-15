# VoxCPM2 few-step validation

The native `/v1/audio/speech` path accepts `inference_timesteps`, `cfg_value`,
`sway_sampling_coef`, and `use_cfg_zero_star` in `stage_params.tts_engine`.
The recipe is validated before generation; one step with zero-star enabled is
rejected because it skips the only DiT evaluation. Requests in one model batch
must use the same recipe.

This work builds on the serving implementation in #2117. The validation uses
`openbmb/VoxCPM2` and `voidful/FDSpeech-VoxCPM2` revision
`9d7a2ebb2067c7481e2a9c986a6e1886b956c491`. The adapter is merged offline into a
separate checkpoint for native serving. This does not implement dynamic adapter
selection or general diffusion-model LoRA loading.

## Reproduce native HTTP validation

Use the repository's pinned SGLang 0.5.19 dependency. To serve the base model:

```bash
python -m sglang_omni.cli serve \
  --model-path /path/to/VoxCPM2 \
  --config examples/configs/voxcpm2.yaml \
  --tts_engine.engine.max_total_tokens 4096 \
  --host 127.0.0.1 --port 8000
```

Then run:

```bash
python benchmarks/validate_voxcpm2_http.py \
  --base-url http://127.0.0.1:8000 --output-dir /tmp/voxcpm2-base
```

The validator writes the request recipes, HTTP status, duration, sample rate,
RMS, finite-value checks, and WAV files. It checks waveform validity; a successful
exit alone does not establish intelligibility or perceptual quality.

For the adapter checkpoint, stop the base server and merge into a new directory:

```bash
python benchmarks/merge_voxcpm2_lora.py \
  --base /path/to/VoxCPM2 --adapter /path/to/FDSpeech-VoxCPM2 \
  --output /path/to/VoxCPM2-FDSpeech-merged
```

Serve the merged checkpoint with the same command, replacing `--model-path`.
Use `--only-four --model-label adapter` with the HTTP validator.
The merge consumes all 384 adapter keys into 192 matrices, computes the updates
in FP32, casts back to the base dtype, and saves an evidence manifest. The base
weights remain unchanged. The base model's effective `mean_mode=false` is
preserved; the adapter's conflicting meanflow metadata is recorded separately.

## Results

All runs use BF16, TP=1, eager execution, one running request, seed 42, and
48 kHz WAV output. The English text is “Hello, this is a test of speech
generation.” The Chinese text is “你好，这是一个语音生成测试。”

| GPU | Recipe | HTTP / valid WAV | EN duration | ZH duration | Whisper-small WER / CER |
| --- | --- | --- | --- | --- | --- |
| RTX 4090 48 GB | Base: 10 steps, CFG 2, sway 1, zero-star on | 2/2 pass | 3.20 s | 3.04 s | 0% / 0% |
| RTX 4090 48 GB | Base: 4 steps, CFG 2.45, sway 1, zero-star off | 2/2 pass | 3.52 s | 3.20 s | 0% / 0% |
| RTX 4090 48 GB | Static adapter: 4 steps, CFG 2.45, sway 1, zero-star off | 2/2 pass | 3.52 s | 3.20 s | 0% / 0% |
| RTX PRO 4000 Blackwell 24 GB | Base: 10 steps, CFG 2, sway 1, zero-star on | 2/2 pass | 3.20 s | 3.20 s | 0% / 0% |
| RTX PRO 4000 Blackwell 24 GB | Base: 4 steps, CFG 2.45, sway 1, zero-star off | 2/2 pass | 3.52 s | 3.20 s | 0% / 0% |
| RTX PRO 4000 Blackwell 24 GB | Static adapter: 4 steps, CFG 2.45, sway 1, zero-star off | 2/2 pass | 3.52 s | 4.80 s | 0% / 0% |

Whisper-small uses temperature 0, beam size 5, fixed language, and no previous-text
conditioning. Scoring removes punctuation, lowercases English, and normalizes
Chinese traditional characters with OpenCC t2s. Whisper-base produced 16.7% CER
for the two post-fix 4090 base-model Chinese samples, recognizing “生成” as
“声程/声称”; its original transcripts are retained. Whisper-small recognized
both correctly. This two-text check is not a MOS/Seed-TTS benchmark, proof of
quality improvement, or exact native waveform parity across runs or GPUs.

| Additional check | Result |
| --- | --- |
| VoxCPM2 unit suite | 63 passed |
| Sampler with real base weights | 9 recipes exactly match the upstream sampler |
| Sampler with adapter weights | 3 recipes exactly match; nonzero adapter effect |
| Upstream AR/VAE with Omni sampler, after warmup | 6/6 complete audio pairs exactly match |
| Native adapter serving memory, RTX PRO 4000, 4096 KV tokens | 6396 MiB device-wide peak, sampled every 250 ms; idle baseline 2 MiB |

The memory observation is for a short, single-request run. It does not establish
the minimum card capacity for longer requests, concurrency, or CUDA graphs.
Native streaming, native reference-audio modes, dynamic LoRA loading, and TP>1
are outside this validation.

The native serving fixes cover checkpoint config normalization, packed projection
loading, text embeddings and Chinese character tokenization, prefill text/audio
masks, eager decode feedback, speech request contracts, and terminal waveform
serialization. Before the tokenizer correction, native Chinese ASR failed;
raw generic-tokenizer input also inserted special tokens absent upstream.
