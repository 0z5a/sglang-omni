# Fun-CosyVoice3 nsys regression, repeatable

Fixed workload: SeedTTS EN, 200 samples, concurrency 16. The cookbook default is
buffered; a streaming claim adds `--stream` on both arms and says so next to the
numbers. Serve with the cookbook defaults only: no snapshot, no `--model-name`,
no TensorRT, no compile.

```bash
nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=100 \
  -o <arm>/nsys/serve --force-overwrite true \
  -- sgl-omni serve --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 --port 8000
```

Wrap the whole serve; do not attach by PID. Nsight 2026.4 or later. Keep only the
`.nsys-rep`. Read SMs Active, SM Issue, GR Active and Tensor Active from the GPU
metrics; never `nvidia-smi utilization.gpu`. Do not add `python-gil` or `osrt`
tracing to an arm whose cost is host launches: the tracer's own overhead lands on
that arm and moves the comparison.

## Environment

- An idle GPU: `CUDA_VISIBLE_DEVICES=<idle>`, with `utilization.gpu` and
  `memory.used` both zero.
- Never kill another user's PID. Run `foreign_guard` for the 200; an arm with
  `foreign_pids != 0` is void.
- `PYTHONPATH` points at the worktree of the commit under test, not the live head.
- CosyVoice and Matcha pinned, with an isolated `setuptools==80.9.0` in a scratch
  directory, never in the shared site packages.
- Weights from the default HF path; record the resolved snapshot in the fingerprint.

## Request segments; the SM window is the fourth only

One serve lifetime synthesises 1 + 48 + 16 + 200 requests:

| segment | requests | in the SM window and the headline req/s |
|---|---|---|
| probe | 1 | no |
| preload | warmup 16 + 32 samples | no |
| headline warmup | 16, sample zero repeated | no |
| headline 200 | 200 samples | yes |

Never cut the window at "the first kernel after init": the probe and the preload
put 49 requests in front of the headline and dilute the means.

## Client

Run from the repository root (the benchmark package); the server still runs from
the worktree under test.

```bash
# 1) probe, one zero shot request
python .claude/skills/model-profiling/cosyvoice3_default_nsys/probe_one_request.py <port> <probe.wav>

# 2) preload: fills caches and graphs, outside the window
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --lang en --max-concurrency 16 \
  --use-existing-server --generate-only \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --host 127.0.0.1 --port <port> \
  --output-dir <arm>/benches/preload32 \
  --max-samples 32 --warmup 16

# 3) headline: the warmup runs first, then the runner logs "Benchmarking 200 requests"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --lang en --max-concurrency 16 \
  --use-existing-server --generate-only \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --host 127.0.0.1 --port <port> \
  --output-dir <arm>/benches/headline200 \
  --max-samples 200 --warmup 16
```

`--warmup 16` matches the concurrency; the runner starts its clock after the
warmup, so the warmup is in neither the window nor the req/s. After the 200,
`TERM` the serve only and wait for the `.nsys-rep` to stop growing and the nsys
parent to exit; never `SIGKILL` nsys.

## The SM means

```bash
nsys export --type sqlite --force-overwrite=true -o <arm>/nsys/serve.sqlite <arm>/nsys/serve.nsys-rep
python .claude/skills/model-profiling/cosyvoice3_default_nsys/compute_sm_window.py \
  --sqlite <arm>/nsys/serve.sqlite \
  --bench-log <arm>/benches/headline200/bench.log
```

The window is `Benchmarking 200 requests` to `Results saved` in the headline
`bench.log`, converted to session relative nanoseconds with the `localTime` of
`TARGET_INFO_SESSION_START_TIME` (the log's asctime is the host's local clock;
the session start is usually 15 to 25 s after the launch wall clock). Metric ids
are looked up by name in `TARGET_INFO_GPU_METRICS`, since the ids follow the
metric set and the driver. Sampling at 100 Hz; a mean is the arithmetic mean of
every sample in the window, zeros included, and the window still holds the
cohort's own ramp and drain, so a mean is quoted with its window length and the
headline req/s. The script prints both, with the sample count and the metric id
per line. A tighter window, first headline prefill to the last kernel, is 1 to 2 s
shorter and about 0.1 to 0.3 pp lower; use one cut on both arms.

## Do not mix

- req/s under nsys carries the tracer's cost; compare it with the other arm's
  nsys run, never with a bare run.
- SM from the 200 window; throughput from `headline200/speed_results.json`
  `throughput_qps`.
- WER separately, `--transcribe-only` on the existing wavs with
  `Qwen/Qwen3-ASR-1.7B`; do not synthesise again.
- `foreign_pids != 0`, or a memory drop inside the window, voids the arm.
