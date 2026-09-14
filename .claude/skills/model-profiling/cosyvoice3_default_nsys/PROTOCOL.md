# CosyVoice3 default-serve nsys 回归（可重复）

固定 workload：**SeedTTS EN，200 samples，concurrency 16，不开 streaming**。
Serve **只许 cookbook 默认**，不要加 snapshot / `--model-name` / TRT / compile。

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --port 8000
```

nsys **wrap 整段 serve**（不要 PID attach）：

```bash
nsys profile \
  --trace=cuda,nvtx \
  --gpu-metrics-devices=cuda-visible \
  --gpu-metrics-set=gh100 \
  --gpu-metrics-frequency=100 \
  -o <arm>/nsys/serve --force-overwrite true \
  -- sgl-omni serve \
    --model-path FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
    --port 8000
```

Nsight ≥ 2026.4。只留 `.nsys-rep`。读 GPU Metrics 的 **SMs Active / SM Issue / Tensor Active**，不要用 nvidia-smi `utilization.gpu`。

原始 trace / wav / sqlite 放 gitignored 的 `.profiling-runs/fun_cosyvoice3/`，不要提交。

## 环境

- 空闲 GPU：`CUDA_VISIBLE_DEVICES=<idle>`，确认 `utilization.gpu` **和** `memory.used` 都是 0。
- 不要杀别人的 PID。跑 200 时挂 `foreign_guard`；`foreign_pids≠0` 这臂作废。
- PYTHONPATH 必须指向**被测 commit 的 worktree**，不要指向 live HEAD。
- CosyVoice / Matcha 固定，另加 isolated `setuptools==80.9.0`（scratch 目录，不要写进共享 site-packages）。
- 权重走 HF 默认 path，实际 snapshot 记到 fingerprint。

## 请求分段（SM 只算第 4 段）

一次 serve 生命周期里会打 **1 + 48 + 16 + 200** 次合成：

| 段 | 请求 | 计入 SM / 正式吞吐？ |
|---|---|---|
| probe | 1 | 否 |
| preload | warmup 16 + 32 samples = 48 | 否 |
| headline warmup | 16（重复 `samples[0]`） | **否** |
| **headline 200** | **200 正式 samples** | **是** |

不要从「初始化后第一个 kernel」切窗。preload 会在 headline 之前先打 49 次（probe+48），再算进去会把 SM 稀释掉。

## Client

`cd` 到 **sglang-omni 仓库根**（benchmark 包），server 仍用 worktree 的 `sgl-omni`。

```bash
# 1) probe（单独脚本，1 条 zero-shot）
python .claude/skills/model-profiling/cosyvoice3_default_nsys/probe_one_request.py <port> <probe.wav>

# 2) preload：填 cache / graph，不进 SM 窗
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --lang en --max-concurrency 16 \
  --use-existing-server --generate-only \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --host 127.0.0.1 --port <port> \
  --output-dir <arm>/benches/preload32 \
  --max-samples 32 --warmup 16

# 3) headline：--warmup 16 会先跑再打日志 "Benchmarking 200"
python -m benchmarks.eval.benchmark_tts_seedtts \
  --model FunAudioLLM/Fun-CosyVoice3-0.5B-2512 \
  --lang en --max-concurrency 16 \
  --use-existing-server --generate-only \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --host 127.0.0.1 --port <port> \
  --output-dir <arm>/benches/headline200 \
  --max-samples 200 --warmup 16
```

不要加 `--stream`。`warmup=16` 与 concurrency 对齐，但 **warmup 不计入 SM、不计入正式 req/s**（runner 在 warmup 之后才 `t0 = perf_counter()`）。

200 跑完：`TERM` 只杀 `sgl-omni serve`，等 `.nsys-rep` 尺寸稳定且 nsys 父进程退出。不要 `SIGKILL nsys`。

## 算 SM（warmup 不计入）

1. 导出 sqlite：

```bash
nsys export --type sqlite --force-overwrite=true \
  -o <arm>/nsys/serve.sqlite \
  <arm>/nsys/serve.nsys-rep
```

2. 用 headline 的 `bench.log` 切窗：`Benchmarking 200 requests` → `Results saved`。
   这两行已经在 warmup 之后、正式 200 的墙钟两端。

```bash
python .claude/skills/model-profiling/cosyvoice3_default_nsys/compute_sm_window.py \
  --sqlite <arm>/nsys/serve.sqlite \
  --bench-log <arm>/benches/headline200/bench.log
```

脚本用 `TARGET_INFO_SESSION_START_TIME` 的 `localTime`（**不是** launch 墙钟，通常晚 15–25 s；日志的 asctime 是主机本地时钟，所以用同一个时钟）把日志时间换成 session 相对 ns，再对 `GPU_METRICS` 做时间平均。metricId 按名字前缀在 `TARGET_INFO_GPU_METRICS` 里查找（GR Active、SMs Active、SM Issue、Tensor Active），id 随 metric set 和驱动变，名字不变；每行输出附带 sample 数和 metricId。

采样 100 Hz，均值 = 窗内所有 sample 的算术平均（含 0）。窗口包含 cohort 自己的 ramp 和 drain，所以引用一个数时要带上它的窗口。H200 等价活跃 SM ≈ `SMs Active% / 100 × 132`。

更紧的 GPU 窗（首个正式 prefill → 最后 kernel）会再短 ~1–2 s，SM 大约低 0.1–0.3 pp；对比时两边用同一种切法。

## 读数时不要混

- nsys wrap 下的 req/s **带税**，不要和裸 DCGM / 不包 nsys 的数字比。
- 比 SM 用上面的 200 窗；比吞吐用 `headline200/speed_results.json` 的 `throughput_qps`。
- WER 另开：对已有 wav `--transcribe-only` + `Qwen/Qwen3-ASR-1.7B`，不要重合成。
- `foreign_pids≠0` 或 200 窗内显存掉档 → 臂作废。
