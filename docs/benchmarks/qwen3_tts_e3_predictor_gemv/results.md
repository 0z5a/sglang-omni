# E3 / C4: Triton split-K skinny GEMM vs cuBLAS (Qwen3-TTS code predictor shapes)

GPU: NVIDIA H100 80GB HBM3 (132 SMs, L2 52 MB), torch 2.13.0+cu130, triton 3.7.1, CUDA 13.0, driver 580.126.20. Peak HBM assumed 3.35 TB/s. Timing: median per-call us from a CUDA graph of 20 back-to-back calls, 30 timed replays after 5 warm-up replays, weights rotated over 40 copies (two graphs alternated) so each call is HBM-cold.

Clocks during the run (nvidia-smi, 1 s samples, n=110): SM 345-1980 MHz, HBM 2619-2619 MHz; last sample: `1980, 2619, 228.54, 33, 0x0000000000000000`.

## Final table

| shape (K->N, MB) | M | cuBLAS us | Triton us | speedup | cuBLAS TB/s (%peak) | Triton TB/s (%peak) | cuBLAS max abs err | Triton max abs err | threshold | pass | stream floor us |
|---|---|---|---|---|---|---|---|---|---|---|---|
| qkv (1024->4096, 8.39) | 1 | 5.78 | 5.10 | 1.13x | 1.45 (43%) | 1.64 (49%) | 0.0098 | 0.0098 | n/a | - | 4.71 |
| qkv (1024->4096, 8.39) | 2 | 5.82 | 5.70 | 1.02x | 1.44 (43%) | 1.47 (44%) | 0.0098 | 0.0098 | n/a | - | 4.70 |
| qkv (1024->4096, 8.39) | 4 | 5.82 | 5.72 | 1.02x | 1.44 (43%) | 1.47 (44%) | 0.0098 | 0.0098 | n/a | - | 4.67 |
| qkv (1024->4096, 8.39) | 8 | 5.91 | 5.72 | 1.03x | 1.42 (42%) | 1.47 (44%) | 0.0137 | 0.0137 | n/a | - | 4.68 |
| qkv (1024->4096, 8.39) | 48 | 6.10 | 6.10 | 1.00x | 1.38 (41%) | 1.38 (41%) | 0.0154 | 0.0154 | <= cuBLAS | PASS | 4.71 |
| o_proj (2048->1024, 4.19) | 1 | 5.97 | 3.98 | 1.50x | 0.70 (21%) | 1.05 (31%) | 0.0192 | 0.0135 | <= 3.0 us | FAIL | 3.42 |
| o_proj (2048->1024, 4.19) | 2 | 5.97 | 4.41 | 1.35x | 0.70 (21%) | 0.95 (28%) | 0.0192 | 0.0135 | <= 3.0 us | FAIL | 3.42 |
| o_proj (2048->1024, 4.19) | 4 | 5.99 | 5.09 | 1.18x | 0.70 (21%) | 0.82 (25%) | 0.0192 | 0.0135 | <= 3.0 us | FAIL | 3.44 |
| o_proj (2048->1024, 4.19) | 8 | 6.01 | 6.38 | 0.94x | 0.70 (21%) | 0.66 (20%) | 0.0192 | 0.0135 | <= 3.0 us | FAIL | 3.42 |
| o_proj (2048->1024, 4.19) | 48 | 6.18 | 7.33 | 0.84x | 0.68 (20%) | 0.57 (17%) | 0.0219 | 0.0155 | <= cuBLAS | FAIL | 3.41 |
| o_proj_k1024 (1024->1024, 2.10) | 1 | 4.61 | 3.22 | 1.43x | 0.46 (14%) | 0.65 (19%) | 0.0192 | 0.0153 | n/a | - | 2.78 |
| o_proj_k1024 (1024->1024, 2.10) | 2 | 4.61 | 3.51 | 1.31x | 0.45 (14%) | 0.60 (18%) | 0.0192 | 0.0153 | n/a | - | 2.76 |
| o_proj_k1024 (1024->1024, 2.10) | 4 | 4.62 | 4.00 | 1.15x | 0.45 (14%) | 0.52 (16%) | 0.0192 | 0.0156 | n/a | - | 2.73 |
| o_proj_k1024 (1024->1024, 2.10) | 8 | 4.64 | 4.98 | 0.93x | 0.45 (13%) | 0.42 (13%) | 0.0192 | 0.0156 | n/a | - | 2.77 |
| o_proj_k1024 (1024->1024, 2.10) | 48 | 4.84 | 5.21 | 0.93x | 0.43 (13%) | 0.40 (12%) | 0.0231 | 0.0156 | <= cuBLAS | FAIL | 2.76 |
| gate_up (1024->6144, 12.58) | 1 | 7.16 | 6.48 | 1.11x | 1.76 (52%) | 1.94 (58%) | 0.0098 | 0.0098 | <= 5.5 us | FAIL | 6.03 |
| gate_up (1024->6144, 12.58) | 2 | 7.18 | 6.89 | 1.04x | 1.75 (52%) | 1.83 (55%) | 0.0098 | 0.0098 | <= 5.5 us | FAIL | 6.00 |
| gate_up (1024->6144, 12.58) | 4 | 7.23 | 6.92 | 1.04x | 1.74 (52%) | 1.82 (54%) | 0.0098 | 0.0098 | <= 5.5 us | FAIL | 6.02 |
| gate_up (1024->6144, 12.58) | 8 | 7.30 | 6.96 | 1.05x | 1.72 (51%) | 1.81 (54%) | 0.0137 | 0.0137 | <= 5.5 us | FAIL | 5.99 |
| gate_up (1024->6144, 12.58) | 48 | 7.65 | 7.75 | 0.99x | 1.64 (49%) | 1.62 (48%) | 0.0154 | 0.0154 | <= cuBLAS | FAIL | 5.98 |
| down (3072->1024, 6.29) | 1 | 7.78 | 4.73 | 1.64x | 0.81 (24%) | 1.33 (40%) | 0.0069 | 0.0146 | n/a | - | 4.05 |
| down (3072->1024, 6.29) | 2 | 7.84 | 5.50 | 1.43x | 0.80 (24%) | 1.14 (34%) | 0.0078 | 0.0146 | n/a | - | 4.07 |
| down (3072->1024, 6.29) | 4 | 7.81 | 6.86 | 1.14x | 0.81 (24%) | 0.92 (27%) | 0.0078 | 0.0154 | n/a | - | 4.06 |
| down (3072->1024, 6.29) | 8 | 7.81 | 7.30 | 1.07x | 0.81 (24%) | 0.86 (26%) | 0.0078 | 0.0154 | n/a | - | 4.06 |
| down (3072->1024, 6.29) | 48 | 8.29 | 8.49 | 0.98x | 0.76 (23%) | 0.74 (22%) | 0.0132 | 0.0156 | <= cuBLAS | FAIL | 4.07 |
| project_input (1024->2048, 4.19) | 1 | 5.20 | 3.85 | 1.35x | 0.81 (24%) | 1.09 (33%) | 0.0143 | 0.0143 | <= 3.0 us | FAIL | 3.43 |
| project_input (1024->2048, 4.19) | 2 | 5.18 | 4.40 | 1.18x | 0.81 (24%) | 0.95 (28%) | 0.0155 | 0.0155 | <= 3.0 us | FAIL | 3.44 |
| project_input (1024->2048, 4.19) | 4 | 5.22 | 4.93 | 1.06x | 0.80 (24%) | 0.85 (25%) | 0.0155 | 0.0155 | <= 3.0 us | FAIL | 3.42 |
| project_input (1024->2048, 4.19) | 8 | 5.21 | 5.00 | 1.04x | 0.81 (24%) | 0.84 (25%) | 0.0155 | 0.0155 | <= 3.0 us | FAIL | 3.42 |
| project_input (1024->2048, 4.19) | 48 | 5.41 | 5.29 | 1.02x | 0.78 (23%) | 0.79 (24%) | 0.0156 | 0.0156 | <= cuBLAS | PASS | 3.41 |
| lm_head (1024->2048, 4.19) | 1 | 5.01 | 3.72 | 1.35x | 0.84 (25%) | 1.13 (34%) | 0.0078 | 0.0078 | <= 3.0 us | FAIL | 3.42 |
| lm_head (1024->2048, 4.19) | 2 | 4.99 | 4.27 | 1.17x | 0.84 (25%) | 0.98 (29%) | 0.0078 | 0.0078 | <= 3.0 us | FAIL | 3.42 |
| lm_head (1024->2048, 4.19) | 4 | 5.02 | 4.82 | 1.04x | 0.83 (25%) | 0.87 (26%) | 0.0078 | 0.0078 | <= 3.0 us | FAIL | 3.41 |
| lm_head (1024->2048, 4.19) | 8 | 5.03 | 4.82 | 1.04x | 0.83 (25%) | 0.87 (26%) | 0.0078 | 0.0078 | <= 3.0 us | FAIL | 3.41 |
| lm_head (1024->2048, 4.19) | 48 | 5.21 | 5.10 | 1.02x | 0.80 (24%) | 0.82 (25%) | 0.0148 | 0.0148 | <= cuBLAS | PASS | 3.42 |

Thresholds: {"o_proj_le_3us": false, "gate_up_le_5p5us": false, "m48_not_slower": false, "numerics_ok": true}

## Cross-checks (eager torch.cuda.Event timing) and secondary cuBLAS variants

| shape | M | variant | graph median us | graph min us | eager single us | eager burst-20 us | replay max abs err (40 copies) |
|---|---|---|---|---|---|---|---|
| qkv | 1 | cublas | 5.78 | 5.71 | 16.86 | 7.27 | 0.0156 |
| qkv | 1 | triton | 5.10 | 5.06 | 24.42 | 6.75 | 0.0156 |
| qkv | 2 | cublas | 5.82 | 5.76 | 20.19 | 7.30 | 0.0156 |
| qkv | 2 | triton | 5.70 | 5.64 | 26.32 | 7.30 | 0.0156 |
| qkv | 4 | cublas | 5.82 | 5.79 | 20.58 | 7.27 | 0.0156 |
| qkv | 4 | triton | 5.72 | 5.68 | 26.42 | 7.33 | 0.0156 |
| qkv | 8 | cublas | 5.91 | 5.85 | 21.23 | 7.31 | 0.0156 |
| qkv | 8 | triton | 5.72 | 5.64 | 26.66 | 7.24 | 0.0156 |
| qkv | 48 | cublas | 6.10 | 6.02 | 22.16 | 7.52 | 0.0156 |
| qkv | 48 | triton | 6.10 | 6.01 | 26.75 | 7.59 | 0.0156 |
| o_proj | 1 | cublas | 5.97 | 5.93 | 21.60 | 7.36 | 0.0227 |
| o_proj | 1 | cublas_alt | 5.96 | 5.92 | 22.82 | 7.33 | 0.0153 |
| o_proj | 1 | triton | 3.98 | 3.94 | 24.46 | 5.48 | 0.0156 |
| o_proj | 2 | cublas | 5.97 | 5.94 | 20.32 | 7.34 | 0.0232 |
| o_proj | 2 | cublas_alt | 5.98 | 5.94 | 22.75 | 7.35 | 0.0153 |
| o_proj | 2 | triton | 4.41 | 4.36 | 23.36 | 5.86 | 0.0156 |
| o_proj | 4 | cublas | 5.99 | 5.92 | 20.70 | 7.32 | 0.0232 |
| o_proj | 4 | cublas_alt | 5.99 | 5.95 | 21.39 | 7.41 | 0.0153 |
| o_proj | 4 | triton | 5.09 | 5.04 | 23.36 | 6.50 | 0.0156 |
| o_proj | 8 | cublas | 6.01 | 5.95 | 18.98 | 7.43 | 0.0234 |
| o_proj | 8 | cublas_alt | 5.99 | 5.95 | 20.00 | 7.36 | 0.0153 |
| o_proj | 8 | triton | 6.38 | 6.30 | 26.80 | 7.86 | 0.0156 |
| o_proj | 48 | cublas | 6.18 | 6.12 | 23.30 | 7.54 | 0.0277 |
| o_proj | 48 | cublas_alt | 6.18 | 6.12 | 24.03 | 7.48 | 0.0156 |
| o_proj | 48 | triton | 7.33 | 7.27 | 30.80 | 8.84 | 0.0156 |
| o_proj_k1024 | 1 | cublas | 4.61 | 4.54 | 21.36 | 6.09 | 0.0232 |
| o_proj_k1024 | 1 | cublas_alt | 4.61 | 4.56 | 22.80 | 6.07 | 0.0133 |
| o_proj_k1024 | 1 | triton | 3.22 | 3.19 | 26.93 | 4.63 | 0.0156 |
| o_proj_k1024 | 2 | cublas | 4.61 | 4.56 | 20.50 | 6.06 | 0.0232 |
| o_proj_k1024 | 2 | cublas_alt | 4.61 | 4.56 | 22.91 | 6.07 | 0.0133 |
| o_proj_k1024 | 2 | triton | 3.51 | 3.48 | 25.44 | 4.84 | 0.0156 |
| o_proj_k1024 | 4 | cublas | 4.62 | 4.56 | 22.35 | 6.06 | 0.0232 |
| o_proj_k1024 | 4 | cublas_alt | 4.62 | 4.56 | 21.71 | 6.07 | 0.0150 |
| o_proj_k1024 | 4 | triton | 4.00 | 3.94 | 26.32 | 5.33 | 0.0156 |
| o_proj_k1024 | 8 | cublas | 4.64 | 4.57 | 21.31 | 6.08 | 0.0303 |
| o_proj_k1024 | 8 | cublas_alt | 4.66 | 4.60 | 21.23 | 6.07 | 0.0150 |
| o_proj_k1024 | 8 | triton | 4.98 | 4.91 | 25.36 | 6.35 | 0.0156 |
| o_proj_k1024 | 48 | cublas | 4.84 | 4.81 | 21.18 | 6.15 | 0.0308 |
| o_proj_k1024 | 48 | cublas_alt | 4.85 | 4.79 | 21.71 | 6.20 | 0.0155 |
| o_proj_k1024 | 48 | triton | 5.21 | 5.17 | 27.54 | 6.52 | 0.0156 |
| gate_up | 1 | cublas | 7.16 | 7.12 | 21.54 | 8.59 | 0.0156 |
| gate_up | 1 | triton | 6.48 | 6.44 | 28.08 | 8.08 | 0.0156 |
| gate_up | 2 | cublas | 7.18 | 7.13 | 24.59 | 8.66 | 0.0156 |
| gate_up | 2 | triton | 6.89 | 6.82 | 29.06 | 8.42 | 0.0156 |
| gate_up | 4 | cublas | 7.23 | 7.17 | 22.78 | 8.72 | 0.0156 |
| gate_up | 4 | triton | 6.92 | 6.88 | 29.55 | 8.46 | 0.0156 |
| gate_up | 8 | cublas | 7.30 | 7.23 | 24.48 | 8.74 | 0.0156 |
| gate_up | 8 | triton | 6.96 | 6.90 | 30.11 | 8.48 | 0.0156 |
| gate_up | 48 | cublas | 7.65 | 7.59 | 28.58 | 9.03 | 0.0156 |
| gate_up | 48 | triton | 7.75 | 7.67 | 28.94 | 9.21 | 0.0156 |
| down | 1 | cublas | 7.78 | 7.71 | 27.86 | 8.52 | 0.0130 |
| down | 1 | cublas_alt | 9.32 | 9.26 | 35.90 | 11.18 | 0.0228 |
| down | 1 | triton | 4.73 | 4.70 | 28.10 | 6.34 | 0.0155 |
| down | 2 | cublas | 7.84 | 7.79 | 24.56 | 8.60 | 0.0140 |
| down | 2 | cublas_alt | 9.46 | 9.40 | 33.71 | 11.24 | 0.0232 |
| down | 2 | triton | 5.50 | 5.44 | 40.18 | 7.20 | 0.0155 |
| down | 4 | cublas | 7.81 | 7.76 | 27.17 | 8.61 | 0.0154 |
| down | 4 | cublas_alt | 9.48 | 9.43 | 32.58 | 11.27 | 0.0233 |
| down | 4 | triton | 6.86 | 6.80 | 26.74 | 8.40 | 0.0156 |
| down | 8 | cublas | 7.81 | 7.76 | 26.64 | 8.55 | 0.0154 |
| down | 8 | cublas_alt | 9.50 | 9.44 | 35.55 | 11.34 | 0.0234 |
| down | 8 | triton | 7.30 | 7.25 | 31.23 | 8.82 | 0.0156 |
| down | 48 | cublas | 8.29 | 8.23 | 27.44 | 9.04 | 0.0154 |
| down | 48 | cublas_alt | 10.04 | 9.96 | 33.60 | 11.78 | 0.0284 |
| down | 48 | triton | 8.49 | 8.44 | 30.50 | 9.96 | 0.0156 |
| project_input | 1 | cublas | 5.20 | 5.16 | 22.02 | 6.66 | 0.0156 |
| project_input | 1 | triton | 3.85 | 3.79 | 25.17 | 5.42 | 0.0156 |
| project_input | 2 | cublas | 5.18 | 5.11 | 22.70 | 6.63 | 0.0156 |
| project_input | 2 | triton | 4.40 | 4.35 | 25.17 | 5.94 | 0.0156 |
| project_input | 4 | cublas | 5.22 | 5.15 | 22.22 | 6.65 | 0.0156 |
| project_input | 4 | triton | 4.93 | 4.88 | 27.30 | 6.51 | 0.0156 |
| project_input | 8 | cublas | 5.21 | 5.17 | 20.75 | 6.62 | 0.0156 |
| project_input | 8 | triton | 5.00 | 4.97 | 25.52 | 6.55 | 0.0156 |
| project_input | 48 | cublas | 5.41 | 5.35 | 21.02 | 6.80 | 0.0156 |
| project_input | 48 | triton | 5.29 | 5.22 | 25.86 | 6.65 | 0.0156 |
| lm_head | 1 | cublas | 5.01 | 4.96 | 25.98 | 6.43 | 0.0153 |
| lm_head | 1 | triton | 3.72 | 3.66 | 24.03 | 5.27 | 0.0153 |
| lm_head | 2 | cublas | 4.99 | 4.95 | 20.05 | 6.42 | 0.0153 |
| lm_head | 2 | triton | 4.27 | 4.22 | 26.22 | 5.76 | 0.0153 |
| lm_head | 4 | cublas | 5.02 | 4.98 | 20.43 | 6.39 | 0.0153 |
| lm_head | 4 | triton | 4.82 | 4.79 | 25.84 | 6.33 | 0.0153 |
| lm_head | 8 | cublas | 5.03 | 4.99 | 20.00 | 6.42 | 0.0156 |
| lm_head | 8 | triton | 4.82 | 4.79 | 25.65 | 6.30 | 0.0156 |
| lm_head | 48 | cublas | 5.21 | 5.13 | 21.89 | 6.55 | 0.0156 |
| lm_head | 48 | triton | 5.10 | 5.05 | 27.20 | 6.50 | 0.0156 |

## Triton config chosen per shape (autotuned per (shape, M))

| shape | M | config | CTAs | autotune best us | configs tried | stage-A top3 | best dot kernel | best reg / reg3 kernel |
|---|---|---|---|---|---|---|---|---|
| qkv | 1 | reg_BN4_KS1024_SK1_w4 | 1024 | 5.100 | 123 | reg_BN4_KS1024_SK1_w4 5.10, reg3_BN2_KS1024_w4 5.10, reg_BN8_KS1024_SK1_w4 5.13 | dot_BN16_BK64_SK1_w2_s8_r0 5.69 | reg_BN4_KS1024_SK1_w4 5.10 / reg3_BN2_KS1024_w4 5.10 |
| qkv | 2 | dot_BN16_BK64_SK1_w2_s8_r0 | 256 | 5.712 | 137 | dot_BN16_BK64_SK1_w2_s8_r0 5.71, dot_BN32_BK128_SK1_w4_s5_r0 5.72, dot_BN32_BK64_SK1_w4_s8_r0 5.76 | dot_BN16_BK64_SK1_w2_s8_r0 5.71 | reg_BN8_KS1024_SK1_w4 5.88 / reg3_BN4_KS1024_w4 5.79 |
| qkv | 4 | dot_BN16_BK64_SK1_w2_s8_r0 | 256 | 5.719 | 137 | dot_BN16_BK64_SK1_w2_s8_r0 5.72, dot_BN32_BK128_SK1_w4_s5_r0 5.73, dot_BN32_BK64_SK1_w4_s8_r0 5.83 | dot_BN16_BK64_SK1_w2_s8_r0 5.72 | reg_BN4_KS1024_SK1_w2 7.37 / reg3_BN4_KS1024_w4 7.13 |
| qkv | 8 | dot_BN32_BK128_SK1_w4_s6_r0 | 128 | 5.761 | 137 | dot_BN32_BK128_SK1_w4_s5_r0 5.79, dot_BN16_BK64_SK1_w2_s8_r0 5.82, dot_BN32_BK64_SK1_w4_s8_r0 5.84 | dot_BN32_BK128_SK1_w4_s6_r0 5.76 | reg_BN4_KS1024_SK1_w2 10.39 / reg3_BN4_KS1024_w4 10.48 |
| qkv | 48 | dot_BN32_BK64_SK1_w4_s8_r0 | 128 | 6.111 | 84 | dot_BN32_BK64_SK1_w4_s5_r1 6.71, dot_BN16_BK128_SK1_w4_s3_r1 6.94, dot_BN64_BK32_SK1_w4_s8_r1 7.48 | dot_BN32_BK64_SK1_w4_s8_r0 6.11 | - / - |
| o_proj | 1 | reg_BN4_KS2048_SK1_w8 | 256 | 3.992 | 109 | reg_BN4_KS2048_SK1_w8 3.99, reg_BN4_KS2048_SK1_w16 4.08, reg3_BN4_KS2048_w4 4.10 | dot_BN32_BK64_SK8_w4_s4_r0 6.22 | reg_BN4_KS2048_SK1_w8 3.99 / reg3_BN4_KS2048_w4 4.10 |
| o_proj | 2 | reg3_BN4_KS2048_w4 | 256 | 4.414 | 109 | reg3_BN4_KS2048_w4 4.41, reg3_BN2_KS2048_w4 4.53, reg3_BN4_KS2048_w8 4.65 | dot_BN64_BK32_SK8_w4_s8_r0 6.37 | reg_BN4_KS2048_SK1_w8 4.94 / reg3_BN4_KS2048_w4 4.41 |
| o_proj | 4 | reg3_BN4_KS2048_w4 | 256 | 5.110 | 109 | reg3_BN4_KS2048_w4 5.11, reg3_BN2_KS2048_w4 5.15, reg3_BN4_KS2048_w8 5.46 | dot_BN32_BK64_SK8_w4_s4_r0 6.54 | reg_BN4_KS2048_SK1_w8 6.70 / reg3_BN4_KS2048_w4 5.11 |
| o_proj | 8 | dot_BN64_BK32_SK8_w4_s6_r0 | 128 | 6.370 | 145 | dot_BN64_BK64_SK8_w4_s4_r0 6.46, dot_BN64_BK32_SK8_w4_s8_r0 6.52, reg3_BN2_KS2048_w4 6.60 | dot_BN64_BK32_SK8_w4_s6_r0 6.37 | reg_BN4_KS2048_SK1_w8 10.22 / reg3_BN2_KS2048_w4 6.60 |
| o_proj | 48 | dot_BN16_BK128_SK2_w4_s4_r1 | 128 | 7.369 | 106 | dot_BN16_BK128_SK2_w4_s3_r1 7.97, dot_BN32_BK64_SK2_w4_s5_r1 8.31, dot_BN32_BK64_SK4_w4_s5_r1 8.44 | dot_BN16_BK128_SK2_w4_s4_r1 7.37 | - / - |
| o_proj_k1024 | 1 | reg3_BN2_KS1024_w4 | 512 | 3.222 | 101 | reg3_BN2_KS1024_w4 3.22, reg_BN4_KS1024_SK1_w4 3.25, reg_BN4_KS1024_SK1_w8 3.33 | dot_BN16_BK64_SK1_w2_s8_r0 4.94 | reg_BN4_KS1024_SK1_w4 3.25 / reg3_BN2_KS1024_w4 3.22 |
| o_proj_k1024 | 2 | reg3_BN2_KS1024_w4 | 512 | 3.500 | 101 | reg3_BN2_KS1024_w4 3.50, reg3_BN4_KS1024_w4 3.54, reg3_BN2_KS1024_w8 3.61 | dot_BN16_BK128_SK1_w2_s8_r0 4.96 | reg_BN4_KS1024_SK1_w4 4.06 / reg3_BN2_KS1024_w4 3.50 |
| o_proj_k1024 | 4 | reg3_BN2_KS1024_w4 | 512 | 4.024 | 101 | reg3_BN2_KS1024_w4 4.02, reg3_BN4_KS1024_w4 4.04, reg3_BN8_KS1024_w4 4.16 | dot_BN16_BK128_SK1_w2_s8_r0 5.01 | reg_BN4_KS1024_SK1_w4 5.59 / reg3_BN2_KS1024_w4 4.02 |
| o_proj_k1024 | 8 | reg3_BN4_KS1024_w4 | 256 | 4.952 | 101 | reg3_BN4_KS1024_w4 4.95, reg3_BN2_KS1024_w4 4.96, dot_BN16_BK128_SK1_w2_s8_r0 5.02 | dot_BN16_BK128_SK1_w2_s8_r0 5.02 | reg_BN4_KS1024_SK1_w4 8.74 / reg3_BN4_KS1024_w4 4.95 |
| o_proj_k1024 | 48 | dot_BN16_BK128_SK1_w4_s6_r0 | 64 | 5.245 | 80 | dot_BN16_BK128_SK1_w4_s3_r1 5.88, dot_BN16_BK128_SK2_w4_s3_r1 6.37, dot_BN32_BK64_SK2_w4_s5_r1 6.67 | dot_BN16_BK128_SK1_w4_s6_r0 5.24 | - / - |
| gate_up | 1 | reg_BN8_KS1024_SK1_w8 | 768 | 6.445 | 109 | reg_BN8_KS1024_SK1_w8 6.44, reg_BN8_KS1024_SK1_w4 6.47, reg3_BN2_KS1024_w4 6.48 | dot_BN16_BK64_SK1_w2_s8_r0 7.00 | reg_BN8_KS1024_SK1_w8 6.44 / reg3_BN2_KS1024_w4 6.48 |
| gate_up | 2 | dot_BN64_BK64_SK1_w4_s8_r0 | 96 | 6.864 | 127 | dot_BN16_BK64_SK1_w2_s8_r0 7.02, dot_BN64_BK64_SK1_w4_s6_r0 7.03, dot_BN32_BK64_SK1_w4_s8_r0 7.16 | dot_BN64_BK64_SK1_w4_s8_r0 6.86 | reg_BN8_KS1024_SK1_w4 7.77 / reg3_BN4_KS1024_w4 7.36 |
| gate_up | 4 | dot_BN64_BK64_SK1_w4_s8_r0 | 96 | 6.941 | 127 | dot_BN64_BK64_SK1_w4_s6_r0 7.09, dot_BN16_BK64_SK1_w2_s8_r0 7.11, dot_BN32_BK64_SK1_w4_s8_r0 7.17 | dot_BN64_BK64_SK1_w4_s8_r0 6.94 | reg_BN8_KS1024_SK1_w4 10.35 / reg3_BN8_KS1024_w4 8.89 |
| gate_up | 8 | dot_BN64_BK64_SK1_w4_s8_r0 | 96 | 6.987 | 127 | dot_BN64_BK64_SK1_w4_s6_r0 7.14, dot_BN16_BK64_SK1_w2_s8_r0 7.17, dot_BN32_BK64_SK1_w4_s8_r0 7.24 | dot_BN64_BK64_SK1_w4_s8_r0 6.99 | reg_BN8_KS1024_SK1_w4 16.39 / reg3_BN8_KS1024_w4 13.52 |
| gate_up | 48 | dot_BN32_BK64_SK1_w4_s6_r0 | 192 | 7.775 | 76 | dot_BN32_BK64_SK1_w4_s5_r1 8.10, dot_BN16_BK128_SK1_w4_s3_r1 8.34, dot_BN64_BK32_SK1_w4_s8_r1 8.58 | dot_BN32_BK64_SK1_w4_s6_r0 7.78 | - / - |
| down | 1 | reg_BN4_KS4096_SK1_w8 | 256 | 4.739 | 140 | reg_BN4_KS4096_SK1_w8 4.74, reg_BN4_KS4096_SK1_w16 4.78, reg_BN8_KS4096_SK1_w16 4.98 | dot_BN64_BK32_SK8_w4_s8_r0 7.14 | reg_BN4_KS4096_SK1_w8 4.74 / reg3_BN2_KS4096_w4 5.01 |
| down | 2 | reg3_BN4_KS4096_w8 | 256 | 5.514 | 140 | reg3_BN4_KS4096_w8 5.51, reg_BN4_KS4096_SK1_w8 5.70, reg3_BN2_KS4096_w8 5.74 | dot_BN64_BK32_SK8_w4_s8_r0 7.16 | reg_BN4_KS4096_SK1_w8 5.70 / reg3_BN4_KS4096_w8 5.51 |
| down | 4 | reg3_BN2_KS4096_w4 | 512 | 6.488 | 140 | reg3_BN2_KS4096_w4 6.49, reg3_BN4_KS4096_w8 6.86, dot_BN32_BK64_SK4_w4_s8_r0 7.29 | dot_BN32_BK64_SK4_w4_s8_r0 7.29 | reg_BN4_KS4096_SK1_w8 7.54 / reg3_BN2_KS4096_w4 6.49 |
| down | 8 | dot_BN64_BK64_SK8_w4_s6_r0 | 128 | 7.299 | 182 | dot_BN64_BK64_SK8_w4_s6_r0 7.30, dot_BN64_BK32_SK8_w4_s8_r0 7.37, dot_BN32_BK128_SK4_w4_s5_r0 7.76 | dot_BN64_BK64_SK8_w4_s6_r0 7.30 | reg_BN4_KS4096_SK1_w8 11.65 / reg3_BN2_KS4096_w4 10.75 |
| down | 48 | dot_BN16_BK128_SK2_w4_s6_r1 | 128 | 8.415 | 98 | dot_BN32_BK64_SK4_w4_s5_r1 9.57, dot_BN16_BK128_SK2_w4_s3_r1 9.69, dot_BN32_BK64_SK2_w4_s5_r1 9.94 | dot_BN16_BK128_SK2_w4_s6_r1 8.42 | - / - |
| project_input | 1 | reg_BN4_KS1024_SK1_w4 | 512 | 3.810 | 119 | reg_BN4_KS1024_SK1_w4 3.81, reg3_BN2_KS1024_w4 3.91, reg_BN4_KS1024_SK1_w8 3.97 | dot_BN32_BK128_SK1_w4_s5_r0 4.94 | reg_BN4_KS1024_SK1_w4 3.81 / reg3_BN2_KS1024_w4 3.91 |
| project_input | 2 | reg_BN4_KS1024_SK1_w4 | 512 | 4.365 | 119 | reg_BN4_KS1024_SK1_w4 4.36, reg3_BN2_KS1024_w4 4.45, reg3_BN4_KS1024_w4 4.45 | dot_BN32_BK128_SK1_w4_s5_r0 4.98 | reg_BN4_KS1024_SK1_w4 4.36 / reg3_BN2_KS1024_w4 4.45 |
| project_input | 4 | dot_BN32_BK128_SK1_w4_s5_r0 | 64 | 4.965 | 133 | dot_BN32_BK128_SK1_w4_s5_r0 4.96, dot_BN16_BK64_SK1_w2_s8_r0 4.99, dot_BN32_BK64_SK1_w4_s8_r0 5.03 | dot_BN32_BK128_SK1_w4_s5_r0 4.96 | reg_BN4_KS1024_SK1_w4 5.68 / reg3_BN8_KS1024_w4 5.18 |
| project_input | 8 | dot_BN32_BK128_SK1_w4_s5_r0 | 64 | 4.970 | 133 | dot_BN32_BK128_SK1_w4_s5_r0 4.97, dot_BN16_BK64_SK1_w2_s8_r0 4.98, dot_BN32_BK64_SK1_w4_s8_r0 5.06 | dot_BN32_BK128_SK1_w4_s5_r0 4.97 | reg_BN4_KS1024_SK1_w4 8.20 / reg3_BN8_KS1024_w4 6.81 |
| project_input | 48 | dot_BN16_BK128_SK1_w4_s6_r0 | 128 | 5.294 | 90 | dot_BN32_BK64_SK1_w4_s5_r1 5.82, dot_BN16_BK128_SK1_w4_s3_r1 6.01, dot_BN32_BK64_SK2_w4_s5_r1 6.70 | dot_BN16_BK128_SK1_w4_s6_r0 5.29 | - / - |
| lm_head | 1 | reg_BN4_KS1024_SK1_w4 | 512 | 3.713 | 119 | reg_BN4_KS1024_SK1_w4 3.71, reg_BN8_KS1024_SK1_w4 3.80, reg3_BN2_KS1024_w4 3.81 | dot_BN16_BK64_SK1_w2_s8_r0 4.81 | reg_BN4_KS1024_SK1_w4 3.71 / reg3_BN2_KS1024_w4 3.81 |
| lm_head | 2 | reg_BN4_KS1024_SK1_w4 | 512 | 4.274 | 119 | reg_BN4_KS1024_SK1_w4 4.27, reg3_BN4_KS1024_w4 4.31, reg3_BN2_KS1024_w4 4.34 | dot_BN16_BK64_SK1_w2_s8_r0 4.86 | reg_BN4_KS1024_SK1_w4 4.27 / reg3_BN4_KS1024_w4 4.31 |
| lm_head | 4 | dot_BN32_BK128_SK1_w4_s5_r0 | 64 | 4.836 | 138 | dot_BN32_BK128_SK1_w4_s5_r0 4.84, dot_BN32_BK64_SK1_w4_s8_r0 4.86, dot_BN16_BK64_SK1_w2_s8_r0 4.86 | dot_BN32_BK128_SK1_w4_s5_r0 4.84 | reg_BN4_KS1024_SK1_w4 5.52 / reg3_BN2_KS1024_w4 5.07 |
| lm_head | 8 | dot_BN32_BK128_SK1_w4_s5_r0 | 64 | 4.846 | 133 | dot_BN32_BK128_SK1_w4_s5_r0 4.85, dot_BN16_BK64_SK1_w2_s8_r0 4.86, dot_BN32_BK64_SK1_w4_s8_r0 4.88 | dot_BN32_BK128_SK1_w4_s5_r0 4.85 | reg_BN4_KS1024_SK1_w4 7.96 / reg3_BN8_KS1024_w4 6.67 |
| lm_head | 48 | dot_BN16_BK128_SK1_w4_s6_r0 | 128 | 5.107 | 90 | dot_BN32_BK64_SK1_w4_s5_r1 5.65, dot_BN16_BK128_SK1_w4_s3_r1 5.81, dot_BN32_BK64_SK2_w4_s5_r1 6.60 | dot_BN16_BK128_SK1_w4_s6_r0 5.11 | - / - |

## cuBLAS kernels observed (torch.profiler over eager calls of the predictor path)

- qkv M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.45us
- qkv M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.27us
- qkv M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.26us
- qkv M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.29us
- qkv M=48 cublas: `nvjet_sm90_tst_64x24_64x16_1x2_h_bz_TNT` x3 5.69us
- o_proj M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 5.44us
- o_proj M=1 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.49us
- o_proj M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 5.49us
- o_proj M=2 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.40us
- o_proj M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 5.45us
- o_proj M=4 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.45us
- o_proj M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 5.47us
- o_proj M=8 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 5.49us
- o_proj M=48 cublas: `nvjet_sm90_tst_64x24_64x16_4x2_h_badd_TNT` x3 5.66us
- o_proj M=48 cublas_alt: `nvjet_sm90_tst_64x24_64x16_4x2_h_bz_TNT` x3 5.61us
- o_proj_k1024 M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 4.35us
- o_proj_k1024 M=1 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.34us
- o_proj_k1024 M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 4.28us
- o_proj_k1024 M=2 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.34us
- o_proj_k1024 M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 4.38us
- o_proj_k1024 M=4 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.32us
- o_proj_k1024 M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_badd_TNT` x3 4.38us
- o_proj_k1024 M=8 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.32us
- o_proj_k1024 M=48 cublas: `nvjet_sm90_tst_64x24_64x16_4x2_h_badd_TNT` x3 4.34us
- o_proj_k1024 M=48 cublas_alt: `nvjet_sm90_tst_64x24_64x16_4x2_h_bz_TNT` x3 4.37us
- gate_up M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 6.65us
- gate_up M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 6.59us
- gate_up M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 6.76us
- gate_up M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 6.72us
- gate_up M=48 cublas: `nvjet_sm90_tst_64x48_64x15_4x1_v_bz_TNT` x3 7.05us
- down M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.15us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.69us
- down M=1 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.15us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.68us; `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` x3 1.13us
- down M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.17us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.70us
- down M=2 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.18us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.65us; `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` x3 1.21us
- down M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.18us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.71us
- down M=4 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.19us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.71us; `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` x3 1.21us
- down M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.21us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.70us
- down M=8 cublas_alt: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_splitK_TNT` x3 5.18us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.67us; `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` x3 1.19us
- down M=48 cublas: `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_splitK_TNT` x3 5.51us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.78us
- down M=48 cublas_alt: `nvjet_sm90_tst_64x32_64x16_4x2_h_bz_splitK_TNT` x3 5.46us; `void cublasLt::splitKreduce_kernel<32, 16, int, float, __nv_bfloat16, float, __nv_bfloat16, false, float, __nv_bfloat16, __nv_bfloat16, true, false, false, false>(cublasLt::cublasSplitKParams<float>, float const*, __nv_bfloat16 const*, float*, __nv_bfloat16*, float const*, float const*, __nv_bfloat16 const*, float const*, __nv_bfloat16*, void*, long, float*, int*, float*, float*, float const*, float const*, float const*, float const*, float const*)` x3 1.82us; `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<c10::BFloat16>, std::array<char*, 3ul>)` x3 1.36us
- project_input M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` x3 4.73us
- project_input M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` x3 4.68us
- project_input M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` x3 4.63us
- project_input M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_bias_TNT` x3 4.68us
- project_input M=48 cublas: `nvjet_sm90_tst_64x24_64x16_4x2_h_bz_bias_TNT` x3 4.86us
- lm_head M=1 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.51us
- lm_head M=2 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.44us
- lm_head M=4 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.50us
- lm_head M=8 cublas: `nvjet_sm90_tst_64x8_64x16_4x1_v_bz_TNT` x3 4.52us
- lm_head M=48 cublas: `nvjet_sm90_tst_64x24_64x16_4x2_h_bz_TNT` x3 4.66us

## Harness floor (what any kernel can reach inside this graph regime)

`floor_probe.py` / `floor_probe.txt` (same harness): an empty kernel costs 1.17-1.36 us per graph node; a pure streaming read that does no math takes 3.41 us for 4.19 MB, 4.01 us for 6.29 MB and 5.97 us for 12.58 MB, i.e. t ~= 2.1 us + bytes / 3.3 TB/s. The per-shape `stream floor us` column in the final table is the same probe run on each shape's own weight copies. The report's go/no-go numbers (4.19 MB <= 3.0 us, gate_up <= 5.5 us) sit below this floor by 0.4 us and 0.5 us respectively, so they cannot be met by any single kernel launched as a graph node; they were derived from bytes / peak bandwidth without the ~2.1 us per-kernel fixed cost. The thresholds are evaluated as written above; the fraction of the *measured* floor is the honest efficiency figure:

| shape | M | cuBLAS us | Triton us | stream floor us | Triton - floor (us) | cuBLAS - floor (us) | saved per call (us) |
|---|---|---|---|---|---|---|---|
| qkv | 1 | 5.78 | 5.10 | 4.71 | 0.39 | 1.07 | +0.67 |
| qkv | 2 | 5.82 | 5.70 | 4.70 | 1.00 | 1.12 | +0.12 |
| qkv | 4 | 5.82 | 5.72 | 4.67 | 1.05 | 1.15 | +0.10 |
| qkv | 8 | 5.91 | 5.72 | 4.68 | 1.04 | 1.23 | +0.19 |
| qkv | 48 | 6.10 | 6.10 | 4.71 | 1.39 | 1.39 | +0.00 |
| o_proj | 1 | 5.97 | 3.98 | 3.42 | 0.56 | 2.55 | +1.98 |
| o_proj | 2 | 5.97 | 4.41 | 3.42 | 0.99 | 2.56 | +1.56 |
| o_proj | 4 | 5.99 | 5.09 | 3.44 | 1.66 | 2.55 | +0.89 |
| o_proj | 8 | 6.01 | 6.38 | 3.42 | 2.96 | 2.59 | -0.37 |
| o_proj | 48 | 6.18 | 7.33 | 3.41 | 3.91 | 2.77 | -1.14 |
| o_proj_k1024 | 1 | 4.61 | 3.22 | 2.78 | 0.45 | 1.83 | +1.38 |
| o_proj_k1024 | 2 | 4.61 | 3.51 | 2.76 | 0.75 | 1.85 | +1.10 |
| o_proj_k1024 | 4 | 4.62 | 4.00 | 2.73 | 1.27 | 1.88 | +0.62 |
| o_proj_k1024 | 8 | 4.64 | 4.98 | 2.77 | 2.21 | 1.87 | -0.34 |
| o_proj_k1024 | 48 | 4.84 | 5.21 | 2.76 | 2.45 | 2.09 | -0.37 |
| gate_up | 1 | 7.16 | 6.48 | 6.03 | 0.45 | 1.13 | +0.68 |
| gate_up | 2 | 7.18 | 6.89 | 6.00 | 0.88 | 1.18 | +0.30 |
| gate_up | 4 | 7.23 | 6.92 | 6.02 | 0.90 | 1.21 | +0.31 |
| gate_up | 8 | 7.30 | 6.96 | 5.99 | 0.97 | 1.31 | +0.34 |
| gate_up | 48 | 7.65 | 7.75 | 5.98 | 1.77 | 1.67 | -0.10 |
| down | 1 | 7.78 | 4.73 | 4.05 | 0.68 | 3.73 | +3.05 |
| down | 2 | 7.84 | 5.50 | 4.07 | 1.43 | 3.77 | +2.35 |
| down | 4 | 7.81 | 6.86 | 4.06 | 2.80 | 3.75 | +0.95 |
| down | 8 | 7.81 | 7.30 | 4.06 | 3.24 | 3.75 | +0.52 |
| down | 48 | 8.29 | 8.49 | 4.07 | 4.42 | 4.22 | -0.20 |
| project_input | 1 | 5.20 | 3.85 | 3.43 | 0.41 | 1.77 | +1.35 |
| project_input | 2 | 5.18 | 4.40 | 3.44 | 0.96 | 1.74 | +0.78 |
| project_input | 4 | 5.22 | 4.93 | 3.42 | 1.51 | 1.80 | +0.29 |
| project_input | 8 | 5.21 | 5.00 | 3.42 | 1.58 | 1.79 | +0.21 |
| project_input | 48 | 5.41 | 5.29 | 3.41 | 1.88 | 2.00 | +0.12 |
| lm_head | 1 | 5.01 | 3.72 | 3.42 | 0.30 | 1.59 | +1.29 |
| lm_head | 2 | 4.99 | 4.27 | 3.42 | 0.85 | 1.57 | +0.72 |
| lm_head | 4 | 5.02 | 4.82 | 3.41 | 1.41 | 1.62 | +0.20 |
| lm_head | 8 | 5.03 | 4.82 | 3.41 | 1.41 | 1.62 | +0.21 |
| lm_head | 48 | 5.21 | 5.10 | 3.42 | 1.68 | 1.79 | +0.12 |

## Kernel design (gemv_bench.py)

Three Triton kernels, one autotuned choice per (shape, M); weights are torch `Linear` layout `[N, K]` row-major, x `[M, K]`, bf16 in, fp32 accumulate, bf16 out, epilogue (bias / residual add) fused.

- `gemv_reg_kernel` ("reg", CUDA cores, M <= 8): grid (N/BLOCK_N, SPLIT_K); each program loads its whole [BLOCK_N, K/SPLIT_K] weight tile with one vectorised load (K padded to a power of two and masked for K = 3072), converts to fp32 once, then for each row m multiplies by the x slice and reduces over K in registers + warp shuffles + one cross-warp exchange. The winning configs use SPLIT_K = 1 with BLOCK_N = 4..8 and 4..16 warps (128-1536 programs, 32-64 elements per thread), so nothing crosses the program boundary: no atomics, no counter, no fences, no scratch.
- `gemv_reg3d_kernel` ("reg3", CUDA cores, M <= 8): same tile, but the [M_PAD, BLOCK_N, K] broadcast product is reduced in one pass, so the cross-warp reduction is paid once instead of M times (the reg kernel's per-row reduction costs ~0.9 us each).
- `splitk_gemm_kernel` ("dot", tensor cores): grid (N/BLOCK_N, SPLIT_K); program streams the [BLOCK_N, K/SPLIT_K] slab through Triton's software pipeline (num_stages tiles of [BLOCK_N, BLOCK_K]) into `tl.dot` with BLOCK_M = 16 (M <= 16) or 64 (M = 48). SPLIT_K > 1 uses a fused reduction: partials go to fp32 scratch (REDUCE=0: `tl.atomic_add` into one [M, N] buffer; REDUCE=1: per-split slots, deterministic), every thread issues `fence.acq_rel.gpu`, the program increments a per-column-block arrival counter (acq_rel, gpu scope) and the last arriver reads the sum with volatile loads, applies the epilogue, and zeroes the scratch + counter it consumed, so one launch does the whole job and back-to-back graph nodes need no memset. This was chosen over a second pass because a second kernel node costs ~1.2 us of launch/drain by itself; the in-kernel chain costs ~2 us of serialised latency (fence -> counter round trip -> fixup loads), which is why the SPLIT_K = 1 configs win wherever the column count gives >= 64 programs. Stage-A pipeline depth is set from a 64 KB-in-flight target and refined over {2,3,4,6,8}.
- Autotune: per (shape, M), stage A sweeps every (kind, BLOCK_N, BLOCK_K, SPLIT_K) tile (dot with warps/stages/reduce at a middle value; reg/reg3 fully), stage B refines warps x stages x reduce for the top tiles; each candidate is first checked against the fp32 reference (tolerance 2 x cuBLAS max-abs + 0.01) and then timed with the same 20-call two-graph protocol (3 warm-up + 10 timed replays); numerically wrong or non-compiling configs are recorded and excluded. Results cached in `autotune_cache.json`; the full per-config table is in `results.json` (`autotune_tried`).

## Caveats

- GPU: host GPU 1 of eval-h100 exposed alone in task container `sglang-omni-jaxan-2` (the parent's container only exposes GPU 0); SM clock held 1980 MHz and HBM 2619 MHz during timing (sampled every second, n=110; the 345 MHz minimum in `results.json` is the idle clock before the first kernel). No application clocks were set (no root); the CI evict reaper was not active on lane 0,1 during the run. cuBLAS from torch 2.13.0+cu130 (CUDA 13.0, driver 580.126.20); `allow_bf16_reduced_precision_reduction=True` (torch default, as the predictor runs).
- Timing includes the inter-node gap: the graph median is (graph time / 20), so every number carries the ~1.2 us launch/drain between dependent graph nodes. The report's profiler durations exclude that gap, which is why cuBLAS o_proj reads 6.0 us here vs 5.50 us in the report.
- Weights are HBM-cold by construction (40 copies, two alternating graphs, > 3x L2 per cycle for the smallest shape); x, bias and residual are L2-hot, as in the predictor.
- The task text lists o_proj as 1024 -> 1024, but the predictor's o_proj is `NUM_HEADS * HEAD_DIM -> HIDDEN` = 2048 -> 1024 (4.19 MB, matching the report); `o_proj` here is the 2048 -> 1024 shape and `o_proj_k1024` is the literal 1024 -> 1024 (2.10 MB) for completeness. `o_proj` cuBLAS is the predictor's own path, in-place `torch.addmm(residual, x, W.t(), out=residual)`; `cublas_alt` is the plain `F.linear`. `down` cuBLAS is GEMM-only (the predictor fuses that residual into the following norm) while the Triton `down` includes the residual add, which favours cuBLAS slightly; `cublas_alt` for down is `F.linear + add`.
- Eager `torch.cuda.Event` timings are launch-bound (16-30 us single call: Python + driver launch cost; the Triton launcher is ~5-8 us slower than the aten call); the burst-20 numbers (GPU pre-loaded with a 2.7 ms matmul so launches queue up) are the meaningful eager cross-check and track the graph numbers +1.2-1.5 us, consistent with stream launch latency exceeding the graph gap.
- `dot` configs with BLOCK_N = 32, BLOCK_M = 64, 8 warps and atomic reduction produce deterministic wrong results (same error to 3 digits across num_stages, unchanged by the fences), i.e. a Triton 3.7.1 codegen issue with replicated wgmma layouts in `tl.atomic_add`, not a race; they are caught by the per-config numerics check and never selected. Configs whose pipelined tiles exceed 227 KB shared memory fail to compile and are recorded as errors.
- Split-K atomics (REDUCE=0) make the fp32 summation order run-dependent (bit-level nondeterminism, within the error budget); the selected configs for M <= 8 are all SPLIT_K = 1 and deterministic. Every Triton variant was validated on all 40 weight copies after the timed graph replays and the self-resetting scratch was verified to be all-zero afterwards (`scratch_clean_after_replays`).
- Not tried: TMA descriptors / warp-specialised persistent kernels, PDL (programmatic dependent launch, which is the only way to hide the ~1.2 us node-to-node gap), or fusing the 27 GEMMs of a sub-step into fewer launches; all of those attack the fixed cost rather than the streaming rate and are outside the C4 question as posed.
