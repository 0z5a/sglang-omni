#!/usr/bin/env python3
"""E3 / candidate C4 of the Qwen3-TTS headroom report: Triton split-K skinny GEMM vs cuBLAS
for the code-predictor GEMMs (bf16 weights, fp32 accumulate, bf16 out) at M in {1,2,4,8,48}.

Timing regime: 20 back-to-back calls captured in one CUDA graph; each call reads a distinct
weight copy (40 copies, two graphs alternated) so every call streams its weight from HBM,
which is the regime of the real predictor (5 layers x 16 sub-steps, ~165 MB per sub-step,
far above the 50 MB L2).
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.profiler import ProfilerActivity, profile

HERE = os.path.dirname(os.path.abspath(__file__))
PEAK_TBS = 3.35
COPIES = 40
CALLS = 20
DEV = torch.device("cuda")
PROPS = torch.cuda.get_device_properties(0)
NUM_SMS = PROPS.multi_processor_count

# name, K (in_features), N (out_features), epilogue
SHAPES = [
    ("qkv", 1024, 4096, "none"),
    ("o_proj", 2048, 1024, "res"),
    ("o_proj_k1024", 1024, 1024, "res"),
    ("gate_up", 1024, 6144, "none"),
    ("down", 3072, 1024, "res"),
    ("project_input", 1024, 2048, "bias"),
    ("lm_head", 1024, 2048, "none"),
]
MS = [1, 2, 4, 8, 48]
THRESH_US = {"o_proj": 3.0, "project_input": 3.0, "lm_head": 3.0, "gate_up": 5.5}


# --------------------------------------------------------------------------- Triton kernels
@triton.jit
def _gpu_fence(tok):
    """fence.acq_rel.gpu executed by every thread (Triton exposes no fence intrinsic)."""
    tl.inline_asm_elementwise("fence.acq_rel.gpu;\nmov.u32 $0, $1;", "=r,r", [tok], dtype=tl.int32, is_pure=False, pack=1)


@triton.jit
def _epilogue(total, bias_ptr, res_ptr, out_ptr, rm, rn, M, stride_rm, stride_om, has_bias, has_res):
    m_mask = rm[:, None] < M
    if has_bias:
        total += tl.load(bias_ptr + rn).to(tl.float32)[None, :]
    if has_res:
        total += tl.load(res_ptr + rm[:, None] * stride_rm + rn[None, :], mask=m_mask, other=0.0).to(tl.float32)
    tl.store(out_ptr + rm[:, None] * stride_om + rn[None, :], total.to(tl.bfloat16), mask=m_mask)


@triton.jit
def _epilogue_row(val, bias_ptr, res_ptr, out_ptr, rn, m, stride_rm, stride_om, has_bias, has_res):
    if has_bias:
        val += tl.load(bias_ptr + rn).to(tl.float32)
    if has_res:
        val += tl.load(res_ptr + m * stride_rm + rn).to(tl.float32)
    tl.store(out_ptr + m * stride_om + rn, val.to(tl.bfloat16))


@triton.jit(do_not_specialize=["M", "has_bias", "has_res"])
def splitk_gemm_kernel(
    x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr, acc_ptr, part_ptr, cnt_ptr,
    M, N, K, stride_xm, stride_wn, stride_rm, stride_om, has_bias, has_res,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr, REDUCE: tl.constexpr,
):
    """Tensor-core split-K GEMM: out[M, N] = x[M, K] @ w[N, K]^T (+ bias) (+ res).

    Grid = (N / BLOCK_N, SPLIT_K): program (pid_n, pid_k) streams the [BLOCK_N, K/SPLIT_K]
    slab of w through the software pipeline (num_stages tiles of [BLOCK_N, BLOCK_K] in
    flight) and multiplies it with the matching K-slice of x on tensor cores (BLOCK_M = 16
    for M <= 16, 64 for M = 48; padding rows are masked).
    SPLIT_K == 1: the program owns its output columns and runs the epilogue directly.
    SPLIT_K  > 1: partial sums are combined in fp32 by the last-arriving program of each
    column block (per-column-block arrival counter, acq_rel at gpu scope, with a
    fence.acq_rel.gpu on every thread on both sides), which also applies the epilogue and
    resets the scratch it consumed, so one launch does the whole job and back-to-back
    launches inside a CUDA graph need no memset.
      REDUCE == 0: partials are atomically added into one fp32 [M, N] buffer (L2 atomics);
                   the last program reads the sum and zeroes the tile.
      REDUCE == 1: each program stores its partial tile to its own fp32 slot and the last
                   program sums the SPLIT_K slots (deterministic, no data atomics).
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rm = tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    k_per_split = K // SPLIT_K
    k0 = pid_k * k_per_split
    m_mask = rm[:, None] < M
    x_ptrs = x_ptr + rm[:, None] * stride_xm + (k0 + rk)[None, :]
    w_ptrs = w_ptr + rn[:, None] * stride_wn + (k0 + rk)[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, k_per_split // BLOCK_K):
        x = tl.load(x_ptrs, mask=m_mask, other=0.0)
        w = tl.load(w_ptrs)
        acc = tl.dot(x, tl.trans(w), acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K

    if SPLIT_K == 1:
        _epilogue(acc, bias_ptr, res_ptr, out_ptr, rm, rn, M, stride_rm, stride_om, has_bias, has_res)
    else:
        if REDUCE == 0:
            acc_ptrs = acc_ptr + rm[:, None] * N + rn[None, :]
            tl.atomic_add(acc_ptrs, acc, mask=m_mask, sem="relaxed", scope="gpu")
        else:
            part_ptrs = part_ptr + pid_k * (BLOCK_M * N) + rm[:, None] * N + rn[None, :]
            tl.store(part_ptrs, acc, mask=m_mask)
        _gpu_fence(pid_n)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            _gpu_fence(pid_n)
            if REDUCE == 0:
                total = tl.load(acc_ptrs, mask=m_mask, other=0.0, volatile=True)
                tl.store(acc_ptrs, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32), mask=m_mask)
            else:
                total = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for s in tl.static_range(SPLIT_K):
                    total += tl.load(part_ptr + s * (BLOCK_M * N) + rm[:, None] * N + rn[None, :],
                                     mask=m_mask, other=0.0, volatile=True)
            tl.store(cnt_ptr + pid_n, 0)
            _epilogue(total, bias_ptr, res_ptr, out_ptr, rm, rn, M, stride_rm, stride_om, has_bias, has_res)


@triton.jit(do_not_specialize=["M", "has_bias", "has_res"])
def gemv_reg_kernel(
    x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr, acc_ptr, cnt_ptr,
    M, N, K, stride_xm, stride_rm, stride_om, has_bias, has_res,
    BLOCK_N: tl.constexpr, KS: tl.constexpr, SPLIT_K: tl.constexpr, M_MAX: tl.constexpr,
):
    """Register-resident GEMV for M <= M_MAX (CUDA cores, no shared-memory pipeline).

    Grid = (N / BLOCK_N, SPLIT_K): program (pid_n, pid_k) loads its whole [BLOCK_N, KS]
    weight tile with one vectorised load (KS = next power of two of K / SPLIT_K, masked
    when K / SPLIT_K is not a power of two), so every byte it will ever need is requested at
    kernel start, which is what a latency-bound 2-12 MB stream needs. For each of the M
    rows it multiplies by the x slice and reduces over K with in-register adds and warp
    shuffles. With SPLIT_K == 1 (the configs that win) nothing crosses the program
    boundary: no atomics, no fences, no arrival counter. With SPLIT_K > 1 partials are
    combined exactly as in splitk_gemm_kernel with REDUCE == 0.
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    k_per_split = K // SPLIT_K
    rk = pid_k * k_per_split + tl.arange(0, KS)
    k_mask = tl.arange(0, KS) < k_per_split
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=k_mask[None, :], other=0.0).to(tl.float32)
    if SPLIT_K == 1:
        for m in tl.static_range(M_MAX):
            if m < M:
                xm = tl.load(x_ptr + m * stride_xm + rk, mask=k_mask, other=0.0).to(tl.float32)
                _epilogue_row(tl.sum(w * xm[None, :], axis=1), bias_ptr, res_ptr, out_ptr, rn, m, stride_rm, stride_om, has_bias, has_res)
    else:
        for m in tl.static_range(M_MAX):
            if m < M:
                xm = tl.load(x_ptr + m * stride_xm + rk, mask=k_mask, other=0.0).to(tl.float32)
                tl.atomic_add(acc_ptr + m * N + rn, tl.sum(w * xm[None, :], axis=1), sem="relaxed", scope="gpu")
        _gpu_fence(pid_n)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem="acq_rel", scope="gpu")
        if prev == SPLIT_K - 1:
            _gpu_fence(pid_n)
            for m in tl.static_range(M_MAX):
                if m < M:
                    total = tl.load(acc_ptr + m * N + rn, volatile=True)
                    tl.store(acc_ptr + m * N + rn, tl.zeros((BLOCK_N,), dtype=tl.float32))
                    _epilogue_row(total, bias_ptr, res_ptr, out_ptr, rn, m, stride_rm, stride_om, has_bias, has_res)
            tl.store(cnt_ptr + pid_n, 0)


@triton.jit(do_not_specialize=["M", "has_bias", "has_res"])
def gemv_reg3d_kernel(
    x_ptr, w_ptr, bias_ptr, res_ptr, out_ptr,
    M, N, K, stride_xm, stride_rm, stride_om, has_bias, has_res,
    BLOCK_N: tl.constexpr, KS: tl.constexpr, M_PAD: tl.constexpr,
):
    """Register-resident GEMV for M <= 8 with a single reduction pass.

    Grid = (N / BLOCK_N,): the program loads its whole [BLOCK_N, K] weight tile (one
    vectorised, K-masked load) and the [M, K] activation tile, forms the broadcast product
    [M_PAD, BLOCK_N, K] and reduces over K once (in-thread adds, then shuffles, then one
    cross-warp exchange), so the per-row reduction cost of gemv_reg_kernel is paid once
    instead of M times. No split-K: nothing crosses the program boundary.
    """
    pid_n = tl.program_id(0)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, KS)
    rm = tl.arange(0, M_PAD)
    k_mask = rk < K
    w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], mask=k_mask[None, :], other=0.0)
    x = tl.load(x_ptr + rm[:, None] * stride_xm + rk[None, :], mask=(rm[:, None] < M) & k_mask[None, :], other=0.0)
    y = tl.sum(x[:, None, :].to(tl.float32) * w[None, :, :].to(tl.float32), axis=2)
    _epilogue(y, bias_ptr, res_ptr, out_ptr, rm, rn, M, stride_rm, stride_om, has_bias, has_res)


@triton.jit
def stream_read_kernel(w_ptr, out_ptr, BLOCK: tl.constexpr):
    """Reads BLOCK bf16 elements and writes one float: the measured streaming floor."""
    pid = tl.program_id(0)
    w = tl.load(w_ptr + pid * BLOCK + tl.arange(0, BLOCK))
    tl.store(out_ptr + pid, tl.sum(w.to(tl.float32)))


@dataclass(frozen=True)
class Cfg:
    kind: str  # "dot" (splitk_gemm_kernel) or "reg" (gemv_reg_kernel; BK is the per-program K slice)
    BN: int
    BK: int
    SK: int
    warps: int
    stages: int
    red: int

    def key(self):
        if self.kind == "reg":
            return f"reg_BN{self.BN}_KS{self.BK}_SK{self.SK}_w{self.warps}"
        if self.kind == "reg3":
            return f"reg3_BN{self.BN}_KS{self.BK}_w{self.warps}"
        return f"dot_BN{self.BN}_BK{self.BK}_SK{self.SK}_w{self.warps}_s{self.stages}_r{self.red}"

    def ctas(self, N):
        return (N // self.BN) * self.SK


class Workspace:
    def __init__(self, n_max):
        self.acc = torch.zeros(64 * n_max, dtype=torch.float32, device=DEV)
        self.part = torch.empty(16 * 64 * n_max, dtype=torch.float32, device=DEV)
        self.cnt = torch.zeros(4096, dtype=torch.int32, device=DEV)
        self.scratch = torch.empty(1 << 16, dtype=torch.float32, device=DEV)

    def reset(self):
        self.acc.zero_()
        self.cnt.zero_()

    def clean(self):
        return bool((self.acc == 0).all().item()) and bool((self.cnt == 0).all().item())


def block_m(M):
    return 16 if M <= 16 else 64


def triton_linear(x, w, bias, res, out, ws, cfg):
    M, K = x.shape
    N = w.shape[0]
    grid = (N // cfg.BN, cfg.SK)
    if cfg.kind == "reg3":
        gemv_reg3d_kernel[(N // cfg.BN,)](
            x, w, bias if bias is not None else out, res if res is not None else out, out,
            M, N, K, x.stride(0), res.stride(0) if res is not None else 0, out.stride(0),
            1 if bias is not None else 0, 1 if res is not None else 0,
            BLOCK_N=cfg.BN, KS=cfg.BK, M_PAD=max(1, 1 << (M - 1).bit_length()), num_warps=cfg.warps,
        )
        return
    if cfg.kind == "reg":
        gemv_reg_kernel[grid](
            x, w, bias if bias is not None else out, res if res is not None else out, out, ws.acc, ws.cnt,
            M, N, K, x.stride(0), res.stride(0) if res is not None else 0, out.stride(0),
            1 if bias is not None else 0, 1 if res is not None else 0,
            BLOCK_N=cfg.BN, KS=cfg.BK, SPLIT_K=cfg.SK, M_MAX=8, num_warps=cfg.warps,
        )
        return
    splitk_gemm_kernel[grid](
        x, w, bias if bias is not None else out, res if res is not None else out, out,
        ws.acc, ws.part, ws.cnt,
        M, N, K, x.stride(0), w.stride(0), res.stride(0) if res is not None else 0, out.stride(0),
        1 if bias is not None else 0, 1 if res is not None else 0,
        BLOCK_M=block_m(M), BLOCK_N=cfg.BN, BLOCK_K=cfg.BK, SPLIT_K=cfg.SK, REDUCE=cfg.red,
        num_warps=cfg.warps, num_stages=cfg.stages,
    )


def allowed_warps(BM, BN, BK):
    out = []
    for w in (2, 4, 8):
        elems_per_thread = BN * BK / (32 * w)
        if 16 <= elems_per_thread <= 128 and BM * BN / w >= 128:
            out.append(w)
    return out


def tile_configs(K, N, M):
    """Stage A: every (kind, BN, BK, SK) tile; for the dot kernel warps/stages/reduce are fixed
    at a middle value and refined in stage B, the reg kernel has only warps left and is
    swept fully here."""
    BM = block_m(M)
    out = []
    for BN in (16, 32, 64, 128):
        for BK in (32, 64, 128, 256):
            for SK in (1, 2, 4, 8, 16):
                if K % SK or (K // SK) % BK:
                    continue
                ctas = (N // BN) * SK
                if ctas < 60 or ctas > 8 * NUM_SMS:
                    continue
                warps = allowed_warps(BM, BN, BK)
                if not warps:
                    continue
                w = 4 if 4 in warps else warps[-1]
                trips = (K // SK) // BK
                # note (luojiaxuan): pipeline depth that keeps ~64 KB of the slab in flight per program
                stages = max(2, min(8, trips, (64 * 1024) // ((BN + BM) * BK * 2)))
                out.append(Cfg("dot", BN, BK, SK, w, stages, 1 if BM == 64 else 0))
    if M <= 8:
        for BN in (4, 8, 16, 32, 64):
            for SK in (1, 2, 3, 4, 6, 8, 12, 16, 24):
                if K % SK:
                    continue
                KS = 1 << (K // SK - 1).bit_length()  # next power of two of the K slice
                ctas = (N // BN) * SK
                if ctas < NUM_SMS // 2 or ctas > 32 * NUM_SMS:
                    continue
                for w in (2, 4, 8, 16):
                    if 8 <= BN * KS / (32 * w) <= 64:
                        out.append(Cfg("reg", BN, KS, SK, w, 0, 0))
        KS = 1 << (K - 1).bit_length()
        for BN in (2, 4, 8, 16, 32):
            if N // BN < 60:
                continue
            for w in (4, 8, 16):
                if 8 <= BN * KS / (32 * w) <= 64:
                    out.append(Cfg("reg3", BN, KS, 1, w, 0, 0))
    return out


def refine_configs(K, N, M, tile):
    if tile.kind != "dot":
        return []
    BM = block_m(M)
    trips = (K // tile.SK) // tile.BK
    out = []
    for w in allowed_warps(BM, tile.BN, tile.BK):
        for stages in ((2, 3, 4, 6, 8) if trips > 2 else (2,)):
            if stages > trips or (tile.BN + BM) * tile.BK * 2 * stages > 200 * 1024:
                continue
            for red in ((0,) if tile.SK == 1 else (0, 1)):
                out.append(Cfg("dot", tile.BN, tile.BK, tile.SK, w, stages, red))
    return out


# --------------------------------------------------------------------------- data + variants
class Data:
    def __init__(self, name, K, N, M, epi):
        self.name, self.K, self.N, self.M, self.epi = name, K, N, M, epi
        g = torch.Generator(device=DEV).manual_seed(1234)
        self.x = torch.randn(M, K, device=DEV, generator=g).to(torch.bfloat16)
        self.w = [(torch.randn(N, K, device=DEV, generator=g) / math.sqrt(K)).to(torch.bfloat16) for _ in range(COPIES)]
        self.bias = torch.randn(N, device=DEV, generator=g).to(torch.bfloat16) if epi == "bias" else None
        self.res0 = [torch.randn(M, N, device=DEV, generator=g).to(torch.bfloat16) for _ in range(COPIES)] if epi == "res" else None
        self.res = [r.clone() for r in self.res0] if epi == "res" else None
        self.outs = [torch.empty(M, N, dtype=torch.bfloat16, device=DEV) for _ in range(COPIES)]
        self.weight_bytes = N * K * 2

    def restore_res(self):
        if self.res is not None:
            for r, r0 in zip(self.res, self.res0):
                r.copy_(r0)

    def ref(self, i, with_epi=True):
        out = self.x.float() @ self.w[i].float().t()
        if with_epi and self.bias is not None:
            out = out + self.bias.float()
        if with_epi and self.res0 is not None:
            out = out + self.res0[i].float()
        return out


def make_variants(d, ws, cfg):
    """Returns {variant_name: (call(i), with_epi, output(i))}; call(i) uses weight copy i and
    output(i) is the tensor that call i (or its graph replay) wrote."""
    v = {}
    held = {}

    def cublas_linear(i):
        held[("lin", i)] = F.linear(d.x, d.w[i])
        return held[("lin", i)]

    def cublas_linear_bias(i):
        held[("linb", i)] = F.linear(d.x, d.w[i], d.bias)
        return held[("linb", i)]

    def cublas_addmm_inplace(i):
        # note (luojiaxuan): the predictor's o_proj path, torch.addmm(residual, x, W.t(), out=residual)
        torch.addmm(d.res[i], d.x, d.w[i].t(), out=d.res[i])
        return d.res[i]

    def cublas_linear_add(i):
        held[("lina", i)] = F.linear(d.x, d.w[i]) + d.res[i]
        return held[("lina", i)]

    def triton_call(i):
        triton_linear(d.x, d.w[i], d.bias, d.res[i] if d.res is not None else None, d.outs[i], ws, cfg)
        return d.outs[i]

    def stream_floor(i):
        stream_read_kernel[(d.N * d.K // 2048,)](d.w[i], ws.scratch, BLOCK=2048, num_warps=4)
        return d.outs[i]

    if d.epi == "none":
        v["cublas"] = (cublas_linear, True, lambda i: held[("lin", i)])
    elif d.epi == "bias":
        v["cublas"] = (cublas_linear_bias, True, lambda i: held[("linb", i)])
    elif d.name.startswith("o_proj"):
        v["cublas"] = (cublas_addmm_inplace, True, lambda i: d.res[i])
        v["cublas_alt"] = (cublas_linear, False, lambda i: held[("lin", i)])
    else:  # down: predictor fuses the residual into the following norm, so GEMM-only is the reference
        v["cublas"] = (cublas_linear, False, lambda i: held[("lin", i)])
        v["cublas_alt"] = (cublas_linear_add, True, lambda i: held[("lina", i)])
    if cfg is not None:
        v["triton"] = (triton_call, True, lambda i: d.outs[i])
    v["stream_floor"] = (stream_floor, None, None)
    return v


# --------------------------------------------------------------------------- timing
def bench_graph(call, warm, reps):
    graphs = []
    for half in (0, 1):
        idx = range(half * CALLS, (half + 1) * CALLS)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in idx:
                call(i)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in idx:
                call(i)
        graphs.append(g)
    for r in range(warm):
        graphs[r % 2].replay()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(reps)]
    for r in range(reps):
        starts[r].record()
        graphs[r % 2].replay()
        ends[r].record()
    torch.cuda.synchronize()
    per_call = sorted(s.elapsed_time(e) * 1e3 / CALLS for s, e in zip(starts, ends))
    stats = {
        "median_us": statistics.median(per_call),
        "min_us": per_call[0],
        "p90_us": per_call[int(0.9 * (len(per_call) - 1))],
        "max_us": per_call[-1],
        "n_replays": reps,
    }
    return stats, graphs


_BUSY = {}


def busy_kernel():
    if "a" not in _BUSY:
        _BUSY["a"] = torch.randn(4096, 4096, device=DEV)
        _BUSY["b"] = torch.randn(4096, 4096, device=DEV)
    torch.mm(_BUSY["a"], _BUSY["b"])


def bench_eager(call):
    single = []
    for i in range(COPIES):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        call(i)
        e.record()
        torch.cuda.synchronize()
        single.append(s.elapsed_time(e) * 1e3)
    burst = []
    for r in range(6):
        busy_kernel()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for i in range((r % 2) * CALLS, (r % 2 + 1) * CALLS):
            call(i)
        e.record()
        torch.cuda.synchronize()
        burst.append(s.elapsed_time(e) * 1e3 / CALLS)
    return {"eager_single_us": statistics.median(single), "eager_burst_us": statistics.median(burst)}


def err_stats(out, ref):
    d = (out.float() - ref).abs()
    return {
        "max_abs": d.max().item(),
        "rel_max": (d.max() / ref.abs().max()).item(),
        "rms_rel": (d.norm() / ref.norm()).item(),
        "has_nan": bool(torch.isnan(out.float()).any().item()),
    }


def kernel_names(call, n=3):
    call(0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for i in range(1, n + 1):
            call(i)
        torch.cuda.synchronize()
    names = {}
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA:
            names.setdefault(ev.name, []).append(ev.time_range.elapsed_us())
    return {k: {"count": len(v), "mean_us": statistics.mean(v)} for k, v in names.items()}


class ClockSampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_throttle_reasons.active",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
            self.samples.append((time.time(), out))
            self.stop.wait(1.0)

    def summary(self):
        sm = [int(s.split(",")[0]) for _, s in self.samples if s]
        mem = [int(s.split(",")[1]) for _, s in self.samples if s]
        return {"n": len(sm), "sm_mhz_min": min(sm), "sm_mhz_max": max(sm), "mem_mhz_min": min(mem), "mem_mhz_max": max(mem),
                "last": self.samples[-1][1] if self.samples else ""}


# --------------------------------------------------------------------------- autotune
def autotune(d, ws, tol, cache, log):
    key = f"{d.name}|K{d.K}|N{d.N}|M{d.M}"
    if key in cache:
        return Cfg(**cache[key]["best"]), cache[key]
    tried = {}

    def evaluate(cfg):
        call = make_variants(d, ws, cfg)["triton"][0]
        try:
            d.restore_res()
            ws.reset()
            call(0)
            torch.cuda.synchronize()
            e = err_stats(d.outs[0], d.ref(0))
            if e["has_nan"] or e["max_abs"] > tol:
                tried[cfg.key()] = {"status": "numerics", "max_abs": e["max_abs"]}
                log(f"    {cfg.key()}: NUMERICS max_abs={e['max_abs']:.4g} tol={tol:.4g}")
                return None
            stats, graphs = bench_graph(call, warm=3, reps=10)
            del graphs
            tried[cfg.key()] = {"status": "ok", "median_us": stats["median_us"]}
            return stats["median_us"]
        except Exception as ex:  # Triton compile/launch failure for this config; recorded, not fatal
            tried[cfg.key()] = {"status": "error", "error": f"{type(ex).__name__}: {str(ex)[:160]}"}
            log(f"    {cfg.key()}: ERROR {type(ex).__name__}: {str(ex)[:160]}")
            return None

    t0 = time.time()
    stage_a = []
    for cfg in tile_configs(d.K, d.N, d.M):
        t = evaluate(cfg)
        if t is not None:
            stage_a.append((t, cfg))
    stage_a.sort(key=lambda p: p[0])
    best_t, best = stage_a[0]
    for _, tile in stage_a[:3 if d.M > 8 else 2]:
        for cfg in refine_configs(d.K, d.N, d.M, tile):
            if cfg.key() in tried:
                continue
            t = evaluate(cfg)
            if t is not None and t < best_t:
                best_t, best = t, cfg
    info = {"best": asdict(best), "best_us_autotune": best_t, "n_tried": len(tried), "seconds": time.time() - t0,
            "stage_a_top3": [(t, c.key()) for t, c in stage_a[:3]], "tried": tried}
    cache[key] = info
    log(f"  autotune {key}: best {best.key()} {best_t:.3f} us ({len(tried)} configs, {info['seconds']:.0f}s)")
    return best, info


# --------------------------------------------------------------------------- main
def run_case(d, ws, cache, log, warm, reps):
    d.restore_res()
    base = make_variants(d, ws, None)
    cublas_err = err_stats(base["cublas"][0](0), d.ref(0, base["cublas"][1]))
    d.restore_res()
    tol = 2 * cublas_err["max_abs"] + 0.01
    cfg, tune = autotune(d, ws, tol, cache, log)
    variants = make_variants(d, ws, cfg)
    rows = {}
    for vname, (call, with_epi, output) in variants.items():
        if vname == "stream_floor":
            stats, graphs = bench_graph(call, warm=warm, reps=reps)
            del graphs
            rows[vname] = dict(stats, tbs=d.weight_bytes / (stats["median_us"] * 1e-6) / 1e12)
            rows[vname]["frac_peak"] = rows[vname]["tbs"] / PEAK_TBS
            log(f"  {d.name} M={d.M} stream_floor graph {stats['median_us']:.3f} us -> {rows[vname]['tbs']:.2f} TB/s")
            continue
        d.restore_res()
        out0 = call(0)
        torch.cuda.synchronize()
        err = err_stats(out0, d.ref(0, with_epi))
        d.restore_res()
        stats, graphs = bench_graph(call, warm=warm, reps=reps)
        d.restore_res()
        graphs[0].replay()
        graphs[1].replay()
        torch.cuda.synchronize()
        replay_err = max(err_stats(output(i), d.ref(i, with_epi))["max_abs"] for i in range(COPIES))
        scratch_clean = ws.clean() if vname == "triton" else None
        del graphs
        d.restore_res()
        eager = bench_eager(call)
        d.restore_res()
        row = dict(stats)
        row.update(eager)
        row["err_eager"] = err
        row["err_graph_replay_max_abs_over_40_copies"] = replay_err
        row["scratch_clean_after_replays"] = scratch_clean
        row["tbs"] = d.weight_bytes / (stats["median_us"] * 1e-6) / 1e12
        row["frac_peak"] = row["tbs"] / PEAK_TBS
        row["with_epilogue"] = with_epi
        if vname.startswith("cublas"):
            d.restore_res()
            row["kernels"] = kernel_names(call)
            d.restore_res()
        rows[vname] = row
        log(f"  {d.name} M={d.M} {vname:10s} graph {stats['median_us']:.3f} us (min {stats['min_us']:.3f}) "
            f"eager single {eager['eager_single_us']:.2f} burst {eager['eager_burst_us']:.3f} | "
            f"{row['tbs']:.3f} TB/s ({100*row['frac_peak']:.1f}%) | max_abs {err['max_abs']:.4g} replay {replay_err:.4g} clean {scratch_clean}")
    return {"shape": d.name, "K": d.K, "N": d.N, "M": d.M, "epilogue": d.epi, "weight_bytes": d.weight_bytes,
            "floor_us": d.weight_bytes / (PEAK_TBS * 1e12) * 1e6, "triton_cfg": asdict(cfg),
            "autotune": {k: v for k, v in tune.items() if k != "tried"}, "autotune_tried": tune["tried"],
            "variants": rows}


def evaluate_thresholds(results):
    def get(shape, M, v):
        for r in results:
            if r["shape"] == shape and r["M"] == M:
                return r["variants"][v]["median_us"]
    o_proj_class = all(get(s, 1, "triton") <= 3.0 for s in ("o_proj", "project_input", "lm_head"))
    gate_up = get("gate_up", 1, "triton") <= 5.5
    m48 = all(r["variants"]["triton"]["median_us"] <= r["variants"]["cublas"]["median_us"] for r in results if r["M"] == 48)
    numerics = all(
        (not r["variants"]["triton"]["err_eager"]["has_nan"])
        and r["variants"]["triton"]["err_eager"]["max_abs"] <= 2 * r["variants"]["cublas"]["err_eager"]["max_abs"] + 0.01
        and r["variants"]["triton"]["err_graph_replay_max_abs_over_40_copies"] <= 2 * r["variants"]["cublas"]["err_eager"]["max_abs"] + 0.01
        and r["variants"]["triton"]["scratch_clean_after_replays"]
        for r in results)
    return {"o_proj_le_3us": o_proj_class, "gate_up_le_5p5us": gate_up, "m48_not_slower": m48, "numerics_ok": numerics}


def write_md(results, meta, thresholds, path):
    L = []
    L.append("# E3 / C4: Triton split-K skinny GEMM vs cuBLAS (Qwen3-TTS code predictor shapes)\n")
    L.append(f"GPU: {meta['gpu']} ({NUM_SMS} SMs, L2 {meta['l2_mb']:.0f} MB), torch {meta['torch']}, triton {meta['triton']}, "
             f"CUDA {meta['cuda']}, driver {meta['driver']}. Peak HBM assumed {PEAK_TBS} TB/s. "
             f"Timing: median per-call us from a CUDA graph of {CALLS} back-to-back calls, {meta['reps']} timed replays after {meta['warm']} warm-up "
             f"replays, weights rotated over {COPIES} copies (two graphs alternated) so each call is HBM-cold.\n")
    L.append(f"Clocks during the run (nvidia-smi, 1 s samples, n={meta['clocks']['n']}): SM {meta['clocks']['sm_mhz_min']}-{meta['clocks']['sm_mhz_max']} MHz, "
             f"HBM {meta['clocks']['mem_mhz_min']}-{meta['clocks']['mem_mhz_max']} MHz; last sample: `{meta['clocks']['last']}`.\n")
    L.append("## Final table\n")
    L.append("| shape (K->N, MB) | M | cuBLAS us | Triton us | speedup | cuBLAS TB/s (%peak) | Triton TB/s (%peak) | cuBLAS max abs err | Triton max abs err | threshold | pass | stream floor us |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        c, t = r["variants"]["cublas"], r["variants"]["triton"]
        if r["M"] == 48:
            thr, ok = "<= cuBLAS", t["median_us"] <= c["median_us"]
        elif r["shape"] in THRESH_US:
            thr, ok = f"<= {THRESH_US[r['shape']]} us", t["median_us"] <= THRESH_US[r["shape"]]
        else:
            thr, ok = "n/a", None
        L.append(f"| {r['shape']} ({r['K']}->{r['N']}, {r['weight_bytes']/1e6:.2f}) | {r['M']} | {c['median_us']:.2f} | {t['median_us']:.2f} | "
                 f"{c['median_us']/t['median_us']:.2f}x | {c['tbs']:.2f} ({100*c['frac_peak']:.0f}%) | {t['tbs']:.2f} ({100*t['frac_peak']:.0f}%) | "
                 f"{c['err_eager']['max_abs']:.4f} | {t['err_eager']['max_abs']:.4f} | {thr} | {'-' if ok is None else ('PASS' if ok else 'FAIL')} | {r['variants']['stream_floor']['median_us']:.2f} |")
    L.append("")
    L.append(f"Thresholds: {json.dumps(thresholds)}\n")
    L.append("## Cross-checks (eager torch.cuda.Event timing) and secondary cuBLAS variants\n")
    L.append("| shape | M | variant | graph median us | graph min us | eager single us | eager burst-20 us | replay max abs err (40 copies) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in results:
        for v, row in r["variants"].items():
            if v == "stream_floor":
                continue
            L.append(f"| {r['shape']} | {r['M']} | {v} | {row['median_us']:.2f} | {row['min_us']:.2f} | {row['eager_single_us']:.2f} | "
                     f"{row['eager_burst_us']:.2f} | {row['err_graph_replay_max_abs_over_40_copies']:.4f} |")
    L.append("")
    L.append("## Triton config chosen per shape (autotuned per (shape, M))\n")
    L.append("| shape | M | config | CTAs | autotune best us | configs tried | stage-A top3 | best dot kernel | best reg / reg3 kernel |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in results:
        cfg = Cfg(**r["triton_cfg"])
        best_kind = {}
        for k, v in r["autotune_tried"].items():
            if v["status"] == "ok":
                kind = k.split("_")[0]
                if kind not in best_kind or v["median_us"] < best_kind[kind][1]:
                    best_kind[kind] = (k, v["median_us"])
        fmt = lambda kind: f"{best_kind[kind][0]} {best_kind[kind][1]:.2f}" if kind in best_kind else "-"
        L.append(f"| {r['shape']} | {r['M']} | {cfg.key()} | {cfg.ctas(r['N'])} | {r['autotune']['best_us_autotune']:.3f} | {r['autotune']['n_tried']} | "
                 f"{', '.join(f'{k} {t:.2f}' for t, k in r['autotune']['stage_a_top3'])} | {fmt('dot')} | {fmt('reg')} / {fmt('reg3')} |")
    L.append("")
    L.append("## cuBLAS kernels observed (torch.profiler over eager calls of the predictor path)\n")
    for r in results:
        for v, row in r["variants"].items():
            if "kernels" in row:
                ks = "; ".join(f"`{k}` x{info['count']} {info['mean_us']:.2f}us" for k, info in row["kernels"].items())
                L.append(f"- {r['shape']} M={r['M']} {v}: {ks}")
    L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--shapes", default="")
    ap.add_argument("--ms", default="")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "smoke.log" if args.smoke else "bench.log")
    logf = open(log_path, "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        logf.write(line + "\n")
        logf.flush()

    shapes = [s for s in SHAPES if not args.shapes or s[0] in args.shapes.split(",")]
    ms = [int(m) for m in args.ms.split(",")] if args.ms else MS
    if args.smoke:
        shapes = [s for s in SHAPES if s[0] in ("o_proj", "down")]
        ms = [1, 8, 48]
        args.reps, args.warm = 10, 3
    cache_path = os.path.join(args.out, "autotune_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) and not args.smoke else {}
    meta = {
        "gpu": PROPS.name, "sms": NUM_SMS, "l2_mb": PROPS.L2_cache_size / 1e6, "torch": torch.__version__,
        "triton": triton.__version__, "cuda": torch.version.cuda, "driver": subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip(),
        "reps": args.reps, "warm": args.warm, "copies": COPIES, "calls_per_graph": CALLS, "peak_tbs": PEAK_TBS,
        "cublas_allow_bf16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
    }
    sampler = ClockSampler()
    sampler.start()
    ws = Workspace(max(s[2] for s in SHAPES))
    log(f"GPU {PROPS.name} SMs={NUM_SMS} torch={torch.__version__} triton={triton.__version__} shapes={[s[0] for s in shapes]} ms={ms}")
    results = []
    t_start = time.time()
    for name, K, N, epi in shapes:
        for M in ms:
            d = Data(name, K, N, M, epi)
            log(f"== {name} K={K} N={N} M={M} epi={epi} weight {d.weight_bytes/1e6:.2f} MB floor {d.weight_bytes/(PEAK_TBS*1e12)*1e6:.2f} us")
            results.append(run_case(d, ws, cache, log, args.warm, args.reps))
            if not args.smoke:
                json.dump(cache, open(cache_path, "w"), indent=1)
            del d
            torch.cuda.empty_cache()
    sampler.stop.set()
    sampler.join()
    meta["clocks"] = sampler.summary()
    meta["wall_seconds"] = time.time() - t_start
    thresholds = evaluate_thresholds(results) if not args.smoke and set(s[0] for s in shapes) >= {"o_proj", "project_input", "lm_head", "gate_up"} and 1 in ms else {}
    tag = "smoke_" if args.smoke else ""
    json.dump({"meta": meta, "thresholds": thresholds, "results": results}, open(os.path.join(args.out, f"{tag}results.json"), "w"), indent=1)
    write_md(results, meta, thresholds, os.path.join(args.out, f"{tag}results.md"))
    log(f"done in {meta['wall_seconds']:.0f}s; thresholds {thresholds}; clocks {meta['clocks']}")


if __name__ == "__main__":
    main()
