"""Batch-invariant persistent Triton GEMM (port of vLLM's batch_invariant.py).

Used only in mode=batch_invariant, replacing F.linear. Row r of the output is
computed in tile r // BLOCK_M with a K-loop whose order never depends on M
(the token/batch dimension), so a given row's result is bitwise identical
whether it is computed alone or inside any larger batch. This is the
"deterministic but slow" baseline the LLM-42 paper argues against: it forgoes
cuBLAS's shape-adaptive kernel selection entirely.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

# Fixed configuration — never autotuned.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
GROUP_M = 8


@triton.jit
def _matmul_persistent_kernel(
    A, B, C, Bias,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    NUM_SMS: tl.constexpr, HAS_BIAS: tl.constexpr,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
):
    start_pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = tl.cdiv(N, BN)
    num_tiles = num_pid_m * num_pid_n
    k_tiles = tl.cdiv(K, BK)

    num_pid_in_group = GM * num_pid_n
    for tile_id in tl.range(start_pid, num_tiles, NUM_SMS):
        # Grouped tile ordering (fixed, M-independent within a row's tile).
        group_id = tile_id // num_pid_in_group
        first_pid_m = group_id * GM
        group_size_m = tl.minimum(num_pid_m - first_pid_m, GM)
        in_group = tile_id % num_pid_in_group
        pid_m = first_pid_m + (in_group % group_size_m)
        pid_n = in_group // group_size_m

        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_n = pid_n * BN + tl.arange(0, BN)
        mask_m = offs_m < M
        mask_n = offs_n < N
        # int64 pointer math: M * N (lm_head: N = vocab_size) can exceed 2^31.
        offs_m64 = offs_m.to(tl.int64)
        offs_n64 = offs_n.to(tl.int64)

        acc = tl.zeros([BM, BN], dtype=tl.float32)
        for kk in tl.range(0, k_tiles):
            offs_k = kk * BK + tl.arange(0, BK)
            mask_k = offs_k < K
            a = tl.load(A + offs_m64[:, None] * stride_am + offs_k[None, :] * stride_ak,
                        mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            b = tl.load(B + offs_k[:, None] * stride_bk + offs_n64[None, :] * stride_bn,
                        mask=mask_k[:, None] & mask_n[None, :], other=0.0)
            acc = tl.dot(a, b, acc)

        if HAS_BIAS:
            bias = tl.load(Bias + offs_n, mask=mask_n, other=0.0).to(tl.float32)
            acc = acc + bias[None, :]

        c_ptrs = C + offs_m64[:, None] * stride_cm + offs_n64[None, :] * stride_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def _num_sms() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


def matmul_persistent(a: torch.Tensor, b: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """C = A @ B (+ bias). A: [M, K], B: [K, N] (strided views are fine)."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty(M, N, device=a.device, dtype=a.dtype)
    num_sms = _num_sms()
    grid = (min(num_sms, triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)),)
    _matmul_persistent_kernel[grid](
        a, b, c, bias if bias is not None else a,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        NUM_SMS=num_sms, HAS_BIAS=bias is not None,
        BM=BLOCK_M, BN=BLOCK_N, BK=BLOCK_K, GM=GROUP_M,
        num_warps=8, num_stages=3,
    )
    return c


def bi_linear(x: torch.Tensor, weight: torch.Tensor,
              bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Drop-in for F.linear(x, weight, bias) with batch-invariant reduction.

    x: [..., K]; weight: [N, K] (HF layout) -> out [..., N].
    """
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    out = matmul_persistent(x2, weight.t(), bias)
    return out.view(*shape[:-1], weight.shape[0])
