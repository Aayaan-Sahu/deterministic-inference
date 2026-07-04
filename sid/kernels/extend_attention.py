"""Varlen causal attention over a paged prefix ("extend" attention).

Used for prefill (prefix_len == 0) and for DVR verification passes
(prefix_len > 0, extend_len == window size). Simplified port of sglang's
triton_ops/extend_attention.py with everything nonessential stripped
(no sliding window, custom masks, MLA, sinks, logit caps).

DETERMINISM: tile constants are HARDCODED — no @triton.autotune, no
arch-conditional block sizes. Q tiles are per-request (indexed off qo_indptr),
so a request's softmax reduction covers only its own rows and its own KV;
packing order and co-batched requests cannot change its result. Combined with
the fixed tiles, the output for a request depends only on (its q/k/v values,
its prefix length, its extend length).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fixed tile configuration (do not autotune — see module docstring).
BLOCK_M = 64
BLOCK_N = 64
NUM_WARPS = 4
# Referenced inside the @jit kernel: must be tl.constexpr instances
# (plain module globals are rejected by Triton >= 3.x).
NUM_STAGES = tl.constexpr(2)
NEG_INF = tl.constexpr(-1e30)


@triton.jit
def _extend_attn_kernel(
    Q_extend, K_extend, V_extend, O_extend,   # [Text, H*, D] bf16
    K_buffer, V_buffer,                        # [S, HKV, D] bf16 (paged pool)
    qo_indptr, kv_indptr, kv_indices,          # int32/int64
    sm_scale,
    stride_qh: tl.constexpr, stride_kh: tl.constexpr,   # head strides (== D)
    NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
    BLOCK_M_C: tl.constexpr, BLOCK_N_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    kv_head = pid_h // (NUM_Q_HEADS // NUM_KV_HEADS)

    q_start = tl.load(qo_indptr + pid_b)
    q_end = tl.load(qo_indptr + pid_b + 1)
    extend_len = q_end - q_start
    if pid_m * BLOCK_M_C >= extend_len:
        return

    kv_start = tl.load(kv_indptr + pid_b)
    kv_end = tl.load(kv_indptr + pid_b + 1)
    prefix_len = kv_end - kv_start

    offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < extend_len

    q_ptrs = Q_extend + (q_start + offs_m[:, None]) * (NUM_Q_HEADS * stride_qh) \
        + pid_h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M_C], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M_C], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_C, BLOCK_DMODEL], dtype=tl.float32)

    # ---- stage 1: attend over the paged prefix (fully visible).
    for start_n in tl.range(0, prefix_len, BLOCK_N_C, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N_C)
        mask_n = offs_n < prefix_len
        slots = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)
        k_ptrs = K_buffer + slots[:, None].to(tl.int64) * (NUM_KV_HEADS * stride_kh) \
            + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        rescale = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None]

        v_ptrs = V_buffer + slots[:, None].to(tl.int64) * (NUM_KV_HEADS * stride_kh) \
            + kv_head * stride_kh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    # ---- stage 2: attend over extend tokens (causal).
    end_n = tl.minimum((pid_m + 1) * BLOCK_M_C, extend_len)
    for start_n in tl.range(0, end_n, BLOCK_N_C, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N_C)
        mask_n = offs_n < extend_len
        k_ptrs = K_extend + (q_start + offs_n[:, None]) * (NUM_KV_HEADS * stride_kh) \
            + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        causal = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal & mask_m[:, None] & mask_n[None, :], qk, NEG_INF)

        m_new = tl.maximum(m_i, tl.max(qk, 1))
        rescale = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * rescale + tl.sum(p, 1)
        acc = acc * rescale[:, None]

        v_ptrs = V_extend + (q_start + offs_n[:, None]) * (NUM_KV_HEADS * stride_kh) \
            + kv_head * stride_kh + offs_d[None, :]
        v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    out = acc / l_i[:, None]
    o_ptrs = O_extend + (q_start + offs_m[:, None]) * (NUM_Q_HEADS * stride_qh) \
        + pid_h * stride_qh + offs_d[None, :]
    tl.store(o_ptrs, out.to(O_extend.dtype.element_ty), mask=mask_m[:, None])


def extend_attention_fwd(
    q_extend: torch.Tensor,   # [Text, HQ, D] bf16
    k_extend: torch.Tensor,   # [Text, HKV, D] bf16
    v_extend: torch.Tensor,   # [Text, HKV, D] bf16
    k_buffer: torch.Tensor,   # [S, HKV, D] bf16
    v_buffer: torch.Tensor,   # [S, HKV, D] bf16
    qo_indptr: torch.Tensor,  # int32 [B+1]
    kv_indptr: torch.Tensor,  # int32 [B+1]  (prefix lengths)
    kv_indices: torch.Tensor, # int32
    max_extend_len: int,
    sm_scale: float,
) -> torch.Tensor:
    Text, num_q_heads, head_dim = q_extend.shape
    num_kv_heads = k_extend.shape[1]
    assert q_extend.is_contiguous() and k_extend.is_contiguous() and v_extend.is_contiguous()
    assert k_buffer.is_contiguous() and v_buffer.is_contiguous()

    o_extend = torch.empty_like(q_extend)
    bs = qo_indptr.numel() - 1
    grid = (bs, num_q_heads, triton.cdiv(max_extend_len, BLOCK_M))
    _extend_attn_kernel[grid](
        q_extend, k_extend, v_extend, o_extend,
        k_buffer, v_buffer,
        qo_indptr, kv_indptr, kv_indices,
        sm_scale,
        stride_qh=head_dim, stride_kh=head_dim,
        NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
        BLOCK_DMODEL=head_dim,
        BLOCK_M_C=BLOCK_M, BLOCK_N_C=BLOCK_N,
        num_warps=NUM_WARPS, num_stages=1,
    )
    return o_extend
