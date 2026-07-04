"""Varlen causal attention over the paged KV pool ("extend" attention).

Used for prefill (prefix_len == 0), DVR verification passes, and the Triton
decode fallback. The new K/V of the extend tokens are written into the pool by
reshape_and_cache BEFORE this kernel runs, so ALL keys — prefix and extend —
are read from the pool through kv_indices, and causality is applied by
absolute position.

DETERMINISM — two properties both matter (LLM-42 paper O2/O3):
  1. No autotuning: tile constants are hardcoded; q tiles are per-request
     (indexed off qo_indptr), so a request's reduction covers only its own
     rows and its own KV — co-batched requests and packing order are
     irrelevant.
  2. KEY TILES ARE ANCHORED AT ABSOLUTE POSITION 0, not at the window start.
     The DVR verifier's window offset varies across runs (rollbacks shift it),
     so the same absolute token position can be verified at different window
     offsets. Anchoring the KV loop at position 0 makes the reduction order
     for the query at position k a function of k alone — bitwise identical
     for any window split. (An earlier version tiled the window tokens
     separately from the prefix, which broke exactly this and produced
     one-in-a-thousand argmax flips after rollbacks.)
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
    Q, O,                                      # [Text, HQ, D] bf16
    K_buffer, V_buffer,                        # [S, HKV, D] bf16 (paged pool)
    qo_indptr, kv_indptr, kv_indices,          # int32
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
    kv_len = kv_end - kv_start          # full sequence: prefix + extend
    prefix_len = kv_len - extend_len

    offs_m = pid_m * BLOCK_M_C + tl.arange(0, BLOCK_M_C)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    mask_m = offs_m < extend_len

    q_ptrs = Q + (q_start + offs_m[:, None]) * (NUM_Q_HEADS * stride_qh) \
        + pid_h * stride_qh + offs_d[None, :]
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

    m_i = tl.full([BLOCK_M_C], NEG_INF, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M_C], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M_C, BLOCK_DMODEL], dtype=tl.float32)

    # Query local row j sits at absolute position prefix_len + j and may see
    # key indices m <= prefix_len + j. Bound the loop by this tile's last row.
    visible_end = tl.minimum(kv_len, prefix_len + (pid_m + 1) * BLOCK_M_C)

    # Single loop over ALL keys, tiles anchored at absolute position 0.
    for start_n in tl.range(0, visible_end, BLOCK_N_C, num_stages=NUM_STAGES):
        offs_n = start_n + tl.arange(0, BLOCK_N_C)
        mask_n = offs_n < kv_len
        slots = tl.load(kv_indices + kv_start + offs_n, mask=mask_n, other=0)
        k_ptrs = K_buffer + slots[:, None].to(tl.int64) * (NUM_KV_HEADS * stride_kh) \
            + kv_head * stride_kh + offs_d[None, :]
        k = tl.load(k_ptrs, mask=mask_n[:, None], other=0.0)

        qk = tl.dot(q, tl.trans(k)) * sm_scale
        visible = offs_n[None, :] <= (prefix_len + offs_m[:, None])
        qk = tl.where(visible & mask_m[:, None] & mask_n[None, :], qk, NEG_INF)

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

    out = acc / l_i[:, None]
    o_ptrs = O + (q_start + offs_m[:, None]) * (NUM_Q_HEADS * stride_qh) \
        + pid_h * stride_qh + offs_d[None, :]
    tl.store(o_ptrs, out.to(O.dtype.element_ty), mask=mask_m[:, None])


def extend_attention_fwd(
    q_extend: torch.Tensor,   # [Text, HQ, D] bf16
    k_buffer: torch.Tensor,   # [S, HKV, D] bf16 — must already contain the
    v_buffer: torch.Tensor,   #                    extend tokens' K/V!
    qo_indptr: torch.Tensor,  # int32 [B+1]
    kv_indptr: torch.Tensor,  # int32 [B+1]  FULL length (prefix + extend)
    kv_indices: torch.Tensor, # int32        pool slots for ALL positions
    max_extend_len: int,
    sm_scale: float,
) -> torch.Tensor:
    Text, num_q_heads, head_dim = q_extend.shape
    num_kv_heads = k_buffer.shape[1]
    assert q_extend.is_contiguous()
    assert k_buffer.is_contiguous() and v_buffer.is_contiguous()

    o_extend = torch.empty_like(q_extend)
    bs = qo_indptr.numel() - 1
    grid = (bs, num_q_heads, triton.cdiv(max_extend_len, BLOCK_M))
    _extend_attn_kernel[grid](
        q_extend, o_extend,
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
