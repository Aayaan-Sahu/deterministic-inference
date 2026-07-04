"""Rotary position embedding (NeoX rotate-half style, as used by Qwen3).

cos/sin table precomputed in fp32 at init; kernel applies RoPE in-place to
Q [T, HQ, D] and K [T, HKV, D] in one launch (one program per token).
Purely per-token/per-position => invariant in every mode.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_kernel(Q, K, POS, COS_SIN,
                 NUM_Q_HEADS: tl.constexpr, NUM_KV_HEADS: tl.constexpr,
                 HEAD_DIM: tl.constexpr, HALF: tl.constexpr):
    t = tl.program_id(0)
    pos = tl.load(POS + t)
    offs = tl.arange(0, HALF)
    cos = tl.load(COS_SIN + pos * HEAD_DIM + offs).to(tl.float32)
    sin = tl.load(COS_SIN + pos * HEAD_DIM + HALF + offs).to(tl.float32)

    for h in tl.static_range(NUM_Q_HEADS):
        base = Q + t * (NUM_Q_HEADS * HEAD_DIM) + h * HEAD_DIM
        x1 = tl.load(base + offs).to(tl.float32)
        x2 = tl.load(base + HALF + offs).to(tl.float32)
        tl.store(base + offs, (x1 * cos - x2 * sin).to(Q.dtype.element_ty))
        tl.store(base + HALF + offs, (x2 * cos + x1 * sin).to(Q.dtype.element_ty))

    for h in tl.static_range(NUM_KV_HEADS):
        base = K + t * (NUM_KV_HEADS * HEAD_DIM) + h * HEAD_DIM
        x1 = tl.load(base + offs).to(tl.float32)
        x2 = tl.load(base + HALF + offs).to(tl.float32)
        tl.store(base + offs, (x1 * cos - x2 * sin).to(K.dtype.element_ty))
        tl.store(base + HALF + offs, (x2 * cos + x1 * sin).to(K.dtype.element_ty))


class RotaryEmbedding:
    def __init__(self, head_dim: int, max_positions: int, theta: float, device: str):
        self.head_dim = head_dim
        half = head_dim // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
        pos = torch.arange(max_positions, dtype=torch.float32, device=device)
        freqs = torch.outer(pos, inv_freq)  # [max_pos, half]
        # Layout: [max_pos, head_dim] = [cos(half) | sin(half)], contiguous fp32.
        self.cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1).contiguous()

    def apply(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor) -> None:
        """In-place. q: [T, HQ, D] bf16, k: [T, HKV, D] bf16, positions: int64 [T]."""
        t, num_q_heads, head_dim = q.shape
        num_kv_heads = k.shape[1]
        assert head_dim == self.head_dim and q.is_contiguous() and k.is_contiguous()
        if t == 0:
            return
        _rope_kernel[(t,)](
            q, k, positions, self.cos_sin,
            NUM_Q_HEADS=num_q_heads, NUM_KV_HEADS=num_kv_heads,
            HEAD_DIM=head_dim, HALF=head_dim // 2,
        )
