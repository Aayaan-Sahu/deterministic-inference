"""Decode attention wrapper: split-KV CUDA kernel + the num_splits policy.

The num_splits policy is where batch-composition nondeterminism deliberately
enters the fast path (modes nondet / dvr):

    splits = clamp(ceil(2 * SM_COUNT / (B * num_kv_heads)), 1, max_kv_splits)
    per-request: min(splits, ceil(seq_len / 32))

i.e. small batches get many splits (to fill the GPU), large batches get few —
the occupancy-driven behavior of FlashAttention-3's scheduler heuristic and
sglang's get_num_kv_splits. Different co-scheduled batches => different split
counts => different floating-point reduction order => different logits.

batch_invariant mode instead uses fixed-size split tiles (256 tokens):
    splits = ceil(seq_len / 256)
which depends only on the request's own length, never on the batch.
"""

from __future__ import annotations

import functools

import torch

from sid.kernels.jit import load_ext


@functools.lru_cache(maxsize=1)
def _sm_count() -> int:
    return torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count


class DecodeAttention:
    """Holds the persistent split-KV workspace (fixed pointers across steps)."""

    def __init__(self, max_batch: int, num_q_heads: int, num_kv_heads: int,
                 head_dim: int, max_kv_splits: int, invariant_split_size: int,
                 max_seq_len: int, device: str):
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_kv_splits = max_kv_splits
        self.invariant_split_size = invariant_split_size
        self.invariant_max_splits = -(-max_seq_len // invariant_split_size)
        # Workspace splits-dim stride covers both policies.
        self.ws_splits = max(max_kv_splits, self.invariant_max_splits)
        self.part_o = torch.empty(
            max_batch, num_q_heads, self.ws_splits, head_dim,
            dtype=torch.float32, device=device,
        )
        self.part_lse = torch.empty(
            max_batch, num_q_heads, self.ws_splits, dtype=torch.float32, device=device
        )
        self.ext = load_ext()

    def forward(self, q: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor,
                kv_indptr: torch.Tensor, kv_indices: torch.Tensor,
                seq_lens: torch.Tensor, max_seq_len: int, sm_scale: float,
                invariant: bool) -> torch.Tensor:
        """q: [B, HQ, HD] bf16 -> o: [B, HQ, HD] bf16.

        seq_lens: int32/int64 GPU tensor [B]; max_seq_len: host-side max(seq_lens)
        (the engine tracks lengths on CPU, so no GPU sync is needed here).
        """
        bs = q.shape[0]
        assert bs <= self.part_o.shape[0], "decode batch exceeds workspace"
        if invariant:
            tile = self.invariant_split_size
            num_splits = ((seq_lens + tile - 1) // tile).to(torch.int32)
            grid_splits = -(-max_seq_len // tile)
            fixed_split_size = tile
        else:
            splits = -(-2 * _sm_count() // (bs * self.num_kv_heads))
            splits = max(1, min(splits, self.max_kv_splits))
            num_splits = torch.clamp((seq_lens + 31) // 32, max=splits).to(torch.int32)
            grid_splits = splits
            fixed_split_size = 0
        o = torch.empty_like(q)
        self.ext.flash_decode_fwd(
            q, k_cache, v_cache,
            kv_indptr, kv_indices, num_splits,
            grid_splits, fixed_split_size, sm_scale,
            self.part_o, self.part_lse, o,
        )
        return o
