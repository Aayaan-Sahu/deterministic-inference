"""Paged KV cache with token-granularity slots (page_size = 1).

Two structures, mirroring sglang's design at 1/100 scale:
  - per-layer K/V pools:  bf16 [num_slots, num_kv_heads, head_dim]
  - req_to_token table:   int32 [max_reqs, max_seq_len] mapping
                          (request row, position) -> pool slot

Slot 0 and row 0 are reserved as null targets so stray gathers hit valid
memory. The allocator is a LIFO free stack living on the GPU.
"""

from __future__ import annotations

import os

import torch

from sid.config import EngineConfig, ModelConfig
from sid.kernels.jit import load_ext


class KVCache:
    def __init__(self, cfg: EngineConfig, mcfg: ModelConfig, device: str):
        self.num_slots = cfg.kv_pool_tokens
        self.max_reqs = cfg.max_reqs
        self.max_seq_len = cfg.max_seq_len
        self.device = device
        self.ext = load_ext()

        self.k_pools = [
            torch.zeros(self.num_slots, mcfg.num_kv_heads, mcfg.head_dim,
                        dtype=torch.bfloat16, device=device)
            for _ in range(mcfg.num_layers)
        ]
        self.v_pools = [
            torch.zeros(self.num_slots, mcfg.num_kv_heads, mcfg.head_dim,
                        dtype=torch.bfloat16, device=device)
            for _ in range(mcfg.num_layers)
        ]

        # Slot allocator: LIFO stack, slot 0 reserved.
        self.free_stack = torch.arange(1, self.num_slots, dtype=torch.int64, device=device)
        self.free_count = self.num_slots - 1

        # Row 0 of req_to_token reserved.
        self.req_to_token = torch.zeros(self.max_reqs, self.max_seq_len,
                                        dtype=torch.int32, device=device)
        self.free_rows = list(range(self.max_reqs - 1, 0, -1))

        # Optional double-free / foreign-free detection (costs GPU syncs).
        self.debug = os.environ.get("SID_DEBUG_ALLOC", "0") == "1"
        self._allocated: set[int] = set()

    def layer(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k_pools[i], self.v_pools[i]

    # ---- slot allocation -------------------------------------------------

    def alloc(self, n: int) -> torch.Tensor:
        assert n <= self.free_count, f"KV pool exhausted: need {n}, free {self.free_count}"
        out = self.free_stack[self.free_count - n:self.free_count].clone()
        self.free_count -= n
        if self.debug:
            ids = out.tolist()
            dup = self._allocated.intersection(ids)
            assert not dup, f"allocator handed out live slots: {sorted(dup)[:5]}"
            self._allocated.update(ids)
        return out

    def free(self, slots: torch.Tensor) -> None:
        n = slots.numel()
        if n == 0:
            return
        if self.debug:
            ids = slots.tolist()
            foreign = [s for s in ids if s not in self._allocated]
            assert not foreign, f"freeing slots not allocated: {foreign[:5]}"
            self._allocated.difference_update(ids)
        self.free_stack[self.free_count:self.free_count + n] = slots.to(torch.int64)
        self.free_count += n

    # ---- request rows ----------------------------------------------------

    def alloc_row(self) -> int:
        assert self.free_rows, "req_to_token rows exhausted"
        return self.free_rows.pop()

    def free_row(self, row: int) -> None:
        assert 0 < row < self.max_reqs
        self.free_rows.append(row)

    def write_row(self, row: int, start: int, slots: torch.Tensor) -> None:
        self.req_to_token[row, start:start + slots.numel()] = slots.to(torch.int32)
