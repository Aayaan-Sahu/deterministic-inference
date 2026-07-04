"""ForwardBatch: the tensors for one model forward, and PREFILL/DECODE builders.

sample_pos convention (shared by ALL paths — decode, prefill, verify):
    sample_pos = absolute position of the token being GENERATED.
prefill of P tokens generates the token at position P;  a decode step feeding
output_ids[N-1] (at position P+N-1) generates the token at position P+N;
verify row j of a window starting at verified_len v generates position P+v+j.
Getting this wrong doesn't break determinism, it silently makes every
verification window mismatch at index 0 (a rollback storm) — see tests.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional

import torch

from sid.engine.kv_cache import KVCache
from sid.engine.request import Request


class ForwardMode(enum.Enum):
    PREFILL = "prefill"
    DECODE = "decode"
    VERIFY = "verify"


@dataclass
class ForwardBatch:
    mode: ForwardMode
    input_ids: torch.Tensor        # int64 [T]
    positions: torch.Tensor        # int64 [T]
    out_slots: torch.Tensor        # int64 [T]  KV write targets (-1 = skip)

    # Attention over the paged pool. kv_indptr/kv_indices always cover the
    # FULL sequence (prefix + freshly written positions) — both the CUDA
    # decode kernel and the extend kernel read every key from the pool with
    # absolute-position tiling (see extend_attention.py on why that anchoring
    # is load-bearing for DVR determinism).
    kv_indptr: Optional[torch.Tensor] = None    # int32 [B+1]
    kv_indices: Optional[torch.Tensor] = None   # int32
    seq_lens: Optional[torch.Tensor] = None     # int32 [B] (GPU, decode only)
    max_seq_len: int = 0                        # host-side max(seq_lens)

    # extend attention (prefill / verify / triton decode fallback)
    qo_indptr: Optional[torch.Tensor] = None    # int32 [B+1]
    max_extend_len: int = 0

    # sampling (R = number of logit rows)
    sample_indices: Optional[torch.Tensor] = None   # int64 [R] rows of hidden
    sample_positions: Optional[torch.Tensor] = None # int64 [R] see docstring
    temperatures: Optional[torch.Tensor] = None     # fp32 [R]
    top_ks: Optional[torch.Tensor] = None           # int64 [R]
    top_ps: Optional[torch.Tensor] = None           # fp32 [R]
    seeds: Optional[torch.Tensor] = None            # int64 [R]

    invariant: bool = False  # use batch-invariant kernel paths


def _sampling_tensors(reqs: list[Request], device: str, repeat: int = 1):
    def t(vals, dtype):
        out = torch.tensor(vals, dtype=dtype, device=device)
        return out.repeat_interleave(repeat) if repeat > 1 else out

    temps = t([r.params.temperature for r in reqs], torch.float32)
    top_ks = t([r.params.top_k if r.params.top_k > 0 else -1 for r in reqs], torch.int64)
    top_ps = t([r.params.top_p for r in reqs], torch.float32)
    seeds = t([r.params.seed for r in reqs], torch.int64)
    return temps, top_ks, top_ps, seeds


def build_prefill_batch(reqs: list[Request], kv: KVCache, device: str,
                        invariant: bool) -> ForwardBatch:
    """Allocates prompt KV slots and builds the extend-attention batch
    (prefix_len = 0: fresh prefill)."""
    input_ids, positions, out_slots = [], [], []
    qo = [0]
    for r in reqs:
        p = r.prompt_len
        slots = kv.alloc(p)
        kv.write_row(r.req_row, 0, slots)
        input_ids.append(torch.tensor(r.prompt_ids, dtype=torch.int64, device=device))
        positions.append(torch.arange(p, dtype=torch.int64, device=device))
        out_slots.append(slots)
        qo.append(qo[-1] + p)

    qo_indptr = torch.tensor(qo, dtype=torch.int32, device=device)
    all_slots = torch.cat(out_slots)
    temps, top_ks, top_ps, seeds = _sampling_tensors(reqs, device)
    return ForwardBatch(
        mode=ForwardMode.PREFILL,
        input_ids=torch.cat(input_ids),
        positions=torch.cat(positions),
        out_slots=all_slots,
        qo_indptr=qo_indptr,
        # prefix is empty: the full KV IS the freshly written prompt slots.
        kv_indptr=qo_indptr,
        kv_indices=all_slots.to(torch.int32),
        max_extend_len=max(r.prompt_len for r in reqs),
        sample_indices=(qo_indptr[1:] - 1).to(torch.int64),
        sample_positions=torch.tensor([r.prompt_len for r in reqs],
                                      dtype=torch.int64, device=device),
        temperatures=temps, top_ks=top_ks, top_ps=top_ps, seeds=seeds,
        invariant=invariant,
    )


def build_decode_batch(reqs: list[Request], kv: KVCache, device: str,
                       invariant: bool) -> ForwardBatch:
    """One new token per request. Allocates one KV slot each; attention runs
    over the full sequence INCLUDING the freshly written position (the Triton
    fallback uses the same full kv_indices with extend_len = 1)."""
    bs = len(reqs)
    input_ids = torch.tensor([r.output_ids[-1] for r in reqs], dtype=torch.int64, device=device)
    pos_list = [r.prompt_len + r.num_output - 1 for r in reqs]
    positions = torch.tensor(pos_list, dtype=torch.int64, device=device)

    new_slots = kv.alloc(bs)
    seq_lens_list = []
    kv_rows = []
    for i, r in enumerate(reqs):
        pos = pos_list[i]
        assert pos == r.kv_len, f"req {r.rid}: decode pos {pos} != kv_len {r.kv_len}"
        kv.req_to_token[r.req_row, pos] = new_slots[i].to(torch.int32)
        seq_len = pos + 1
        seq_lens_list.append(seq_len)
        kv_rows.append(kv.req_to_token[r.req_row, :seq_len])

    kv_indices = torch.cat(kv_rows)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=device)
    kv_indptr[1:] = torch.cumsum(
        torch.tensor(seq_lens_list, dtype=torch.int64, device=device), dim=0
    ).to(torch.int32)
    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32, device=device)

    fb = ForwardBatch(
        mode=ForwardMode.DECODE,
        input_ids=input_ids,
        positions=positions,
        out_slots=new_slots,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        seq_lens=seq_lens,
        max_seq_len=max(seq_lens_list),
        sample_indices=torch.arange(bs, dtype=torch.int64, device=device),
        sample_positions=positions + 1,  # generated token position = fed pos + 1
        qo_indptr=torch.arange(bs + 1, dtype=torch.int32, device=device),
        max_extend_len=1,
        invariant=invariant,
    )
    fb.temperatures, fb.top_ks, fb.top_ps, fb.seeds = _sampling_tensors(reqs, device)
    return fb
