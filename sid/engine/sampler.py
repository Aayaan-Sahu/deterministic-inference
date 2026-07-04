"""Deterministic sampler shared by the decode fast path and the DVR verifier.

Greedy (temperature == 0): plain argmax with first-index tie-break.
Stochastic: top-k/top-p filtering then seeded Gumbel-max, a port of LLM-42's
multinomial_with_seed — the uniform driving each draw is a hash of
(seed, sample_pos, column), so a request's draw at a given position is
identical no matter where it sits in a batch or which pass produced the
logits. This replaces torch.multinomial, whose global-RNG consumption order
depends on batch layout.

Everything here is row-independent: no reduction ever crosses rows, so the
number of co-batched rows cannot change any row's token.
"""

from __future__ import annotations

import torch

_EPS = 1e-10


def _seeded_gumbel(seeds: torch.Tensor, sample_pos: torch.Tensor,
                   num_cols: int) -> torch.Tensor:
    """[R, num_cols] Gumbel noise, a pure function of (seed, pos, col)."""
    step_seed = seeds * 19349663 ^ sample_pos * 73856093            # int64 [R]
    cols = torch.arange(num_cols, dtype=torch.int64, device=seeds.device)
    hashed = step_seed.unsqueeze(-1) * 8589934591 ^ cols * 479001599
    uniform = (hashed % (2 ** 24)).float() / (2 ** 24)              # [0, 1)
    return -torch.log(-torch.log(uniform + _EPS) + _EPS)


def sample(logits: torch.Tensor, temperatures: torch.Tensor, top_ks: torch.Tensor,
           top_ps: torch.Tensor, seeds: torch.Tensor,
           sample_pos: torch.Tensor) -> torch.Tensor:
    """logits: fp32 [R, V] -> int64 [R] token ids."""
    # torch.argmax/max document first-index tie-breaking (deterministic given
    # identical logits).
    greedy_tokens = torch.argmax(logits, dim=-1)
    stochastic = temperatures > 0
    if not bool(stochastic.any()):
        return greedy_tokens

    # Stochastic path (computed for all rows, selected per row at the end).
    temps = torch.where(stochastic, temperatures, torch.ones_like(temperatures))
    probs = torch.softmax(logits / temps.unsqueeze(-1), dim=-1)

    probs_sort, probs_idx = probs.sort(dim=-1, descending=True)
    # top-p: drop tokens once the cumulative mass BEFORE them exceeds top_p.
    cum = torch.cumsum(probs_sort, dim=-1)
    top_p_mask = (cum - probs_sort) > top_ps.unsqueeze(-1)
    # top-k: drop ranks >= k (k = -1 disables).
    ranks = torch.arange(probs.shape[-1], device=probs.device).expand_as(probs_sort)
    k = torch.where(top_ks > 0, top_ks, torch.full_like(top_ks, probs.shape[-1]))
    top_k_mask = ranks >= k.unsqueeze(-1)
    dropped = top_p_mask | top_k_mask
    dropped[:, 0] = False  # never drop the top token

    # Scatter the keep-mask back to ORIGINAL vocab order and add Gumbel noise
    # keyed by vocab id, so the sort's tie ordering cannot change the draw
    # (rank-keyed noise would).
    keep = torch.zeros_like(dropped)
    keep.scatter_(-1, probs_idx, ~dropped)
    log_probs = torch.where(
        keep, torch.log(probs + _EPS), torch.full_like(probs, float("-inf"))
    )
    gumbel = _seeded_gumbel(seeds, sample_pos, probs.shape[-1])
    sampled_tokens = torch.argmax(log_probs + gumbel, dim=-1)

    return torch.where(stochastic, sampled_tokens, greedy_tokens)
