"""LLM-42 decode-verify-rollback (DVR).

Every verification pass has the EXACT same shape: G sequences x W extend
tokens, padded with dummy sequences when fewer than G real requests are ready.
Fixed shape => same kernels, same tiles, same reduction order => the verifier
is run-consistent, and its outputs depend only on each request's own history
(see extend_attention.py for why co-batched sequences can't interact).

Per real request with prompt length P, verified_len v (>= 1), and
u = min(unverified, W) window tokens t[0..u-1] = output_ids[v : v+u]:

  input row  = [output_ids[v-1], t[0], ..., t[u-2], DUMMY x (W-u)]
  positions  = P+v-1 .. P+v+W-2
  prefix     = KV positions 0 .. P+v-2   (all verified-consistent)
  out_slots  = THE SAME physical KV slots decode wrote for positions
               P+v-1 .. P+v+u-2 (overwrite!), then scratch padding slots.
  prediction j is compared against t[j]; sample_pos = P+v+j.

On the first mismatch at j: keep the j matching tokens, append the verifier's
token (>= 1 token of guaranteed progress per pass), roll everything after it
back — including freeing the now-stale KV slots — and clear/recompute the
finished state. The verifier's KV overwrite is what makes the *next* window
deterministic even though decode wrote drifted values.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sid.config import DUMMY_TOKEN_ID, EngineConfig
from sid.engine.batch import ForwardBatch, ForwardMode
from sid.engine.kv_cache import KVCache
from sid.engine.request import Request


class DummyPool:
    """Static KV scratch for dummy sequences and window padding.

    Dummy sequences have prefix_len = 0: their K/V come entirely from the
    extend tensors, so they need pool slots only as reshape_and_cache targets
    (G*W slots). Padding positions of short real windows also need scratch
    targets (G*(W-1) slots, reused every pass).
    """

    def __init__(self, kv: KVCache, group_size: int, window_size: int):
        self.dummy_slots = kv.alloc(group_size * window_size)      # [G*W]
        self.padding_slots = kv.alloc(group_size * (window_size - 1))
        self.window_size = window_size


@dataclass
class VerifyMeta:
    reqs: list  # real requests, in batch order
    window_lens: list  # u per real request
    windows: list  # decode tokens under verification per request


class Verifier:
    def __init__(self, cfg: EngineConfig, runner, kv: KVCache):
        self.cfg = cfg
        self.runner = runner
        self.kv = kv
        self.W = cfg.dvr_window_size
        self.G = cfg.dvr_group_size
        self.pool = DummyPool(kv, self.G, self.W)
        self.device = kv.device

        buf = runner.verify_buffers
        gw = self.G * self.W
        buf["qo_indptr"].copy_(
            torch.arange(0, gw + 1, self.W, dtype=torch.int32, device=self.device)
        )
        self._sample_indices = torch.arange(gw, dtype=torch.int64, device=self.device)

    # ------------------------------------------------------------------

    @staticmethod
    def ready(req: Request, window_size: int) -> bool:
        if not req.params.is_deterministic:
            return False
        if req.unverified() <= 0:
            return False
        return req.unverified() >= window_size or req.finished_reason is not None

    def collect_ready(self, reqs: list[Request]) -> list[list[Request]]:
        ready = sorted((r for r in reqs if self.ready(r, self.W)), key=lambda r: r.rid)
        return [ready[i:i + self.G] for i in range(0, len(ready), self.G)]

    # ------------------------------------------------------------------

    def build_verify_batch(self, reqs: list[Request]) -> tuple[ForwardBatch, VerifyMeta]:
        assert 1 <= len(reqs) <= self.G
        W, G = self.W, self.G
        kv, dev = self.kv, self.device
        buf = self.runner.verify_buffers

        input_ids, positions, sample_positions = [], [], []
        out_slot_chunks = []
        kv_lens = []      # full per-sequence KV length: prefix + W
        kv_chunks = []    # pool slots for ALL positions (prefix + window)
        temps, top_ks, top_ps, seeds = [], [], [], []
        window_lens, windows = [], []
        pad_used = 0

        for r in reqs:
            p, v = r.prompt_len, r.verified_len
            assert v >= 1, f"req {r.rid} entered verify with verified_len 0"
            u = min(r.unverified(), W)
            assert u >= 1
            window = r.output_ids[v:v + u]
            window_lens.append(u)
            windows.append(window)

            row_ids = [r.output_ids[v - 1]] + window[:-1] + [DUMMY_TOKEN_ID] * (W - u)
            input_ids.extend(row_ids)
            positions.extend(range(p + v - 1, p + v - 1 + W))
            sample_positions.extend(range(p + v, p + v + W))

            prefix_len = p + v - 1

            # Overwrite decode's slots for the u real positions; scratch for padding.
            avail_end = p + v - 1 + u
            assert r.kv_len >= avail_end, (
                f"req {r.rid}: window needs KV slots through position {avail_end - 1}, "
                f"but only {r.kv_len} positions are valid (u={u}, v={v})"
            )
            real_slots = kv.req_to_token[r.req_row, p + v - 1:avail_end].to(torch.int64)
            pad = W - u
            if pad:
                window_slots = torch.cat([
                    real_slots,
                    self.pool.padding_slots[pad_used:pad_used + pad],
                ])
                pad_used += pad
            else:
                window_slots = real_slots
            out_slot_chunks.append(window_slots)

            # Full KV for attention = verified prefix + the window slots
            # (whose K/V this pass writes before attending).
            kv_lens.append(prefix_len + W)
            kv_chunks.append(torch.cat([
                kv.req_to_token[r.req_row, :prefix_len].to(torch.int32),
                window_slots.to(torch.int32),
            ]))

            temps.extend([r.params.temperature] * W)
            top_ks.extend([r.params.top_k if r.params.top_k > 0 else -1] * W)
            top_ps.extend([r.params.top_p] * W)
            seeds.extend([r.params.seed] * W)

        num_dummies = G - len(reqs)
        for d in range(num_dummies):
            di = len(reqs) + d
            input_ids.extend([DUMMY_TOKEN_ID] * W)
            positions.extend(range(W))
            sample_positions.extend(range(1, W + 1))
            dummy_slots = self.pool.dummy_slots[di * W:(di + 1) * W]
            out_slot_chunks.append(dummy_slots)
            kv_lens.append(W)  # prefix 0: the window IS the whole sequence
            kv_chunks.append(dummy_slots.to(torch.int32))
            temps.extend([0.0] * W)
            top_ks.extend([-1] * W)
            top_ps.extend([1.0] * W)
            seeds.extend([0] * W)

        # Write into the persistent buffers (fixed pointers across passes).
        buf["input_ids"].copy_(torch.tensor(input_ids, dtype=torch.int64, device=dev))
        buf["positions"].copy_(torch.tensor(positions, dtype=torch.int64, device=dev))
        buf["out_slots"].copy_(torch.cat(out_slot_chunks))
        buf["sample_positions"].copy_(
            torch.tensor(sample_positions, dtype=torch.int64, device=dev))
        buf["temperatures"].copy_(torch.tensor(temps, dtype=torch.float32, device=dev))
        buf["top_ks"].copy_(torch.tensor(top_ks, dtype=torch.int64, device=dev))
        buf["top_ps"].copy_(torch.tensor(top_ps, dtype=torch.float32, device=dev))
        buf["seeds"].copy_(torch.tensor(seeds, dtype=torch.int64, device=dev))

        total_kv = sum(kv_lens)
        buf["kv_indices"][:total_kv].copy_(torch.cat(kv_chunks))
        indptr = torch.zeros(G + 1, dtype=torch.int64)
        indptr[1:] = torch.cumsum(torch.tensor(kv_lens, dtype=torch.int64), dim=0)
        buf["kv_indptr"].copy_(indptr.to(torch.int32).to(dev))

        fb = ForwardBatch(
            mode=ForwardMode.VERIFY,
            input_ids=buf["input_ids"],
            positions=buf["positions"],
            out_slots=buf["out_slots"],
            qo_indptr=buf["qo_indptr"],
            kv_indptr=buf["kv_indptr"],
            kv_indices=buf["kv_indices"][:total_kv],
            max_extend_len=W,
            sample_indices=self._sample_indices,
            sample_positions=buf["sample_positions"],
            temperatures=buf["temperatures"],
            top_ks=buf["top_ks"],
            top_ps=buf["top_ps"],
            seeds=buf["seeds"],
            invariant=True,
        )
        return fb, VerifyMeta(reqs=list(reqs), window_lens=window_lens, windows=windows)

    # ------------------------------------------------------------------

    def compare_and_rollback(self, meta: VerifyMeta, verify_tokens: torch.Tensor) -> None:
        """verify_tokens: int64 [G*W] sampled from the verification logits."""
        W = self.W
        tokens = verify_tokens.tolist()
        for i, r in enumerate(meta.reqs):
            u = meta.window_lens[i]
            window = meta.windows[i]
            verify_out = tokens[i * W:i * W + u]

            j = u
            for idx in range(u):
                if verify_out[idx] != window[idx]:
                    j = idx
                    break

            r.num_windows += 1
            v = r.verified_len
            if j < u:
                old_len = r.num_output
                new_len = v + j + 1
                # Truncate; accept the verifier's token at the mismatch
                # (guaranteed >= 1 verified token of progress per pass).
                r.output_ids = r.output_ids[:v + j] + [verify_out[j]]
                r.verified_len = new_len
                r.num_rollbacks += 1
                r.tokens_rolled_back += old_len - (v + j)

                # Free KV for the discarded tail: positions
                # P+new_len-1 .. P+old_len-2 (old frontier was P+old_len-2).
                start_pos = r.prompt_len + new_len - 1
                end_pos = r.prompt_len + old_len - 1  # exclusive
                if end_pos > start_pos:
                    self.kv.free(self.kv.req_to_token[r.req_row, start_pos:end_pos])

                r.finished_reason = None
                r.check_finished()  # corrected token may itself be EOS / hit length
            else:
                r.verified_len = v + u

            r.assert_invariants(dvr_mode=True)
            assert r.kv_len == r.prompt_len + r.num_output - 1
