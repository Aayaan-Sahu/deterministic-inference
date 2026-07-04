"""FCFS continuous-batching scheduler with full KV reservation at admission.

A request is admitted only if the KV pool can cover EVERY active request's
worst case (prompt + max_new_tokens), so decode can never run out of slots
mid-flight — this deletes the whole preemption/retraction problem class
(fine at 0.6B scale where the pool holds hundreds of thousands of tokens).

Isolated prefill (dvr mode): deterministic requests must prefill alone —
prefill batch shape must be a function of the request only, so token 0 (which
is never covered by a verification window) is deterministic by construction.
Non-deterministic requests batch freely under max_prefill_tokens.
"""

from __future__ import annotations

from collections import deque

from sid.config import EngineConfig, Mode
from sid.engine.kv_cache import KVCache
from sid.engine.request import Request, RequestState


class Scheduler:
    def __init__(self, cfg: EngineConfig, kv: KVCache):
        self.cfg = cfg
        self.kv = kv
        self.waiting: deque[Request] = deque()
        self.active: list[Request] = []  # RUNNING or FINISHED_PENDING_VERIFY

    # ---- admission --------------------------------------------------------

    @staticmethod
    def _worst_case_slots(req: Request) -> int:
        # Final kv_len at max_new_tokens outputs is P + max_new - 1; +1 slack.
        return req.prompt_len + req.params.max_new_tokens

    def _future_need(self) -> int:
        return sum(max(0, self._worst_case_slots(r) - r.kv_len) for r in self.active)

    def can_admit(self, req: Request) -> bool:
        if len(self.active) >= self.cfg.max_decode_batch:
            return False
        if not self.kv.free_rows:
            return False
        return self.kv.free_count >= self._future_need() + self._worst_case_slots(req)

    def admit(self) -> list[Request]:
        """Pop admissible requests FCFS. Returns newly admitted (need prefill)."""
        admitted = []
        while self.waiting and self.can_admit(self.waiting[0]):
            req = self.waiting.popleft()
            req.req_row = self.kv.alloc_row()
            req.state = RequestState.RUNNING
            self.active.append(req)
            admitted.append(req)
        return admitted

    # ---- prefill grouping --------------------------------------------------

    def prefill_groups(self, new_reqs: list[Request]) -> list[list[Request]]:
        """Split newly admitted requests into prefill batches.

        dvr: each deterministic request gets its own batch; the rest share
        batches under the token budget. Other modes: budget-batched only.
        """
        isolated, batched = [], []
        for r in new_reqs:
            if self.cfg.mode == Mode.DVR and r.params.is_deterministic:
                isolated.append([r])
            else:
                batched.append(r)

        groups = isolated
        cur, cur_tokens = [], 0
        for r in batched:
            if cur and cur_tokens + r.prompt_len > self.cfg.max_prefill_tokens:
                groups.append(cur)
                cur, cur_tokens = [], 0
            cur.append(r)
            cur_tokens += r.prompt_len
        if cur:
            groups.append(cur)
        return groups

    # ---- views --------------------------------------------------------------

    def decode_reqs(self) -> list[Request]:
        """Requests that take part in the next decode step."""
        return [r for r in self.active
                if r.state == RequestState.RUNNING and r.finished_reason is None]

    def release_finished(self) -> list[Request]:
        """Retire requests that are finished and (if deterministic in dvr mode)
        fully verified. Frees their KV slots and row. Returns released reqs."""
        dvr = self.cfg.mode == Mode.DVR
        done = []
        for r in self.active:
            if r.finished_reason is None:
                continue
            if dvr and r.params.is_deterministic and r.unverified() > 0:
                r.state = RequestState.FINISHED_PENDING_VERIFY
                continue
            done.append(r)
        for r in done:
            if r.kv_len > 0:
                self.kv.free(self.kv.req_to_token[r.req_row, :r.kv_len])
            self.kv.free_row(r.req_row)
            r.req_row = -1
            r.state = RequestState.DONE
            self.active.remove(r)
        return done

    def resume_rolled_back(self) -> None:
        """After verification, un-finished requests go back to RUNNING."""
        for r in self.active:
            if r.state == RequestState.FINISHED_PENDING_VERIFY and r.finished_reason is None:
                r.state = RequestState.RUNNING

    def has_work(self) -> bool:
        return bool(self.waiting or self.active)
