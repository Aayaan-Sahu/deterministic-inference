"""Request object, lifecycle state machine, and the KV-length invariant.

State machine:
    WAITING -> RUNNING -> DONE                      (nondet / batch_invariant)
    WAITING -> RUNNING <-> FINISHED_PENDING_VERIFY -> DONE   (dvr deterministic;
        a rollback can clear finished_reason and send the request back to RUNNING)

THE invariant everything hangs off (asserted, not folklore):
    with P = len(prompt_ids) and N = len(output_ids) >= 1,
    valid KV covers positions 0 .. P+N-2  (the newest sampled token's KV is
    not written until that token is fed back through the model).
Prefill writes positions 0..P-1 and samples output_ids[0]; each decode step
feeds output_ids[N-1] at position P+N-1, writes its KV there, and samples
output_ids[N].
"""

from __future__ import annotations

import enum
from typing import Optional

from sid.config import SamplingParams


class RequestState(enum.Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED_PENDING_VERIFY = "finished_pending_verify"
    DONE = "done"


class Request:
    def __init__(self, rid: int, prompt_ids: list[int], params: SamplingParams,
                 eos_token_ids: tuple):
        self.rid = rid
        self.prompt_ids = list(prompt_ids)
        self.params = params
        self.eos_token_ids = set(eos_token_ids)
        if params.stop_token_ids:
            self.eos_token_ids.update(params.stop_token_ids)

        self.output_ids: list[int] = []
        self.state = RequestState.WAITING
        self.finished_reason: Optional[str] = None
        self.req_row: int = -1

        # DVR bookkeeping.
        self.verified_len = 0      # output tokens verified (monotone non-decreasing)
        self.num_windows = 0
        self.num_rollbacks = 0
        self.tokens_rolled_back = 0

        # Streaming.
        self.stream_offset = 0     # tokens already emitted to the client

    # ---- derived quantities ---------------------------------------------

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_ids)

    @property
    def num_output(self) -> int:
        return len(self.output_ids)

    @property
    def kv_len(self) -> int:
        """Number of valid KV positions (see module docstring)."""
        if self.num_output == 0:
            return 0
        return self.prompt_len + self.num_output - 1

    def unverified(self) -> int:
        return self.num_output - self.verified_len

    def gate(self, dvr_mode: bool) -> int:
        """Tokens visible to the client."""
        if dvr_mode and self.params.is_deterministic:
            return self.verified_len
        return self.num_output

    # ---- lifecycle -------------------------------------------------------

    def check_finished(self) -> None:
        """(Re-)derive finished_reason from current output. Safe to call after
        rollback: it can re-finish on a corrected EOS token."""
        if self.num_output == 0:
            self.finished_reason = None
            return
        if not self.params.ignore_eos and self.output_ids[-1] in self.eos_token_ids:
            self.finished_reason = "stop"
        elif self.num_output >= self.params.max_new_tokens:
            self.finished_reason = "length"
        else:
            self.finished_reason = None

    def assert_invariants(self, dvr_mode: bool) -> None:
        assert 0 <= self.verified_len <= self.num_output, \
            f"req {self.rid}: verified_len {self.verified_len} vs output {self.num_output}"
        assert self.stream_offset <= self.gate(dvr_mode), \
            f"req {self.rid}: emitted beyond gate"

    def __repr__(self) -> str:
        return (f"Request(rid={self.rid}, state={self.state.value}, P={self.prompt_len}, "
                f"N={self.num_output}, verified={self.verified_len}, "
                f"finished={self.finished_reason})")
