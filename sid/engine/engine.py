"""Engine: the step loop tying scheduler, model runner, sampler, and DVR
together, plus the offline generate() API.

One step =
  1. admit waiting requests (full KV reservation)
  2. prefill newly admitted (deterministic requests isolated in dvr mode)
  3. else: one decode iteration over all running requests
  4. dvr: run fixed-shape verification groups, compare, roll back
  5. retire finished (and, for deterministic requests, fully verified) requests
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import torch

from sid.config import EngineConfig, Mode, SamplingParams
from sid.engine.batch import build_decode_batch, build_prefill_batch
from sid.engine.dvr import Verifier
from sid.engine.model_runner import ModelRunner
from sid.engine.request import Request, RequestState
from sid.engine.scheduler import Scheduler

logger = logging.getLogger(__name__)


@dataclass
class RequestOutput:
    rid: int
    prompt_ids: list
    token_ids: list
    text: str
    finish_reason: str
    num_windows: int = 0
    num_rollbacks: int = 0
    tokens_rolled_back: int = 0


class Engine:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.runner = ModelRunner(cfg, cfg.device)
        self.kv = self.runner.kv_cache
        self.scheduler = Scheduler(cfg, self.kv)
        self.verifier = Verifier(cfg, self.runner, self.kv) if cfg.mode == Mode.DVR else None

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.runner.model_dir))

        self._next_rid = 0
        self._done: dict[int, Request] = {}
        # Free slots with nothing admitted (verifier scratch already carved
        # out): the most any single request can ever reserve.
        self._pool_capacity = self.kv.free_count

    # ---- public API --------------------------------------------------------

    def add_request(self, prompt: Optional[str] = None,
                    prompt_ids: Optional[list] = None,
                    params: Optional[SamplingParams] = None,
                    chat: bool = False) -> int:
        params = params or SamplingParams()
        if prompt_ids is None:
            assert prompt is not None
            if chat:
                prompt_ids = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True, add_generation_prompt=True,
                    return_dict=False,  # newer transformers default to a dict
                )
            else:
                prompt_ids = self.tokenizer.encode(prompt)
        if len(prompt_ids) > self.cfg.max_prompt_len:
            raise ValueError(
                f"prompt too long: {len(prompt_ids)} > {self.cfg.max_prompt_len} "
                "(no chunked prefill in this engine)")
        pos_limit = min(self.cfg.max_seq_len, self.runner.mcfg.max_position_embeddings)
        if len(prompt_ids) + params.max_new_tokens > pos_limit:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens "
                f"({params.max_new_tokens}) exceeds the position limit {pos_limit}")
        if len(prompt_ids) + params.max_new_tokens > self._pool_capacity:
            raise ValueError(
                f"prompt ({len(prompt_ids)}) + max_new_tokens "
                f"({params.max_new_tokens}) exceeds the KV pool capacity "
                f"{self._pool_capacity} — the request could never be admitted; "
                "raise kv_pool_tokens or lower max_new_tokens")
        rid = self._next_rid
        self._next_rid += 1
        req = Request(rid, prompt_ids, params, self.runner.mcfg.eos_token_ids)
        self.scheduler.waiting.append(req)
        return rid

    def generate(self, prompts: list, params: Optional[list] = None,
                 chat: bool = False) -> list[RequestOutput]:
        """Offline API: run all prompts to completion, return in input order."""
        if params is None:
            params = [SamplingParams() for _ in prompts]
        if isinstance(params, SamplingParams):
            params = [params] * len(prompts)
        assert len(params) == len(prompts), \
            f"{len(prompts)} prompts but {len(params)} SamplingParams"
        rids = [self.add_request(prompt=p, params=sp, chat=chat)
                for p, sp in zip(prompts, params)]
        while self.scheduler.has_work():
            self.step()
        return [self._output(self._done.pop(rid)) for rid in rids]

    def get_output(self, rid: int) -> Optional[RequestOutput]:
        """Returns the finished output and forgets the request (one-shot)."""
        req = self._done.pop(rid, None)
        return self._output(req) if req else None

    def visible_tokens(self, req: Request) -> list:
        """Tokens the client may see right now (DVR gate)."""
        return req.output_ids[:req.gate(self.cfg.mode == Mode.DVR)]

    def abort_waiting_head(self) -> Optional[int]:
        """Fail the head waiting request (e.g. after a step() error) so its
        client gets an answer instead of waiting forever. Returns its rid."""
        if not self.scheduler.waiting:
            return None
        req = self.scheduler.waiting.popleft()
        req.finished_reason = req.finished_reason or "error"
        self._done[req.rid] = req
        return req.rid

    # ---- step loop ----------------------------------------------------------

    def step(self) -> None:
        new_reqs = self.scheduler.admit()

        if not new_reqs and not self.scheduler.active and self.scheduler.waiting:
            head = self.scheduler.waiting[0]
            raise RuntimeError(
                f"request {head.rid} can never be admitted (needs "
                f"{head.prompt_len + head.params.max_new_tokens} KV slots, pool "
                f"has {self.kv.free_count} free) — raise kv_pool_tokens or lower "
                "max_new_tokens")

        if new_reqs:
            for group in self.scheduler.prefill_groups(new_reqs):
                self._run_prefill(group)
        else:
            decode_reqs = self.scheduler.decode_reqs()
            if decode_reqs:
                self._run_decode(decode_reqs)

        if self.verifier is not None:
            for group in self.verifier.collect_ready(self.scheduler.active):
                fb, meta = self.verifier.build_verify_batch(group)
                logits = self.runner.forward(fb)
                tokens = self.runner.sample(fb, logits)
                self.verifier.compare_and_rollback(meta, tokens)
            self.scheduler.resume_rolled_back()

        for req in self.scheduler.release_finished():
            self._done[req.rid] = req

    # ---- internals ------------------------------------------------------------

    def _run_prefill(self, reqs: list[Request]) -> None:
        fb = build_prefill_batch(
            reqs, self.kv, self.cfg.device,
            invariant=self.cfg.mode == Mode.BATCH_INVARIANT,
        )
        logits = self.runner.forward(fb)
        tokens = self.runner.sample(fb, logits).tolist()
        dvr = self.cfg.mode == Mode.DVR
        for r, tok in zip(reqs, tokens):
            r.output_ids.append(int(tok))
            if dvr and r.params.is_deterministic:
                # Token 0 came from an isolated, shape-fixed prefill:
                # verified by construction.
                r.verified_len = 1
            r.check_finished()
            assert r.kv_len == r.prompt_len

    def _run_decode(self, reqs: list[Request]) -> None:
        fb = build_decode_batch(
            reqs, self.kv, self.cfg.device,
            invariant=self.cfg.mode == Mode.BATCH_INVARIANT,
        )
        logits = self.runner.forward(fb)
        tokens = self.runner.sample(fb, logits).tolist()
        for r, tok in zip(reqs, tokens):
            r.output_ids.append(int(tok))
            r.check_finished()

    def _output(self, req: Request) -> RequestOutput:
        return RequestOutput(
            rid=req.rid,
            prompt_ids=req.prompt_ids,
            token_ids=list(req.output_ids),
            text=self.tokenizer.decode(req.output_ids, skip_special_tokens=True),
            finish_reason=req.finished_reason or "abort",
            num_windows=req.num_windows,
            num_rollbacks=req.num_rollbacks,
            tokens_rolled_back=req.tokens_rolled_back,
        )
