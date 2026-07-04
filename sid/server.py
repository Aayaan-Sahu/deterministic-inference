"""Minimal OpenAI-ish HTTP server around the Engine.

    python -m sid.server --mode dvr --port 8000

Endpoints:
  POST /v1/completions        {model?, prompt: str|[str], max_tokens,
                               temperature, top_p, top_k, seed,
                               is_deterministic: bool, stream: bool}
  POST /v1/chat/completions   {messages: [...], ...same knobs}
  GET  /health

The engine steps in a dedicated background thread; request handlers enqueue
work and wait on per-request events. `is_deterministic` maps straight onto
the LLM-42 per-request determinism flag (only meaningful in --mode dvr).
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from sid.config import EngineConfig, Mode, SamplingParams
from sid.detok import IncrementalDetokenizer
from sid.engine.engine import Engine


class CompletionRequest(BaseModel):
    model: str = "sid"
    prompt: str | list[str] = ""
    messages: list[dict] | None = None
    max_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 42
    is_deterministic: bool = False
    stream: bool = False
    ignore_eos: bool = False


class ServerState:
    def __init__(self, cfg: EngineConfig):
        self.engine = Engine(cfg)
        self.lock = threading.Lock()
        self.shutdown = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while not self.shutdown:
            with self.lock:
                busy = self.engine.scheduler.has_work()
                if busy:
                    try:
                        self.engine.step()
                    except Exception:
                        # A dead engine thread would leave every current and
                        # future request spinning in _wait_done while /health
                        # still answers ok — fail the head waiting request
                        # (the usual culprit: infeasible KV reservation) and
                        # keep stepping.
                        logging.getLogger(__name__).exception("engine.step failed")
                        self.engine.abort_waiting_head()
            if not busy:
                time.sleep(0.005)

    def submit(self, prompt=None, prompt_ids=None, params=None, chat=False) -> int:
        with self.lock:
            return self.engine.add_request(
                prompt=prompt, prompt_ids=prompt_ids, params=params, chat=chat)


state: ServerState | None = None
app = FastAPI()


def _params(req: CompletionRequest) -> SamplingParams:
    return SamplingParams(
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        max_new_tokens=req.max_tokens,
        seed=req.seed,
        is_deterministic=req.is_deterministic,
        ignore_eos=req.ignore_eos,
    )


def _wait_done(rid: int, poll: float = 0.005):
    while True:
        with state.lock:
            out = state.engine.get_output(rid)
        if out is not None:
            return out
        time.sleep(poll)


@app.get("/health")
def health():
    return {"status": "ok", "mode": state.engine.cfg.mode.value}


@app.post("/v1/completions")
def completions(req: CompletionRequest):
    prompts = req.prompt if isinstance(req.prompt, list) else [req.prompt]
    params = _params(req)

    if req.stream:
        assert len(prompts) == 1, "streaming supports a single prompt"
        rid = state.submit(prompt=prompts[0], params=params)
        return StreamingResponse(_stream(rid), media_type="text/event-stream")

    rids = [state.submit(prompt=p, params=params) for p in prompts]
    outs = [_wait_done(rid) for rid in rids]
    return JSONResponse({
        "id": f"cmpl-{uuid.uuid4().hex[:24]}",
        "object": "text_completion",
        "model": req.model,
        "choices": [
            {
                "index": i,
                "text": o.text,
                "token_ids": o.token_ids,
                "finish_reason": o.finish_reason,
                "llm42_stats": {
                    "num_windows": o.num_windows,
                    "num_rollbacks": o.num_rollbacks,
                    "tokens_rolled_back": o.tokens_rolled_back,
                },
            }
            for i, o in enumerate(outs)
        ],
    })


@app.post("/v1/chat/completions")
def chat_completions(req: CompletionRequest):
    assert req.messages, "messages required"
    with state.lock:
        prompt_ids = state.engine.tokenizer.apply_chat_template(
            req.messages, tokenize=True, add_generation_prompt=True,
            return_dict=False)
    rid = state.submit(prompt_ids=prompt_ids, params=_params(req))
    out = _wait_done(rid)
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "model": req.model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": out.text},
            "finish_reason": out.finish_reason,
        }],
    })


def _stream(rid: int):
    detok = IncrementalDetokenizer(state.engine.tokenizer)
    sent = 0
    while True:
        with state.lock:
            done_out = state.engine.get_output(rid)
            if done_out is None:
                req = next((r for r in state.engine.scheduler.active if r.rid == rid), None)
                visible = state.engine.visible_tokens(req) if req else []
            else:
                visible = done_out.token_ids
        if len(visible) > sent or done_out is not None:
            delta = detok.update(list(visible))
            sent = len(visible)
            if delta:
                yield f"data: {json.dumps({'text': delta})}\n\n"
        if done_out is not None:
            yield f"data: {json.dumps({'finish_reason': done_out.finish_reason})}\n\n"
            yield "data: [DONE]\n\n"
            return
        time.sleep(0.01)


def main():
    global state
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--mode", default="dvr", choices=[m.value for m in Mode])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--dvr-window-size", type=int, default=32)
    ap.add_argument("--dvr-group-size", type=int, default=4)
    ap.add_argument("--kv-pool-tokens", type=int, default=256 * 1024)
    ap.add_argument("--decode-backend", default="cuda", choices=["cuda", "triton"])
    args = ap.parse_args()

    cfg = EngineConfig(
        model_path=args.model,
        mode=Mode(args.mode),
        dvr_window_size=args.dvr_window_size,
        dvr_group_size=args.dvr_group_size,
        kv_pool_tokens=args.kv_pool_tokens,
        decode_backend=args.decode_backend,
    )
    state = ServerState(cfg)

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
