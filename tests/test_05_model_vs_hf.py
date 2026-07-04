"""Phase 5: our Qwen3 forward vs HuggingFace transformers (bf16, eager).

Prefill logits: argmax must match on a suite of prompts; top-5 sets must
overlap heavily (bf16 kernel differences make exact logit equality impossible).
"""

import pytest
import torch

from tests.conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.slow]

MODEL = "Qwen/Qwen3-0.6B"

PROMPTS = [
    "The capital of France is",
    "1 + 1 =",
    "def fibonacci(n):",
    "Once upon a time, in a village by the sea,",
    "The chemical symbol for gold is",
    "In machine learning, overfitting means",
    "Photosynthesis is the process by which",
    "Q: What is the tallest mountain on Earth? A:",
    "SELECT name FROM users WHERE",
    "To be, or not to be, that is",
    "El resultado de dos más dos es",
    "The three primary colors are",
    "Newton's second law states that",
    "A haiku about rain:",
    "The speed of light in vacuum is approximately",
    "import numpy as np\n\narr =",
    "The opposite of 'hot' is",
    "月は地球の",
    "Grace Hopper was famous for",
    "The Pythagorean theorem relates",
]


@pytest.fixture(scope="module")
def engine():
    from sid.config import EngineConfig, Mode
    from sid.engine.engine import Engine

    cfg = EngineConfig(model_path=MODEL, mode=Mode.NONDET, kv_pool_tokens=64 * 1024)
    return Engine(cfg)


@pytest.fixture(scope="module")
def hf_model(engine):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        str(engine.runner.model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    return model


def _our_prefill_logits(engine, prompt_ids):
    from sid.config import SamplingParams
    from sid.engine.batch import build_prefill_batch
    from sid.engine.request import Request

    req = Request(10_000 + len(prompt_ids), prompt_ids, SamplingParams(),
                  engine.runner.mcfg.eos_token_ids)
    req.req_row = engine.kv.alloc_row()
    fb = build_prefill_batch([req], engine.kv, "cuda", invariant=False)
    logits = engine.runner.forward(fb)
    # free
    engine.kv.free(engine.kv.req_to_token[req.req_row, :req.prompt_len])
    engine.kv.free_row(req.req_row)
    return logits[0]


def test_prefill_logits_match_hf(engine, hf_model):
    tok = engine.tokenizer
    argmax_matches = 0
    for prompt in PROMPTS:
        ids = tok.encode(prompt)
        ours = _our_prefill_logits(engine, ids)

        with torch.no_grad():
            hf_out = hf_model(torch.tensor([ids], device="cuda")).logits[0, -1].float()

        our_top5 = set(ours.topk(5).indices.tolist())
        hf_top5 = set(hf_out.topk(5).indices.tolist())
        assert len(our_top5 & hf_top5) >= 3, \
            f"top-5 divergence on {prompt!r}: {our_top5} vs {hf_top5}"
        if ours.argmax().item() == hf_out.argmax().item():
            argmax_matches += 1

    assert argmax_matches >= int(0.9 * len(PROMPTS)), \
        f"argmax matched HF on only {argmax_matches}/{len(PROMPTS)} prompts"
