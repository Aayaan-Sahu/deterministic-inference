"""Phase 6: end-to-end generation sanity vs HF generate (greedy, bs=1),
plus batching + streaming-gate smoke tests."""

import pytest
import torch

from tests.conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.slow]

MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture(scope="module")
def engine():
    from sid.config import EngineConfig, Mode
    from sid.engine.engine import Engine

    cfg = EngineConfig(model_path=MODEL, mode=Mode.NONDET, kv_pool_tokens=64 * 1024)
    return Engine(cfg)


def test_greedy_matches_hf_prefix(engine):
    from transformers import AutoModelForCausalLM

    from sid.config import SamplingParams

    prompt = "The three laws of thermodynamics are:"
    n_tokens = 128

    outs = engine.generate([prompt], SamplingParams(max_new_tokens=n_tokens,
                                                    temperature=0.0, ignore_eos=True))
    ours = outs[0].token_ids

    hf = AutoModelForCausalLM.from_pretrained(
        str(engine.runner.model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    ).cuda().eval()
    ids = engine.tokenizer.encode(prompt)
    with torch.no_grad():
        hf_ids = hf.generate(
            torch.tensor([ids], device="cuda"), max_new_tokens=n_tokens,
            do_sample=False, num_beams=1,
        )[0, len(ids):].tolist()
    del hf
    torch.cuda.empty_cache()

    agree = 0
    for a, b in zip(ours, hf_ids):
        if a != b:
            break
        agree += 1
    assert agree >= int(0.9 * n_tokens), (
        f"greedy prefix agreement with HF too short: {agree}/{n_tokens}\n"
        f"ours: {ours[:agree+5]}\nhf:   {hf_ids[:agree+5]}"
    )


def test_batched_generation_smoke(engine):
    from sid.config import SamplingParams

    prompts = [f"Write one sentence about the number {i}." for i in range(8)]
    outs = engine.generate(prompts, SamplingParams(max_new_tokens=32))
    assert len(outs) == 8
    for o in outs:
        assert 1 <= len(o.token_ids) <= 32
        assert o.finish_reason in ("stop", "length")
    # engine returned to a clean state
    assert not engine.scheduler.active and not engine.scheduler.waiting


def test_temperature_sampling_smoke(engine):
    from sid.config import SamplingParams

    outs = engine.generate(
        ["Tell me something interesting."],
        SamplingParams(max_new_tokens=32, temperature=0.7, top_p=0.95, seed=1234),
    )
    assert len(outs[0].token_ids) >= 1
