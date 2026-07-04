"""Phase 9: end-to-end DVR determinism — the headline property.

The watched prompt must produce byte-identical output no matter which batch
composition it runs in. Batch compositions deliberately include different
SIZES (that's what changes the decode split heuristic; permutations of an
identical set usually agree even in nondet mode because kernels are
position-invariant — paper O2).
"""

import hashlib

import pytest
import torch

from tests.conftest import requires_gpu

pytestmark = [requires_gpu, pytest.mark.slow]

MODEL = "Qwen/Qwen3-0.6B"

WATCHED = ("Give a concise two-sentence explanation of why deterministic "
           "LLM inference matters for production systems.")
FILLERS = [
    "Write a short haiku about a database migration finishing cleanly.",
    "List three practical checks before deploying a model-serving change.",
    "Explain request batching to a new ML infrastructure engineer.",
    "Summarize the plot of a heist movie set inside a compiler.",
    "Describe the water cycle in exactly four sentences.",
]

# index lists into [WATCHED] + FILLERS; 0 = watched prompt
COMPOSITIONS = [
    [0],
    [0, 1, 2, 3],
    [3, 0, 1, 2],
    [2, 1, 4, 0],
    [0, 1, 2, 3, 4, 5][:5],
    [0] + [1, 2, 3, 4, 5] * 3,   # watched among 15 fillers
]
PROMPTS = [WATCHED] + FILLERS


def _make_engine(mode, window=32, group=4):
    from sid.config import EngineConfig, Mode
    from sid.engine.engine import Engine

    cfg = EngineConfig(model_path=MODEL, mode=Mode(mode), kv_pool_tokens=64 * 1024,
                       dvr_window_size=window, dvr_group_size=group)
    return Engine(cfg)


def _run_composition(engine, comp, temperature=0.0, max_new=256, deterministic=True):
    from sid.config import SamplingParams

    prompts = [PROMPTS[i] for i in comp]
    params = [SamplingParams(max_new_tokens=max_new, temperature=temperature,
                             top_p=0.95 if temperature > 0 else 1.0,
                             seed=1234, is_deterministic=deterministic,
                             ignore_eos=True)
              for _ in prompts]
    outs = engine.generate(prompts, params, chat=True)
    watched_out = outs[comp.index(0)]
    return watched_out


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_dvr_watched_prompt_identical_across_compositions(temperature):
    engine = _make_engine("dvr")
    results = {}
    for trial in range(2):
        for ci, comp in enumerate(COMPOSITIONS):
            out = _run_composition(engine, comp, temperature=temperature)
            h = hashlib.sha256(str(out.token_ids).encode()).hexdigest()
            results.setdefault("hash", h)
            assert h == results["hash"], (
                f"DVR determinism violated (temp={temperature}) at composition "
                f"{ci} trial {trial}:\nbaseline != {out.token_ids[:32]}..."
            )
    del engine
    torch.cuda.empty_cache()


def test_dvr_rollback_rate_is_sane():
    """A rollback storm (every window mismatching) means a decode/verify skew
    (e.g. sample_pos convention) even though outputs stay deterministic."""
    engine = _make_engine("dvr")
    out = _run_composition(engine, [0, 1, 2, 3], temperature=0.0, max_new=256)
    assert out.num_windows > 0
    rollback_rate = out.num_rollbacks / out.num_windows
    assert rollback_rate < 0.5, (
        f"rollback storm: {out.num_rollbacks}/{out.num_windows} windows rolled "
        "back — check sample_pos convention and verify-path numerics"
    )
    del engine
    torch.cuda.empty_cache()


def test_verify_pass_is_run_consistent():
    """Run the SAME verify batch twice; logits must match bitwise."""
    from sid.config import SamplingParams

    engine = _make_engine("dvr")
    rid = engine.add_request(prompt=WATCHED,
                             params=SamplingParams(max_new_tokens=40,
                                                   is_deterministic=True,
                                                   ignore_eos=True))
    # Step until a verify-ready state exists, then verify twice manually.
    for _ in range(64):
        new = engine.scheduler.admit()
        if new:
            for g in engine.scheduler.prefill_groups(new):
                engine._run_prefill(g)
        else:
            reqs = engine.scheduler.decode_reqs()
            if not reqs:
                break
            engine._run_decode(reqs)
        ready = engine.verifier.collect_ready(engine.scheduler.active)
        if ready:
            fb1, meta1 = engine.verifier.build_verify_batch(ready[0])
            logits1 = engine.runner.forward(fb1).clone()
            fb2, meta2 = engine.verifier.build_verify_batch(ready[0])
            logits2 = engine.runner.forward(fb2).clone()
            assert torch.equal(logits1, logits2), \
                "verify pass is not run-consistent (bitwise)"
            return
    pytest.fail("never reached a verify-ready state")


def test_nondet_mode_diverges_across_compositions():
    """Sanity check on the demo's premise: the fast path SHOULD diverge for
    some composition (if it never does, the demo comparison is vacuous)."""
    engine = _make_engine("nondet")
    hashes = set()
    for comp in COMPOSITIONS:
        out = _run_composition(engine, comp, temperature=0.0, max_new=256,
                               deterministic=False)
        hashes.add(hashlib.sha256(str(out.token_ids).encode()).hexdigest())
    # Not a hard failure if equal — but flag it loudly.
    if len(hashes) == 1:
        pytest.skip("nondet mode did not diverge on this hardware/workload; "
                    "demo should use longer outputs or more fillers")
    del engine
    torch.cuda.empty_cache()
