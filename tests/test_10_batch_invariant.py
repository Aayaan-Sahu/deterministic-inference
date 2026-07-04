"""Phase 10: batch_invariant mode — deterministic by construction.

Same watched-prompt-across-compositions property as the DVR test, but with no
verifier at all: the fixed-split decode kernel + persistent Triton GEMM must
make outputs identical directly. Also sanity-checks the persistent matmul
numerics and its row-invariance."""

import hashlib

import pytest
import torch

from tests.conftest import requires_gpu

pytestmark = requires_gpu

MODEL = "Qwen/Qwen3-0.6B"


def test_persistent_matmul_matches_cublas():
    from sid.kernels.matmul_persistent import bi_linear

    torch.manual_seed(0)
    for m, n, k in [(1, 1024, 1024), (7, 3072, 1024), (128, 151936, 1024), (500, 1024, 3072)]:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        ref = torch.nn.functional.linear(x, w).float()
        out = bi_linear(x, w).float()
        torch.testing.assert_close(out, ref, atol=3e-1, rtol=3e-2)


def test_persistent_matmul_row_invariance():
    """Row 0's result must be bitwise identical whether M=1 or M=333."""
    from sid.kernels.matmul_persistent import bi_linear

    torch.manual_seed(1)
    x0 = torch.randn(1, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(3072, 1024, device="cuda", dtype=torch.bfloat16)
    alone = bi_linear(x0, w)
    for m in (2, 64, 333):
        x = torch.cat([x0, torch.randn(m - 1, 1024, device="cuda", dtype=torch.bfloat16)])
        batched = bi_linear(x, w)
        assert torch.equal(alone[0], batched[0]), f"row invariance broken at M={m}"


@pytest.mark.slow
def test_batch_invariant_mode_identical_across_compositions():
    from sid.config import EngineConfig, Mode, SamplingParams
    from sid.engine.engine import Engine

    cfg = EngineConfig(model_path=MODEL, mode=Mode.BATCH_INVARIANT,
                       kv_pool_tokens=64 * 1024)
    engine = Engine(cfg)

    watched = "Explain why floating point addition is not associative."
    fillers = [
        "Name three sorting algorithms.",
        "Write a limerick about GPUs.",
        "What is the boiling point of water at sea level?",
    ]
    comps = [[0], [0, 1, 2, 3], [2, 0, 3, 1], [0] + [1, 2, 3] * 4]
    prompts = [watched] + fillers

    hashes = set()
    for comp in comps:
        ps = [prompts[i] for i in comp]
        params = [SamplingParams(max_new_tokens=200, ignore_eos=True) for _ in ps]
        outs = engine.generate(ps, params, chat=True)
        watched_out = outs[comp.index(0)]
        hashes.add(hashlib.sha256(str(watched_out.token_ids).encode()).hexdigest())

    assert len(hashes) == 1, "batch_invariant mode diverged across compositions"
    del engine
    torch.cuda.empty_cache()
