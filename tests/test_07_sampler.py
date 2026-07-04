"""Phase 7: sampler determinism properties. Runs on CPU or GPU.

The critical cross-path property: given identical logits, the decode path and
the verify path must produce the identical token — they share sample() and the
sample_pos convention, so any skew here becomes a DVR rollback storm.
"""

import torch

from sid.engine.sampler import sample


def _sample_one(logits, temp, seed, pos, top_k=-1, top_p=1.0):
    r = logits.shape[0]
    dev = logits.device
    return sample(
        logits,
        torch.full((r,), temp, dtype=torch.float32, device=dev),
        torch.full((r,), top_k, dtype=torch.int64, device=dev),
        torch.full((r,), top_p, dtype=torch.float32, device=dev),
        torch.full((r,), seed, dtype=torch.int64, device=dev),
        torch.as_tensor(pos, dtype=torch.int64, device=dev),
    )


def test_greedy_is_argmax_with_first_tie():
    logits = torch.zeros(2, 100)
    logits[0, 7] = 5.0
    logits[1, 3] = 2.0
    logits[1, 42] = 2.0  # tie -> first index wins
    toks = _sample_one(logits, 0.0, 42, [10, 11])
    assert toks.tolist() == [7, 3]


def test_same_seed_pos_same_token_across_batch_shapes():
    torch.manual_seed(0)
    logits_row = torch.randn(1, 5000)
    alone = _sample_one(logits_row, 0.8, 1234, [17])

    for batch in (2, 7, 33):
        logits = torch.cat([torch.randn(batch - 1, 5000), logits_row])
        toks = _sample_one(logits, 0.8, 1234, list(range(100, 100 + batch - 1)) + [17])
        assert toks[-1].item() == alone[0].item(), \
            f"seeded draw changed with batch shape {batch}"


def test_different_pos_gives_different_draws():
    torch.manual_seed(1)
    logits = torch.randn(1, 1000).repeat(64, 1)
    toks = _sample_one(logits, 1.5, 42, list(range(64)))
    assert len(set(toks.tolist())) > 1, "position seeding produced constant draws"


def test_decode_vs_verify_path_equivalence():
    """Simulates the two call sites: decode samples one row at its position;
    verify samples W rows at consecutive positions. Row j of verify with the
    same logits and position as a decode step must give the same token."""
    torch.manual_seed(2)
    W = 32
    logits = torch.randn(W, 4000)
    seed, base_pos = 777, 250

    verify_toks = _sample_one(logits, 0.7, seed, list(range(base_pos, base_pos + W)),
                              top_k=50, top_p=0.9)
    for j in [0, 1, W // 2, W - 1]:
        decode_tok = _sample_one(logits[j:j + 1], 0.7, seed, [base_pos + j],
                                 top_k=50, top_p=0.9)
        assert decode_tok[0].item() == verify_toks[j].item(), \
            f"decode/verify sampling skew at window index {j}"


def test_top_k_top_p_respected():
    logits = torch.tensor([[10.0, 9.0, 8.0, -20.0, -20.0]])
    for pos in range(200):
        tok = _sample_one(logits, 2.0, 5, [pos], top_k=3).item()
        assert tok in (0, 1, 2), f"top_k=3 violated: sampled {tok}"
    peaked = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0]])
    for pos in range(100):
        tok = _sample_one(peaked, 1.0, 5, [pos], top_p=0.5).item()
        assert tok == 0, f"top_p=0.5 violated: sampled {tok}"
