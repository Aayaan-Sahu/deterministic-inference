"""Phase 4: reshape_and_cache scatter/gather roundtrip + slot overwrite."""

import torch

from tests.conftest import requires_gpu

pytestmark = requires_gpu

HKV, D = 8, 128


def test_scatter_gather_roundtrip():
    from sid.kernels.jit import load_ext

    ext = load_ext()
    torch.manual_seed(0)
    t, slots = 100, 500
    k = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    mapping = (torch.randperm(slots - 1, device="cuda")[:t] + 1).to(torch.int64)

    ext.reshape_and_cache(k, v, k_cache, v_cache, mapping)
    assert torch.equal(k_cache[mapping], k)
    assert torch.equal(v_cache[mapping], v)
    # untouched slots stay zero
    untouched = torch.ones(slots, dtype=torch.bool, device="cuda")
    untouched[mapping] = False
    assert k_cache[untouched].abs().sum() == 0


def test_overwrite_same_slots():
    """The DVR verifier overwrites decode's KV in-place at the same slots."""
    from sid.kernels.jit import load_ext

    ext = load_ext()
    torch.manual_seed(1)
    t = 32
    k_cache = torch.zeros(t + 8, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(t + 8, HKV, D, device="cuda", dtype=torch.bfloat16)
    mapping = torch.arange(1, t + 1, device="cuda", dtype=torch.int64)

    k1 = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    v1 = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    ext.reshape_and_cache(k1, v1, k_cache, v_cache, mapping)

    k2 = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    v2 = torch.randn(t, HKV, D, device="cuda", dtype=torch.bfloat16)
    ext.reshape_and_cache(k2, v2, k_cache, v_cache, mapping)
    assert torch.equal(k_cache[mapping], k2)
    assert torch.equal(v_cache[mapping], v2)


def test_negative_slot_skipped():
    from sid.kernels.jit import load_ext

    ext = load_ext()
    k = torch.randn(2, HKV, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(2, HKV, D, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.zeros(4, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros(4, HKV, D, device="cuda", dtype=torch.bfloat16)
    mapping = torch.tensor([-1, 2], device="cuda", dtype=torch.int64)
    ext.reshape_and_cache(k, v, k_cache, v_cache, mapping)
    assert k_cache[2].abs().sum() > 0
    assert torch.equal(k_cache[2], k[1])
    assert k_cache[0].abs().sum() == 0 and k_cache[1].abs().sum() == 0
