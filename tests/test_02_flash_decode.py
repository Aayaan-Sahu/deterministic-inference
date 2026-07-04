"""Phase 2: CUDA split-KV flash-decode kernel vs fp32 SDPA reference.

Sweeps batch sizes, sequence lengths (including split-boundary cases around
32/256), split policies (heuristic + fixed-256 + forced counts), and shuffled
non-contiguous slot assignments.
"""

import pytest
import torch

from tests.conftest import ref_attention, requires_gpu

pytestmark = requires_gpu

HQ, HKV, D = 16, 8, 128
SCALE = D ** -0.5


def _setup(bs, seq_lens, seed=0):
    torch.manual_seed(seed)
    total = sum(seq_lens)
    num_slots = total + 64
    q = torch.randn(bs, HQ, D, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    # Shuffled, non-contiguous slots (slot 0 reserved).
    perm = torch.randperm(num_slots - 1, device="cuda")[:total] + 1
    kv_indices = perm.to(torch.int32)
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    kv_indptr[1:] = torch.cumsum(torch.tensor(seq_lens, device="cuda"), 0).to(torch.int32)
    return q, k_cache, v_cache, kv_indptr, kv_indices


def _run(q, k_cache, v_cache, kv_indptr, kv_indices, num_splits, grid_splits,
         fixed_split_size):
    from sid.kernels.jit import load_ext

    ext = load_ext()
    bs = q.shape[0]
    ws = max(grid_splits, 1)
    part_o = torch.empty(bs, HQ, ws, D, device="cuda", dtype=torch.float32)
    part_lse = torch.empty(bs, HQ, ws, device="cuda", dtype=torch.float32)
    o = torch.empty_like(q)
    ext.flash_decode_fwd(q, k_cache, v_cache, kv_indptr, kv_indices,
                         num_splits, grid_splits, fixed_split_size, SCALE,
                         part_o, part_lse, o)
    return o


def _reference(q, k_cache, v_cache, kv_indptr, kv_indices):
    bs = q.shape[0]
    outs = []
    for b in range(bs):
        idx = kv_indices[kv_indptr[b]:kv_indptr[b + 1]].long()
        k = k_cache[idx]
        v = v_cache[idx]
        outs.append(ref_attention(q[b:b + 1], k, v, SCALE, causal_offset=None))
    return torch.cat(outs)


@pytest.mark.parametrize("bs,seq_lens", [
    (1, [1]),
    (1, [31]), (1, [32]), (1, [33]),
    (1, [255]), (1, [256]), (1, [257]),
    (2, [300, 5]),
    (3, [1000, 32, 999]),
    (8, [100] * 8),
    (17, [17 * i + 1 for i in range(17)]),
    (64, [50] * 64),
    (1, [5000]),
])
@pytest.mark.parametrize("policy", ["heuristic", "fixed256", "splits1", "splits8", "splits32"])
def test_flash_decode_vs_reference(bs, seq_lens, policy):
    q, k_cache, v_cache, kv_indptr, kv_indices = _setup(bs, seq_lens)
    seq_t = torch.tensor(seq_lens, device="cuda")

    if policy == "heuristic":
        sm = torch.cuda.get_device_properties(0).multi_processor_count
        splits = max(1, min(-(-2 * sm // (bs * HKV)), 32))
        num_splits = torch.clamp((seq_t + 31) // 32, max=splits).to(torch.int32)
        grid, fixed = splits, 0
    elif policy == "fixed256":
        num_splits = ((seq_t + 255) // 256).to(torch.int32)
        grid, fixed = int(num_splits.max().item()), 256
    else:
        s = int(policy.replace("splits", ""))
        num_splits = torch.clamp((seq_t + 31) // 32, max=s).to(torch.int32)
        grid, fixed = s, 0

    out = _run(q, k_cache, v_cache, kv_indptr, kv_indices, num_splits, grid, fixed)
    ref = _reference(q, k_cache, v_cache, kv_indptr, kv_indices)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_same_input_same_splits_is_bitwise_identical():
    """Run-to-run consistency for a fixed split configuration."""
    q, k_cache, v_cache, kv_indptr, kv_indices = _setup(4, [123, 456, 789, 7])
    seq_t = torch.tensor([123, 456, 789, 7], device="cuda")
    num_splits = ((seq_t + 255) // 256).to(torch.int32)
    grid = int(num_splits.max().item())
    a = _run(q, k_cache, v_cache, kv_indptr, kv_indices, num_splits, grid, 256)
    b = _run(q, k_cache, v_cache, kv_indptr, kv_indices, num_splits, grid, 256)
    assert torch.equal(a, b)


def test_split_count_changes_bits():
    """Documents the nondeterminism mechanism: different split counts give
    numerically close but (generally) bitwise-different outputs."""
    q, k_cache, v_cache, kv_indptr, kv_indices = _setup(1, [4096])
    seq_t = torch.tensor([4096], device="cuda")
    outs = []
    for s in (1, 3, 8, 32):
        num_splits = torch.clamp((seq_t + 31) // 32, max=s).to(torch.int32)
        outs.append(_run(q, k_cache, v_cache, kv_indptr, kv_indices,
                         num_splits, s, 0))
    torch.testing.assert_close(outs[0].float(), outs[-1].float(), atol=3e-2, rtol=3e-2)
    assert any(not torch.equal(outs[0], o) for o in outs[1:]), \
        "expected at least one split count to change low-order bits"
