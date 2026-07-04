"""Phase 3: Triton extend attention vs fp32 SDPA reference.

The kernel reads ALL keys (prefix + extend) from the paged pool with tiles
anchored at absolute position 0. Covers fresh prefill (prefix 0),
verify-shaped batches, mixed varlen batches, position-invariance across batch
slots, and — critically — WINDOW-OFFSET invariance: the same absolute query
position must produce bitwise-identical output regardless of how the sequence
is split into prefix vs extend (DVR rollbacks shift window offsets across
runs; this is paper Observation O3 and it's what an earlier window-relative
tiling version of this kernel got wrong).
"""

import pytest
import torch

from tests.conftest import ref_attention, requires_gpu

pytestmark = requires_gpu

HQ, HKV, D = 16, 8, 128
SCALE = D ** -0.5


def _make_batch(specs, seed=0):
    """specs: list of (prefix_len, extend_len). ALL K/V live in the pool."""
    torch.manual_seed(seed)
    total_kv = sum(p + e for p, e in specs)
    total_extend = sum(e for _, e in specs)
    num_slots = total_kv + 16

    q = torch.randn(total_extend, HQ, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)

    perm = torch.randperm(num_slots - 1, device="cuda")[:total_kv] + 1
    kv_indices = perm.to(torch.int32)
    qo = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
    qo[1:] = torch.cumsum(torch.tensor([e for _, e in specs], device="cuda"), 0).to(torch.int32)
    kvp = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
    kvp[1:] = torch.cumsum(torch.tensor([p + e for p, e in specs], device="cuda"), 0).to(torch.int32)
    return q, k_buf, v_buf, qo, kvp, kv_indices


def _reference(specs, q, k_buf, v_buf, qo, kvp, kv_indices):
    outs = []
    for i, (p, e) in enumerate(specs):
        qi = q[qo[i]:qo[i + 1]]
        idx = kv_indices[kvp[i]:kvp[i + 1]].long()
        k_full = k_buf[idx]
        v_full = v_buf[idx]
        # query j (extend-local) sees keys <= p + j.
        outs.append(ref_attention(qi, k_full, v_full, SCALE, causal_offset=p))
    return torch.cat(outs)


@pytest.mark.parametrize("specs", [
    [(0, 1)],
    [(0, 32)], [(0, 64)], [(0, 100)], [(0, 500)],
    [(1, 1)], [(100, 32)], [(1000, 64)],
    [(100, 32), (0, 17), (999, 1), (5, 64)],
    [(31, 32), (32, 32), (33, 32), (256, 32)],  # exact G x W verify shape
    [(7, 1), (100, 1), (5000, 1)],              # triton decode fallback shape
])
def test_extend_vs_reference(specs):
    from sid.kernels.extend_attention import extend_attention_fwd

    tensors = _make_batch(specs)
    q, k_buf, v_buf, qo, kvp, kv_indices = tensors
    out = extend_attention_fwd(q, k_buf, v_buf, qo, kvp, kv_indices,
                               max(e for _, e in specs), SCALE)
    ref = _reference(specs, *tensors)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_window_offset_invariance():
    """THE DVR regression test: one underlying sequence of L tokens, queried
    under different prefix/extend splits. The output for a given absolute
    position must be bitwise identical for every split that includes it."""
    from sid.kernels.extend_attention import extend_attention_fwd

    torch.manual_seed(11)
    L = 300  # total sequence length; not tile-aligned on purpose
    q_all = torch.randn(L, HQ, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(L + 32, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(L + 32, HKV, D, device="cuda", dtype=torch.bfloat16)
    idx = (torch.randperm(L + 31, device="cuda")[:L] + 1).to(torch.int32)

    def run(prefix: int):
        """Query positions prefix..L-1 with the rest as prefix."""
        e = L - prefix
        q = q_all[prefix:].contiguous()
        qo = torch.tensor([0, e], dtype=torch.int32, device="cuda")
        kvp = torch.tensor([0, L], dtype=torch.int32, device="cuda")
        return extend_attention_fwd(q, k_buf, v_buf, qo, kvp, idx, e, SCALE)

    # Window starts at deliberately awkward offsets (incl. non-multiples of
    # the 64-token tiles). Position 268..299 is covered by every split.
    baseline = run(prefix=268)  # 32-token window, like a verify pass
    for prefix in (233, 245, 267, 269, 236):
        other = run(prefix=prefix)
        overlap = L - 268
        got = other[268 - prefix:268 - prefix + overlap]
        assert torch.equal(baseline, got), (
            f"window-offset variance: prefix={prefix} changes bits for the "
            "same absolute positions — DVR determinism would break on rollback"
        )


def test_position_invariance_within_batch():
    """A sequence's output must be bitwise identical no matter which batch
    slot it occupies and what else is co-batched."""
    from sid.kernels.extend_attention import extend_attention_fwd

    torch.manual_seed(7)
    p, e = 100, 32
    L = p + e
    q0 = torch.randn(e, HQ, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(4096, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(4096, HKV, D, device="cuda", dtype=torch.bfloat16)
    idx0 = (torch.randperm(4095, device="cuda")[:L] + 1).to(torch.int32)

    def run(slot: int, others: int):
        specs, qs, idxs = [], [], []
        torch.manual_seed(1234)
        for i in range(others + 1):
            if i == slot:
                specs.append((p, e))
                qs.append(q0)
                idxs.append(idx0)
            else:
                pe, ee = 40 + 13 * i, 8 + i
                specs.append((pe, ee))
                qs.append(torch.randn(ee, HQ, D, device="cuda", dtype=torch.bfloat16))
                idxs.append(torch.randint(1, 4096, (pe + ee,), device="cuda",
                                          dtype=torch.int32))
        q = torch.cat(qs)
        kv_indices = torch.cat(idxs)
        qo = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
        qo[1:] = torch.cumsum(torch.tensor([s[1] for s in specs], device="cuda"), 0).to(torch.int32)
        kvp = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
        kvp[1:] = torch.cumsum(torch.tensor([s[0] + s[1] for s in specs], device="cuda"), 0).to(torch.int32)
        out = extend_attention_fwd(q, k_buf, v_buf, qo, kvp, kv_indices,
                                   max(s[1] for s in specs), SCALE)
        return out[qo[slot]:qo[slot] + e].clone()

    baseline = run(slot=0, others=0)
    for slot, others in [(0, 3), (2, 3), (3, 3), (1, 7)]:
        other = run(slot=slot, others=others)
        assert torch.equal(baseline, other), \
            f"extend attention is not position-invariant (slot={slot}, others={others})"
