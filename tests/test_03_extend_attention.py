"""Phase 3: Triton extend attention vs fp32 SDPA reference.

Covers fresh prefill (prefix 0), verify-shaped batches (prefix > 0, extend =
window), mixed varlen batches, and the position-invariance property that the
DVR determinism argument relies on.
"""

import pytest
import torch

from tests.conftest import ref_attention, requires_gpu

pytestmark = requires_gpu

HQ, HKV, D = 16, 8, 128
SCALE = D ** -0.5


def _make_batch(specs, seed=0):
    """specs: list of (prefix_len, extend_len)."""
    torch.manual_seed(seed)
    total_prefix = sum(p for p, _ in specs)
    total_extend = sum(e for _, e in specs)
    num_slots = total_prefix + 16

    q = torch.randn(total_extend, HQ, D, device="cuda", dtype=torch.bfloat16)
    k_ext = torch.randn(total_extend, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_ext = torch.randn(total_extend, HKV, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(num_slots, HKV, D, device="cuda", dtype=torch.bfloat16)

    perm = torch.randperm(num_slots - 1, device="cuda")[:total_prefix] + 1
    kv_indices = perm.to(torch.int32)
    qo = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
    qo[1:] = torch.cumsum(torch.tensor([e for _, e in specs], device="cuda"), 0).to(torch.int32)
    kvp = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
    kvp[1:] = torch.cumsum(torch.tensor([p for p, _ in specs], device="cuda"), 0).to(torch.int32)
    return q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kv_indices


def _reference(specs, q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kv_indices):
    outs = []
    for i, (p, e) in enumerate(specs):
        qi = q[qo[i]:qo[i + 1]]
        idx = kv_indices[kvp[i]:kvp[i + 1]].long()
        k_full = torch.cat([k_buf[idx], k_ext[qo[i]:qo[i + 1]]])
        v_full = torch.cat([v_buf[idx], v_ext[qo[i]:qo[i + 1]]])
        # query j (extend-local) sees prefix + extend tokens <= j.
        outs.append(ref_attention(qi, k_full, v_full, SCALE, causal_offset=p))
    return torch.cat(outs)


@pytest.mark.parametrize("specs", [
    [(0, 1)],
    [(0, 32)], [(0, 64)], [(0, 100)], [(0, 500)],
    [(1, 1)], [(100, 32)], [(1000, 64)],
    [(100, 32), (0, 17), (999, 1), (5, 64)],
    [(31, 32), (32, 32), (33, 32), (256, 32)],  # exact G x W verify shape
])
def test_extend_vs_reference(specs):
    from sid.kernels.extend_attention import extend_attention_fwd

    tensors = _make_batch(specs)
    q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kv_indices = tensors
    out = extend_attention_fwd(q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kv_indices,
                               max(e for _, e in specs), SCALE)
    ref = _reference(specs, *tensors)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)


def test_position_invariance_within_batch():
    """A sequence's output must be bitwise identical no matter which batch
    slot it occupies and what else is co-batched — the property the DVR
    verifier's determinism induction rests on."""
    from sid.kernels.extend_attention import extend_attention_fwd

    torch.manual_seed(7)
    p, e = 100, 32
    q0 = torch.randn(e, HQ, D, device="cuda", dtype=torch.bfloat16)
    k0 = torch.randn(e, HKV, D, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn(e, HKV, D, device="cuda", dtype=torch.bfloat16)
    k_buf = torch.randn(p + 500, HKV, D, device="cuda", dtype=torch.bfloat16)
    v_buf = torch.randn(p + 500, HKV, D, device="cuda", dtype=torch.bfloat16)
    idx0 = (torch.randperm(p + 499, device="cuda")[:p] + 1).to(torch.int32)

    def run(slot: int, others: int):
        """Place the watched sequence at `slot` among `others` co-batched seqs."""
        specs = []
        qs, ks, vs, idxs = [], [], [], []
        torch.manual_seed(1234)
        for i in range(others + 1):
            if i == slot:
                specs.append((p, e))
                qs.append(q0); ks.append(k0); vs.append(v0); idxs.append(idx0)
            else:
                pe, ee = 40 + 13 * i, 8 + i
                specs.append((pe, ee))
                qs.append(torch.randn(ee, HQ, D, device="cuda", dtype=torch.bfloat16))
                ks.append(torch.randn(ee, HKV, D, device="cuda", dtype=torch.bfloat16))
                vs.append(torch.randn(ee, HKV, D, device="cuda", dtype=torch.bfloat16))
                idxs.append(torch.randint(1, p + 500, (pe,), device="cuda", dtype=torch.int32))
        q = torch.cat(qs); k_ext = torch.cat(ks); v_ext = torch.cat(vs)
        kv_indices = torch.cat(idxs)
        qo = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
        qo[1:] = torch.cumsum(torch.tensor([s[1] for s in specs], device="cuda"), 0).to(torch.int32)
        kvp = torch.zeros(len(specs) + 1, dtype=torch.int32, device="cuda")
        kvp[1:] = torch.cumsum(torch.tensor([s[0] for s in specs], device="cuda"), 0).to(torch.int32)
        out = extend_attention_fwd(q, k_ext, v_ext, k_buf, v_buf, qo, kvp, kv_indices,
                                   max(s[1] for s in specs), SCALE)
        return out[qo[slot]:qo[slot] + e].clone()

    baseline = run(slot=0, others=0)
    for slot, others in [(0, 3), (2, 3), (3, 3), (1, 7)]:
        other = run(slot=slot, others=others)
        assert torch.equal(baseline, other), \
            f"extend attention is not position-invariant (slot={slot}, others={others})"
