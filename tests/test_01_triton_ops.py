"""Phase 1: Triton rmsnorm / qk-norm / rope vs fp32 torch references."""

import pytest
import torch

from tests.conftest import requires_gpu

pytestmark = requires_gpu

SHAPES_T = [1, 7, 32, 128, 4096]


def torch_rmsnorm(x, w, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)) * w.float()


@pytest.mark.parametrize("t", SHAPES_T)
def test_rmsnorm(t):
    from sid.kernels.rmsnorm import rmsnorm

    torch.manual_seed(0)
    x = torch.randn(t, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)
    ref = torch_rmsnorm(x, w, 1e-6)
    out = rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("t", SHAPES_T)
def test_fused_add_rmsnorm(t):
    from sid.kernels.rmsnorm import fused_add_rmsnorm

    torch.manual_seed(1)
    x = torch.randn(t, 1024, device="cuda", dtype=torch.bfloat16)
    res = torch.randn(t, 1024, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(1024, device="cuda", dtype=torch.bfloat16)

    ref_res = (x.float() + res.float())
    ref_out = torch_rmsnorm(ref_res.to(torch.bfloat16), w, 1e-6)

    out, new_res = fused_add_rmsnorm(x.clone(), res.clone(), w, 1e-6)
    torch.testing.assert_close(new_res.float(), ref_res, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(out.float(), ref_out, atol=3e-2, rtol=3e-2)


@pytest.mark.parametrize("t", [1, 33, 500])
def test_qk_head_rmsnorm(t):
    from sid.kernels.rmsnorm import qk_head_rmsnorm

    torch.manual_seed(2)
    x = torch.randn(t, 16, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, device="cuda", dtype=torch.bfloat16)
    ref = torch_rmsnorm(x, w, 1e-6)
    out = qk_head_rmsnorm(x, w, 1e-6)
    torch.testing.assert_close(out.float(), ref, atol=2e-2, rtol=2e-2)


@pytest.mark.parametrize("t", [1, 5, 257])
def test_rope_matches_hf_convention(t):
    """Reference: NeoX rotate-half exactly as HF modeling_qwen3 applies it."""
    from sid.kernels.rope import RotaryEmbedding

    torch.manual_seed(3)
    head_dim, theta, max_pos = 128, 1e6, 8192
    rope = RotaryEmbedding(head_dim, max_pos, theta, "cuda")

    q = torch.randn(t, 16, head_dim, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(t, 8, head_dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, max_pos, (t,), device="cuda", dtype=torch.int64)

    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device="cuda") / half))
    freqs = positions.float()[:, None] * inv_freq[None, :]
    cos = freqs.cos()[:, None, :]  # [t, 1, half]
    sin = freqs.sin()[:, None, :]

    def ref_apply(x):
        xf = x.float()
        x1, x2 = xf[..., :half], xf[..., half:]
        return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)

    ref_q, ref_k = ref_apply(q), ref_apply(k)
    rope.apply(q, k, positions)  # in-place
    torch.testing.assert_close(q.float(), ref_q, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(k.float(), ref_k, atol=2e-2, rtol=2e-2)
