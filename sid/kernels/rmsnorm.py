"""RMSNorm Triton kernels: one program per row, fp32 accumulation.

Row-wise with a fixed in-row reduction order => both batch-invariant and
position-invariant by construction, so these are safe in every mode.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_kernel(X, W, Out, hidden: tl.constexpr, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden
    x = tl.load(X + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(Out + row * hidden + offs, y.to(Out.dtype.element_ty), mask=mask)


@triton.jit
def _fused_add_rmsnorm_kernel(X, Residual, W, hidden: tl.constexpr, eps,
                              BLOCK: tl.constexpr):
    """residual += x  (stored);  x = rmsnorm(residual)  (stored in-place)."""
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    mask = offs < hidden
    x = tl.load(X + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(Residual + row * hidden + offs, mask=mask, other=0.0).to(tl.float32)
    res = res + x
    tl.store(Residual + row * hidden + offs, res.to(Residual.dtype.element_ty), mask=mask)
    var = tl.sum(res * res, axis=0) / hidden
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + offs, mask=mask, other=0.0).to(tl.float32)
    y = res * rstd * w
    tl.store(X + row * hidden + offs, y.to(X.dtype.element_ty), mask=mask)


def rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """x: [T, hidden] (or any 2D-viewable last-dim-contiguous tensor)."""
    shape = x.shape
    x2 = x.view(-1, shape[-1])
    assert x2.is_contiguous()
    out = torch.empty_like(x2)
    hidden = x2.shape[1]
    _rmsnorm_kernel[(x2.shape[0],)](
        x2, weight, out, hidden, eps, BLOCK=triton.next_power_of_2(hidden)
    )
    return out.view(shape)


def fused_add_rmsnorm(x: torch.Tensor, residual: torch.Tensor,
                      weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """In-place: residual += x; x = rmsnorm(residual). Returns (x, residual)."""
    assert x.is_contiguous() and residual.is_contiguous()
    assert x.shape == residual.shape
    hidden = x.shape[-1]
    x2 = x.view(-1, hidden)
    r2 = residual.view(-1, hidden)
    _fused_add_rmsnorm_kernel[(x2.shape[0],)](
        x2, r2, weight, hidden, eps, BLOCK=triton.next_power_of_2(hidden)
    )
    return x, residual


def qk_head_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """Per-head RMSNorm for Qwen3 QK-norm. x: [T, H, head_dim] -> same shape."""
    t, h, d = x.shape
    return rmsnorm(x.view(t * h, d), weight, eps).view(t, h, d)
