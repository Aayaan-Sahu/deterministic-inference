from __future__ import annotations

import pytest
import torch

requires_gpu = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA GPU"
)


def ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                  sm_scale: float, causal_offset: int | None = None) -> torch.Tensor:
    """fp32 reference attention for one sequence.

    q: [Tq, HQ, D]; k/v: [Tk, HKV, D]. GQA: q head h uses kv head h // (HQ//HKV).
    causal_offset: if set, query i (0-based) may attend keys j <= causal_offset + i.
    None = full visibility.
    """
    tq, hq, d = q.shape
    tk, hkv, _ = k.shape
    group = hq // hkv
    qf, kf, vf = q.float(), k.float(), v.float()
    out = torch.empty(tq, hq, d, dtype=torch.float32, device=q.device)
    for h in range(hq):
        kh = h // group
        scores = qf[:, h] @ kf[:, kh].t() * sm_scale  # [Tq, Tk]
        if causal_offset is not None:
            qi = torch.arange(tq, device=q.device).unsqueeze(1)
            kj = torch.arange(tk, device=q.device).unsqueeze(0)
            scores = scores.masked_fill(kj > causal_offset + qi, float("-inf"))
        out[:, h] = torch.softmax(scores, dim=-1) @ vf[:, kh]
    return out


def assert_close_bf16(actual: torch.Tensor, ref_fp32: torch.Tensor,
                      atol=2e-2, rtol=2e-2):
    torch.testing.assert_close(actual.float(), ref_fp32, atol=atol, rtol=rtol)
