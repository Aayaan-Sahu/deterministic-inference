"""JIT compilation of the CUDA extension via torch.utils.cpp_extension.

No CMake, no setup.py build step: the first call compiles sid/csrc/flash_decode.cu
into TORCH_EXTENSIONS_DIR (~1-2 min), later calls reuse the cached .so.
"""

from __future__ import annotations

import os
from pathlib import Path

_ext = None

CSRC_DIR = Path(__file__).resolve().parent.parent / "csrc"


def load_ext():
    """Compile (once) and return the sid_flash_decode extension module."""
    global _ext
    if _ext is not None:
        return _ext

    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "9.0")

    from torch.utils.cpp_extension import load

    _ext = load(
        name="sid_flash_decode",
        sources=[str(CSRC_DIR / "flash_decode.cu")],
        extra_cuda_cflags=["-O3", "-std=c++17", "--expt-relaxed-constexpr"],
        extra_cflags=["-O3", "-std=c++17"],
        verbose=os.environ.get("SID_JIT_VERBOSE", "0") == "1",
    )
    return _ext


def build_all():
    """Warm-build entry point used by setup_server.sh."""
    import torch

    assert torch.cuda.is_available(), "CUDA device required to build/verify kernels"
    ext = load_ext()
    # Tiny smoke test: 1 request, seq_len 4, 1 split.
    dev = "cuda"
    num_kv, num_q, hd = 8, 16, 128
    q = torch.randn(1, num_q, hd, device=dev, dtype=torch.bfloat16)
    k_cache = torch.randn(8, num_kv, hd, device=dev, dtype=torch.bfloat16)
    v_cache = torch.randn(8, num_kv, hd, device=dev, dtype=torch.bfloat16)
    kv_indptr = torch.tensor([0, 4], device=dev, dtype=torch.int32)
    kv_indices = torch.arange(1, 5, device=dev, dtype=torch.int32)
    num_splits = torch.ones(1, device=dev, dtype=torch.int32)
    part_o = torch.empty(1, num_q, 1, hd, device=dev, dtype=torch.float32)
    part_lse = torch.empty(1, num_q, 1, device=dev, dtype=torch.float32)
    o = torch.empty_like(q)
    ext.flash_decode_fwd(
        q, k_cache, v_cache, kv_indptr, kv_indices, num_splits,
        1, 0, hd ** -0.5, part_o, part_lse, o,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(o.float()).all(), "smoke test produced non-finite output"
    print("sid_flash_decode built and smoke-tested OK "
          f"(SM {torch.cuda.get_device_capability()})")


if __name__ == "__main__":
    build_all()
