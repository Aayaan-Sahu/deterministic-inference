"""Phase 0: environment + JIT build smoke test."""

import torch

from tests.conftest import requires_gpu


@requires_gpu
def test_gpu_is_sm90_class():
    major, _minor = torch.cuda.get_device_capability()
    assert major >= 8, f"need SM80+, got {torch.cuda.get_device_capability()}"


@requires_gpu
def test_jit_builds_and_smokes():
    from sid.kernels.jit import build_all

    build_all()
