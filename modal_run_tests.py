"""Run the test suite on a Modal GPU.

Usage:
    modal run modal_run_tests.py                        # fast tests (not slow)
    modal run modal_run_tests.py --args "tests/ -x -q"  # full suite, incl. model downloads
"""

import shlex

import modal

app = modal.App("sid-tests")

image = (
    # devel image: the test suite JIT-compiles sid/csrc/flash_decode.cu, so we
    # need nvcc + gcc at runtime. CUDA 12.6 to match the cu126 torch wheels.
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12"
    )
    .pip_install("torch>=2.5", index_url="https://download.pytorch.org/whl/cu126")
    .pip_install(
        "ninja",  # required by torch.utils.cpp_extension.load (sid/kernels/jit.py)
        "transformers>=4.51",
        "safetensors",
        "huggingface_hub",
        "numpy",
        "fastapi",
        "uvicorn",
        "pydantic>=2",
        "pytest",
        "requests",
    )
    .add_local_dir(
        ".",
        remote_path="/root/app",
        ignore=[".venv", ".venv-local", ".git", "**/__pycache__", "*.pyc"],
    )
)

hf_cache = modal.Volume.from_name("sid-hf-cache", create_if_missing=True)
ext_cache = modal.Volume.from_name("sid-torch-ext-cache", create_if_missing=True)


@app.function(
    gpu="A10G",
    image=image,
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/torch_extensions": ext_cache,
    },
)
def run_tests(args: list[str]) -> int:
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    env["TORCH_CUDA_ARCH_LIST"] = "8.6"  # A10G; jit.py defaults to 9.0 (H100)
    env["TORCH_EXTENSIONS_DIR"] = "/root/.cache/torch_extensions"
    r = subprocess.run(
        [sys.executable, "-m", "pytest", *args],
        cwd="/root/app",
        env=env,
    )
    return r.returncode


@app.local_entrypoint()
def main(args: str = "tests/ -x -q -m 'not slow'"):
    rc = run_tests.remote(shlex.split(args))
    if rc != 0:
        raise SystemExit(rc)
    print("ALL TESTS PASSED")
