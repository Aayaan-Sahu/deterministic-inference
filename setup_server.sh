#!/usr/bin/env bash
# One-shot setup on a CUDA box (tested target: NCSA H200, no root needed).
#
#   bash setup_server.sh            # main venv + JIT build + fast tests
#   bash setup_server.sh --vllm     # also create the vLLM venv for the demo
#
# Prereqs: python3.10+, nvcc on PATH (e.g. `module load cuda/12.x`).
set -euo pipefail
cd "$(dirname "$0")"

export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0}"
export TORCH_EXTENSIONS_DIR="$PWD/.torch_ext_cache"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export HF_HOME="${HF_HOME:-$PWD/.hf_cache}"

if ! command -v nvcc >/dev/null; then
  echo "ERROR: nvcc not on PATH. Try: module load cuda" >&2
  exit 1
fi

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install torch --index-url https://download.pytorch.org/whl/cu126
pip -q install -r requirements.txt

echo "== building CUDA extension (first time takes 1-2 min) =="
python -m sid.kernels.jit

echo "== fast test suite (kernels + logic; model tests are -m slow) =="
python -m pytest tests/ -x -q -m "not slow"

cat <<'EOF'

Setup OK. Next steps:
  source .venv/bin/activate
  python -m pytest tests/ -x -q            # full suite (downloads Qwen3-0.6B)
  python -m sid.server --mode dvr          # OpenAI-ish server on :8000
  python demo/demo_determinism.py          # the determinism demo (our modes)

For the vLLM comparison half of the demo:
  bash setup_server.sh --vllm
  .venv-vllm/bin/python demo/vllm_probe.py > /tmp/vllm_results.json
  python demo/demo_determinism.py --vllm-results /tmp/vllm_results.json
EOF

if [ "${1:-}" = "--vllm" ]; then
  echo "== creating vLLM venv (separate: vllm pins its own torch) =="
  if [ ! -d .venv-vllm ]; then
    python3 -m venv .venv-vllm
  fi
  .venv-vllm/bin/pip -q install --upgrade pip
  .venv-vllm/bin/pip -q install vllm
  echo "vLLM venv ready: .venv-vllm"
fi
