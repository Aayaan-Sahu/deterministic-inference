# small-inference-deterministic (`sid`)

A deliberately small LLM inference engine whose headline feature is
**deterministic inference via LLM-42-style decode-verify-rollback** (DVR,
[arXiv 2601.17768](https://arxiv.org/abs/2601.17768)) — with the modern
fast-inference machinery intact: hand-written CUDA flash-decode attention with
split-KV, paged KV cache, continuous batching, Triton prefill/norm/RoPE
kernels, and an OpenAI-ish server. ~3k lines total, no build system: the CUDA
extension JIT-compiles on first run.

Target: one NVIDIA GPU (SM90 tuned; SM80+ works), `Qwen/Qwen3-0.6B`, bf16.

## Why LLM inference is nondeterministic, in one paragraph

Floating-point addition is not associative, and fast GPU kernels pick their
*reduction order* based on the shape of the batch: decode attention splits the
KV sequence across more blocks when the batch is small (to fill the GPU),
cuBLAS picks different GEMM algorithms for different token counts. With
continuous batching, *your* request's batch composition depends on whoever
else is queued — so the same prompt returns different tokens on different
runs. The known fix — batch-invariant kernels — pins one reduction strategy
for every shape and pays 24-63% throughput ([vLLM's `VLLM_BATCH_INVARIANT`,
SGLang's deterministic mode). LLM-42's observation: keep the fast kernels,
and enforce determinism *after the fact* with a verifier whose input shape is
fixed — replay the last W tokens, keep what matches, roll back what doesn't.
Verification is prefill-like (cheap), rollbacks are rare, and requests that
don't ask for determinism pay nothing.

## Three modes

| mode | decode kernels | deterministic? | relative speed |
|---|---|---|---|
| `nondet` | batch-adaptive split-KV heuristic | no | fastest |
| `dvr` | same fast path + fixed-shape verify/rollback | **yes** (per-request `is_deterministic`) | ~fast |
| `batch_invariant` | fixed 256-token split tiles + persistent Triton GEMM | yes (by construction) | slowest |

## Quick start (GPU box)

```bash
module load cuda                    # nvcc must be on PATH
bash setup_server.sh                # venv, deps, JIT build, fast tests
source .venv/bin/activate
python -m pytest tests/ -x -q       # full suite (downloads Qwen3-0.6B, ~1.5GB)

# offline
python - <<'EOF'
from sid.config import EngineConfig, Mode, SamplingParams
from sid.engine.engine import Engine
eng = Engine(EngineConfig(mode=Mode.DVR))
outs = eng.generate(["Why is the sky blue?"],
                    SamplingParams(max_new_tokens=64, is_deterministic=True),
                    chat=True)
print(outs[0].text)
EOF

# server
python -m sid.server --mode dvr --port 8000
curl -s localhost:8000/v1/completions -H 'content-type: application/json' -d '{
  "prompt": "Why is the sky blue?", "max_tokens": 64, "is_deterministic": true}'
```

## The demo

Shows the same watched prompt submitted inside different batch compositions
(`[1]`, `[1,4,3,2]`, `[3,0,1,2]`, …, 16-wide mixed). vLLM and `nondet` mode
produce different outputs depending on what else is in the batch; `dvr` and
`batch_invariant` produce SHA-256-identical output every single time.

```bash
bash setup_server.sh --vllm                                  # once
.venv-vllm/bin/python demo/vllm_probe.py > /tmp/vllm_results.json
python demo/demo_determinism.py --vllm-results /tmp/vllm_results.json
# seeded-sampling variant:
python demo/demo_determinism.py --temperature 0.7 --modes dvr
```

Note on batch *order* vs batch *composition*: permuting the same prompt set at
the same batch size often does NOT diverge — kernels are position-invariant
(paper Observation O2). The divergence driver is batch size/membership, which
is why the compositions include a size-1 baseline and a 16-wide mix.

## How DVR works here (short version)

- Fast path decodes normally under continuous batching (batch-adaptive
  kernels, drifting bits and all).
- Every `W=32` tokens (or at finish), a request is verified: a fixed-shape
  batch of exactly `G=4` sequences × `W` tokens (dummy-padded) replays the
  window as an extend/prefill pass. Fixed shape ⇒ same kernels, tiles, and
  reduction orders every run ⇒ the verifier is run-consistent.
- The verifier **overwrites the decode-phase KV in the same physical slots**,
  so future steps read verifier-consistent state.
- First mismatching token: output is truncated there, the verifier's token is
  accepted (≥1 token progress per pass), stale KV slots are freed, EOS state
  is recomputed. Only verified tokens are ever streamed to the client.
- Deterministic requests prefill alone (batch shape = f(prompt) only), so
  token 0 is deterministic by construction; induction does the rest.
- Sampling is position-seeded Gumbel-max (`multinomial_with_seed`): the draw
  is a hash of (seed, position), never a global RNG.

## Layout

```
sid/csrc/flash_decode.cu        CUDA: split-KV decode attention (stage1+stage2), reshape_and_cache
sid/kernels/                    Triton: extend attention, rmsnorm, rope, persistent matmul; JIT loader
sid/model/                      Qwen3 forward + safetensors loader
sid/engine/                     KV cache, requests, batches, sampler, DVR verifier, scheduler, engine
sid/server.py                   FastAPI, per-request is_deterministic
tests/                          phased: kernels -> model-vs-HF -> DVR logic (CPU) -> e2e determinism
demo/                           the 4-way determinism demo
benchmarks/                     decode-attention microbenchmark
```

## Debugging aids

- `--decode-backend triton` routes decode through the (slow, simple) Triton
  extend kernel to bisect CUDA kernel bugs.
- `SID_DEBUG_ALLOC=1` turns on double-free/foreign-free checks in the KV
  allocator.
- `tests/test_08_dvr_logic.py` runs the full rollback arithmetic on CPU — no
  GPU needed (works on a laptop).
