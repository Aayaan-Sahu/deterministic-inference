#!/usr/bin/env python3
"""Standalone vLLM batch-composition determinism demo.

Self-contained: no other files needed — copy this single script anywhere
(e.g. into your vllm clone) and run it with a python that has vLLM installed:

    python vllm_standalone_demo.py
    python vllm_standalone_demo.py --max-tokens 512 --trials 3
    python vllm_standalone_demo.py --output-json /tmp/vllm_results.json

What it shows: one "watched" prompt is submitted inside several different
batch compositions (alone / batch of 4 in different orders / different
members / a 16-wide mix). With greedy sampling the output SHOULD be a pure
function of the prompt — but fast GPU kernels choose their floating-point
reduction strategy (attention split-KV counts, GEMM algorithms) based on the
BATCH shape, so the watched prompt's output can change with what else is
co-scheduled. The table prints the SHA-256 of the watched output per run and
the first divergent token vs the baseline.

Note: permuting the same prompt set at the same batch size usually does NOT
diverge (kernels are position-invariant); the divergence driver is batch
size/membership — hence the size-1 and 16-wide compositions.

The --output-json file is compatible with small-inference-deterministic's
`demo/demo_determinism.py --vllm-results` for the combined 4-way table.

Prompts/compositions are identical to small-inference-deterministic/demo/prompts.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

WATCHED = (
    "Give a concise two-sentence explanation of why deterministic LLM "
    "inference matters for production systems."
)

FILLERS = [
    "Write a short haiku about a database migration finishing cleanly.",
    "List three practical checks before deploying a model-serving change.",
    "Explain request batching to a new ML infrastructure engineer.",
    "Summarize the idea of speculative decoding in three sentences.",
    "Describe the water cycle in exactly four sentences.",
    "Explain what a page table does in an operating system.",
    "Write two sentences about why GPUs are good at matrix multiplication. "
    * 8,  # long filler: stretches prefill shapes
]

PROMPTS = [WATCHED] + FILLERS

# Index lists into PROMPTS; 0 is the watched prompt.
COMPOSITIONS = [
    ("alone",          [0]),
    ("batch4-a",       [0, 1, 2, 3]),
    ("batch4-b",       [3, 0, 1, 2]),      # same set, different order
    ("batch4-c",       [2, 1, 4, 0]),      # different member set
    ("batch6",         [0, 1, 2, 3, 4, 5]),
    ("batch16-mixed",  [0] + [1, 2, 3, 4, 5, 6, 7] * 2 + [6]),
]


def first_divergence(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234,
                    help="sampling seed (used when --temperature > 0)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--output-json", default=None,
                    help="write rows compatible with demo_determinism.py --vllm-results")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16",
              gpu_memory_utilization=args.gpu_memory_utilization)
    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=True,
        seed=args.seed if args.temperature > 0 else None,
    )

    rows = []
    print(f"\nmodel={args.model}  temperature={args.temperature}  "
          f"max_tokens={args.max_tokens}  trials={args.trials}", file=sys.stderr)
    for name, comp in COMPOSITIONS:
        for trial in range(args.trials):
            prompts = [PROMPTS[i] for i in comp]
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0
            watched = outs[comp.index(0)]
            token_ids = list(watched.outputs[0].token_ids)
            total = sum(len(o.outputs[0].token_ids) for o in outs)
            rows.append({
                "engine": "vllm",
                "composition": name,
                "trial": trial,
                "sha256": hashlib.sha256(str(token_ids).encode()).hexdigest(),
                "token_ids": token_ids,
                "text": watched.outputs[0].text,
                "total_tokens": total,
                "seconds": dt,
            })
            print(f"  {name:>14} trial {trial}: {rows[-1]['sha256'][:12]}  "
                  f"({total / dt:7.1f} tok/s)", file=sys.stderr)

    # ---- report ----------------------------------------------------------
    baseline = rows[0]
    hashes = {r["sha256"] for r in rows}
    print("\n" + "=" * 74)
    print("vLLM: WATCHED PROMPT ACROSS BATCH COMPOSITIONS")
    print("=" * 74)
    for r in rows:
        mark = "==" if r["sha256"] == baseline["sha256"] else "!!"
        div = first_divergence(baseline["token_ids"], r["token_ids"])
        extra = f"  first divergent token @ {div}" if div >= 0 else ""
        print(f"  {mark} {r['composition']:>14} trial {r['trial']}: "
              f"{r['sha256'][:16]}{extra}")

    print("\n" + "=" * 74)
    if len(hashes) == 1:
        print("VERDICT: vLLM was DETERMINISTIC on this workload "
              f"({len(rows)} runs, 1 unique hash).")
        print("Nondeterminism is probabilistic — a token only flips when "
              "drift crosses a close top-2 logit gap. Retry with "
              "--max-tokens 512 --trials 5 to give it more chances.")
    else:
        print(f"VERDICT: vLLM was NON-DETERMINISTIC — {len(hashes)} distinct "
              f"outputs for the same prompt across {len(rows)} runs, purely "
              "from batch composition.")
    print("=" * 74)

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump({"engine": "vllm", "rows": rows}, f)
        print(f"\nJSON written to {args.output_json} "
              "(usable with demo_determinism.py --vllm-results)")


if __name__ == "__main__":
    main()
