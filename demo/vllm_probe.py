"""vLLM half of the demo: run the watched prompt inside each batch composition
and report its output hash per (composition, trial). Run this with the vLLM
venv's python (see README); it prints one JSON document to stdout.

    .venv-vllm/bin/python demo/vllm_probe.py --model Qwen/Qwen3-0.6B \
        --max-tokens 256 --trials 3 > /tmp/vllm_results.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prompts import COMPOSITIONS, PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=0.85)
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_tokens,
                        ignore_eos=True)

    rows = []
    for name, comp in COMPOSITIONS:
        for trial in range(args.trials):
            prompts = [PROMPTS[i] for i in comp]
            t0 = time.perf_counter()
            outs = llm.generate(prompts, sp)
            dt = time.perf_counter() - t0
            watched = outs[comp.index(0)]
            token_ids = list(watched.outputs[0].token_ids)
            rows.append({
                "engine": "vllm",
                "composition": name,
                "trial": trial,
                "sha256": hashlib.sha256(str(token_ids).encode()).hexdigest(),
                "token_ids": token_ids,
                "text": watched.outputs[0].text,
                "total_tokens": sum(len(o.outputs[0].token_ids) for o in outs),
                "seconds": dt,
            })
    print(json.dumps({"engine": "vllm", "rows": rows}))


if __name__ == "__main__":
    main()
