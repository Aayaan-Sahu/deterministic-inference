"""The determinism demo: same watched prompt, different batch compositions.

    Stage (a) stock vLLM          -> output CHANGES with batch composition
    Stage (b) sid --mode nondet   -> also changes (same class of fast kernels)
    Stage (c) sid --mode dvr      -> SHA-256 identical EVERY time (LLM-42)
    Stage (d) sid --mode batch_invariant -> identical, but slower

Run on the GPU box:
    # vLLM half (separate venv; skip with --skip-vllm)
    .venv-vllm/bin/python demo/vllm_probe.py > /tmp/vllm_results.json
    # our engine + final report
    python demo/demo_determinism.py --vllm-results /tmp/vllm_results.json

Options: --modes nondet,dvr,batch_invariant --max-tokens 256 --trials 3
         --temperature 0.7 (seeded-sampling variant)
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import COMPOSITIONS, PROMPTS  # noqa: E402


def run_sid_mode(mode: str, args) -> list[dict]:
    import torch

    from sid.config import EngineConfig, Mode, SamplingParams
    from sid.engine.engine import Engine

    print(f"\n=== sid --mode {mode} ===", file=sys.stderr)
    cfg = EngineConfig(model_path=args.model, mode=Mode(mode),
                       kv_pool_tokens=args.kv_pool_tokens,
                       dvr_window_size=args.window, dvr_group_size=args.group)
    engine = Engine(cfg)

    rows = []
    for name, comp in COMPOSITIONS:
        for trial in range(args.trials):
            prompts = [PROMPTS[i] for i in comp]
            params = [
                SamplingParams(
                    max_new_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=0.95 if args.temperature > 0 else 1.0,
                    seed=1234,
                    is_deterministic=(mode == "dvr"),
                    ignore_eos=True,
                ) for _ in prompts
            ]
            t0 = time.perf_counter()
            outs = engine.generate(prompts, params)
            dt = time.perf_counter() - t0
            watched = outs[comp.index(0)]
            rows.append({
                "engine": f"sid-{mode}",
                "composition": name,
                "trial": trial,
                "sha256": hashlib.sha256(str(watched.token_ids).encode()).hexdigest(),
                "token_ids": watched.token_ids,
                "text": watched.text,
                "total_tokens": sum(len(o.token_ids) for o in outs),
                "seconds": dt,
                "rollbacks": watched.num_rollbacks,
                "windows": watched.num_windows,
            })
            print(f"  {name:>14} trial {trial}: "
                  f"{rows[-1]['sha256'][:12]}  ({rows[-1]['total_tokens']/dt:7.1f} tok/s)",
                  file=sys.stderr)

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def first_divergence(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def report(all_rows: list[dict]) -> None:
    engines = []
    for r in all_rows:
        if r["engine"] not in engines:
            engines.append(r["engine"])

    print("\n" + "=" * 78)
    print("WATCHED PROMPT ACROSS BATCH COMPOSITIONS — output hash per run")
    print("=" * 78)
    verdicts = {}
    for eng in engines:
        rows = [r for r in all_rows if r["engine"] == eng]
        baseline = rows[0]
        hashes = {r["sha256"] for r in rows}
        deterministic = len(hashes) == 1
        verdicts[eng] = deterministic
        tput = sum(r["total_tokens"] for r in rows) / sum(r["seconds"] for r in rows)

        print(f"\n--- {eng}   (throughput ~{tput:.0f} tok/s)")
        for r in rows:
            div = first_divergence(baseline["token_ids"], r["token_ids"])
            mark = "==" if r["sha256"] == baseline["sha256"] else "!!"
            extra = ""
            if div >= 0:
                extra = f"  first divergent token @ {div}"
            if "windows" in r and r.get("windows"):
                extra += f"  [windows={r['windows']} rollbacks={r['rollbacks']}]"
            print(f"  {mark} {r['composition']:>14} trial {r['trial']}: "
                  f"{r['sha256'][:16]}{extra}")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for eng, det in verdicts.items():
        expected_det = eng in ("sid-dvr", "sid-batch_invariant")
        status = "DETERMINISTIC" if det else "NON-DETERMINISTIC"
        note = ""
        if det == expected_det:
            note = "(as expected)" if expected_det else \
                "(as expected — fast kernels adapt to batch composition)"
        elif not det and expected_det:
            note = "*** BUG: this mode must be deterministic ***"
        else:
            note = "(did not diverge on this workload; try longer outputs / more fillers)"
        print(f"  {eng:>22}: {status:<18} {note}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--modes", default="nondet,dvr,batch_invariant")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--kv-pool-tokens", type=int, default=64 * 1024)
    ap.add_argument("--vllm-results", default=None,
                    help="JSON from demo/vllm_probe.py (run it first, separate venv)")
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    all_rows: list[dict] = []
    if args.vllm_results:
        data = json.loads(Path(args.vllm_results).read_text())
        all_rows.extend(data["rows"])

    for mode in args.modes.split(","):
        all_rows.extend(run_sid_mode(mode.strip(), args))

    report(all_rows)
    if args.output_json:
        slim = [{k: v for k, v in r.items() if k != "token_ids"} for r in all_rows]
        Path(args.output_json).write_text(json.dumps(slim, indent=2))
        print(f"\nresults written to {args.output_json}")


if __name__ == "__main__":
    main()
