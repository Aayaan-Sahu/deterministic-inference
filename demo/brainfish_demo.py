#!/usr/bin/env python3
"""Live customer demo: OUR engine is deterministic across batch composition.

Story for the room
-------------------
The audience gives you a prompt. You paste it into WATCHED below. We run that
exact prompt inside many *different* batch shapes (alone, batched with 3, 6,
16 other requests, reordered, different members) — the thing that makes stock
vLLM / Gemini / OpenAI drift. With `sid` in deterministic (dvr) mode every run
returns the SHA-256-identical token stream. The output file gives you two
independent ways to prove sameness on screen:

    1. the SHA-256 hash of each run   (machine check)
    2. the detokenized, readable text (human check)

Run on the GPU box:
    python demo/brainfish_demo.py
    python demo/brainfish_demo.py --out /tmp/brainfish.txt --max-tokens 512
    python demo/brainfish_demo.py --contrast     # also show nondet mode drifting

Pair it with gemini_demo.py (same WATCHED prompt) for the side-by-side.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from prompts import COMPOSITIONS, FILLERS  # noqa: E402

# ===========================================================================
# EDIT THIS LINE LIVE with the prompt the customer gives you.
# Keep it identical to WATCHED in gemini_demo.py for the side-by-side.
WATCHED = (
    "Give a concise two-sentence explanation of why deterministic LLM "
    "inference matters for production systems."
)
# ===========================================================================

# Batch members other than the watched prompt are fixed (from prompts.py).
PROMPTS = [WATCHED] + FILLERS


def run_mode(mode: str, args) -> list[dict]:
    """Run the watched prompt inside every batch composition, `trials` times."""
    import torch

    from sid.config import EngineConfig, Mode, SamplingParams
    from sid.engine.engine import Engine

    print(f"\n=== sid --mode {mode} ===", file=sys.stderr)
    cfg = EngineConfig(
        model_path=args.model,
        mode=Mode(mode),
        kv_pool_tokens=args.kv_pool_tokens,
        dvr_window_size=args.window,
        dvr_group_size=args.group,
    )
    engine = Engine(cfg)

    rows: list[dict] = []
    for name, comp in COMPOSITIONS:
        for trial in range(args.trials):
            prompts = [PROMPTS[i] for i in comp]
            params = [
                SamplingParams(
                    max_new_tokens=args.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    seed=1234,
                    is_deterministic=(mode == "dvr"),
                    ignore_eos=args.ignore_eos,
                )
                for _ in prompts
            ]
            t0 = time.perf_counter()
            outs = engine.generate(prompts, params, chat=args.chat)
            dt = time.perf_counter() - t0
            watched = outs[comp.index(0)]  # 0 is the watched prompt
            rows.append(
                {
                    "mode": mode,
                    "composition": name,
                    "batch_size": len(comp),
                    "trial": trial,
                    "sha256": hashlib.sha256(str(watched.token_ids).encode()).hexdigest(),
                    "token_ids": watched.token_ids,
                    "text": watched.text,
                    "seconds": dt,
                }
            )
            print(
                f"  {name:>14} (bs={len(comp):>2}) trial {trial}: "
                f"{rows[-1]['sha256'][:16]}",
                file=sys.stderr,
            )

    del engine
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def first_divergence(a: list, b: list) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def write_report(path: Path, sections: list[tuple[str, list[dict]]]) -> None:
    """Human-readable proof file: hash + full detokenized text per section."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("sid DETERMINISM PROOF  —  one watched prompt, many batch shapes")
    lines.append("=" * 78)
    lines.append("")
    lines.append("WATCHED PROMPT (given by the audience):")
    lines.append(f"  {WATCHED!r}")
    lines.append("")

    for title, rows in sections:
        baseline = rows[0]
        hashes = {r["sha256"] for r in rows}
        deterministic = len(hashes) == 1

        lines.append("-" * 78)
        lines.append(f"MODE: {title}")
        lines.append("-" * 78)
        lines.append(f"  {'composition':>14}  {'bs':>3}  trial   sha256")
        for r in rows:
            mark = "==" if r["sha256"] == baseline["sha256"] else "!!"
            div = first_divergence(baseline["token_ids"], r["token_ids"])
            extra = f"   first divergent token @ {div}" if div >= 0 else ""
            lines.append(
                f"  {mark} {r['composition']:>11}  {r['batch_size']:>3}  "
                f"  {r['trial']}     {r['sha256']}{extra}"
            )
        lines.append("")
        lines.append(f"  unique hashes: {len(hashes)}   "
                     f"({len(rows)} runs total)")
        if deterministic:
            lines.append("  VERDICT: DETERMINISTIC  —  every run is byte-identical.")
        else:
            lines.append("  VERDICT: NON-DETERMINISTIC  —  output changes with the batch.")
        lines.append("")
        lines.append("  Detokenized output (readable — same for every run if deterministic):")
        lines.append("  " + "-" * 60)
        # Show text per distinct hash so a divergence is visible in full.
        seen: dict[str, str] = {}
        for r in rows:
            if r["sha256"] not in seen:
                seen[r["sha256"]] = r["text"]
        for h, text in seen.items():
            lines.append(f"  [hash {h[:16]}]")
            for tl in text.splitlines() or [""]:
                lines.append(f"    {tl}")
            lines.append("")

    path.write_text("\n".join(lines))


def print_verdict(sections: list[tuple[str, list[dict]]]) -> None:
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    for title, rows in sections:
        hashes = {r["sha256"] for r in rows}
        det = len(hashes) == 1
        status = "DETERMINISTIC" if det else "NON-DETERMINISTIC"
        print(f"  {title:>26}: {status:<18} "
              f"({len(hashes)} unique hash / {len(rows)} runs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--group", type=int, default=4)
    ap.add_argument("--kv-pool-tokens", type=int, default=64 * 1024)
    ap.add_argument("--no-chat", dest="chat", action="store_false",
                    help="feed raw prompt (default: wrap in the chat template)")
    ap.add_argument("--ignore-eos", action="store_true",
                    help="keep generating past EOS (forces loops; off by default "
                         "so output stops naturally and reads cleanly)")
    ap.add_argument("--out", default="/tmp/brainfish_sid.txt",
                    help="readable proof file (hash + detokenized text)")
    ap.add_argument("--contrast", action="store_true",
                    help="also run nondet mode to show the fast kernels drifting")
    args = ap.parse_args()

    sections: list[tuple[str, list[dict]]] = []
    if args.contrast:
        sections.append(("sid nondet (fast kernels, no determinism)",
                         run_mode("nondet", args)))
    sections.append(("sid dvr (DETERMINISTIC)", run_mode("dvr", args)))

    print_verdict(sections)
    out = Path(args.out)
    write_report(out, sections)
    print(f"\nProof written to {out}")
    print("Open it on screen: same SHA-256 + same readable text across every batch.")


if __name__ == "__main__":
    main()
