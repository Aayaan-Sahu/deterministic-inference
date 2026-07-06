#!/usr/bin/env python3
"""Live customer demo: Gemini is NON-deterministic even at temperature 0.

Story for the room
-------------------
We send Google the *exact same* request N times: same prompt, temperature 0
(greedy), no seed, one request each. Nothing on our side varies. Yet the
answers come back different — because your request is batched server-side with
other users' traffic, and fast GPU kernels pick their floating-point reduction
order from the batch shape. That drift is invisible to you and impossible to
control from the client. Contrast with brainfish_demo.py: our engine pins one
hash no matter the batch.

The output file gives two independent ways to see it on screen:
    1. the SHA-256 hash of each run   (machine check — count the unique ones)
    2. the returned text              (human check — read where they differ)

Setup:
    pip install google-genai
    export GEMINI_API_KEY=...        # or GOOGLE_API_KEY

Run:
    python demo/gemini_demo.py
    python demo/gemini_demo.py --n 16 --max-tokens 512
    python demo/gemini_demo.py --model gemini-2.5-pro   # bigger MoE, drifts more

Demo notes
----------
* Nondeterminism here is PROBABILISTIC. A token only flips when server drift
  crosses a close top-2 logit gap. If all N agree, raise --n and --max-tokens,
  or switch to --model gemini-2.5-pro. Longer, more open-ended prompts diverge
  more readily than short factual ones.
* No seed is set on purpose. Setting one lets a skeptic say "just use a seed" —
  and seeds only pin sampling, not the batch-driven reduction order, so it
  still drifts. Leave it off and make that point verbally.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# ===========================================================================
# EDIT THIS LINE LIVE with the prompt the customer gives you.
# Keep it identical to WATCHED in brainfish_demo.py for the side-by-side.
WATCHED = (
    "Give a concise two-sentence explanation of why deterministic LLM "
    "inference matters for production systems."
)
# ===========================================================================


def make_client():
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Set GEMINI_API_KEY (or GOOGLE_API_KEY) in the environment.")
    try:
        from google import genai
    except ImportError:
        sys.exit("google-genai not installed. Run: pip install google-genai")
    return genai.Client(api_key=api_key)


def one_call(client, args, idx: int) -> dict:
    """One identical request. Returns text + hash, or an error record."""
    from google.genai import types

    cfg_kwargs = dict(temperature=0.0, max_output_tokens=args.max_tokens)
    if args.no_thinking:
        # Disable 2.5 "thinking": faster, and the thinking tokens aren't returned.
        cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
    config = types.GenerateContentConfig(**cfg_kwargs)

    t0 = time.perf_counter()
    try:
        resp = client.models.generate_content(
            model=args.model, contents=WATCHED, config=config
        )
        text = resp.text or ""
        return {
            "run": idx,
            "ok": True,
            "text": text,
            "sha256": hashlib.sha256(text.encode()).hexdigest(),
            "seconds": time.perf_counter() - t0,
        }
    except Exception as e:  # noqa: BLE001 — one failure shouldn't kill the demo
        return {"run": idx, "ok": False, "error": f"{type(e).__name__}: {e}",
                "seconds": time.perf_counter() - t0}


def first_divergence(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def write_report(path: Path, args, rows: list[dict]) -> None:
    ok = [r for r in rows if r["ok"]]
    hashes = {r["sha256"] for r in ok}
    baseline = ok[0] if ok else None

    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("GEMINI NON-DETERMINISM PROOF  —  same request, sent N times")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"model={args.model}  temperature=0  seed=NONE  "
                 f"max_tokens={args.max_tokens}  runs={len(rows)}")
    lines.append("Nothing on the client varies between runs — only Google's "
                 "server-side batch does.")
    lines.append("")
    lines.append("WATCHED PROMPT (given by the audience):")
    lines.append(f"  {WATCHED!r}")
    lines.append("")
    lines.append("-" * 78)
    lines.append(f"  {'run':>3}   sha256")
    for r in rows:
        if not r["ok"]:
            lines.append(f"  {r['run']:>3}   ERROR: {r['error']}")
            continue
        mark = "==" if baseline and r["sha256"] == baseline["sha256"] else "!!"
        div = first_divergence(baseline["text"], r["text"]) if baseline else -1
        extra = f"   first divergent char @ {div}" if div >= 0 else ""
        lines.append(f"  {mark} {r['run']:>3}   {r['sha256']}{extra}")
    lines.append("")
    lines.append(f"  unique hashes: {len(hashes)}   "
                 f"({len(ok)} successful runs)")
    if len(hashes) <= 1:
        lines.append("  VERDICT: did NOT diverge on this run. Raise --n / "
                     "--max-tokens, or try --model gemini-2.5-pro.")
    else:
        lines.append(f"  VERDICT: NON-DETERMINISTIC — {len(hashes)} distinct "
                     f"outputs for one identical request.")
    lines.append("")
    lines.append("  Returned text per distinct hash (read where they differ):")
    lines.append("  " + "-" * 60)
    seen: dict[str, dict] = {}
    for r in ok:
        seen.setdefault(r["sha256"], r)
    for h, r in seen.items():
        n = sum(1 for x in ok if x["sha256"] == h)
        lines.append(f"  [hash {h[:16]}  ({n}/{len(ok)} runs)]")
        for tl in r["text"].splitlines() or [""]:
            lines.append(f"    {tl}")
        lines.append("")

    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--n", type=int, default=16, help="identical requests to send")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-thinking", action="store_true", default=True,
                    help="disable 2.5 thinking (default on: faster, cleaner)")
    ap.add_argument("--thinking", dest="no_thinking", action="store_false",
                    help="allow the model to think (slower)")
    ap.add_argument("--out", default="/tmp/gemini.txt",
                    help="readable proof file (hash + returned text)")
    args = ap.parse_args()

    client = make_client()
    print(f"\nmodel={args.model}  temp=0  no-seed  n={args.n}  "
          f"max_tokens={args.max_tokens}", file=sys.stderr)
    print(f"Sending {args.n} identical requests in parallel...", file=sys.stderr)

    with ThreadPoolExecutor(max_workers=args.n) as ex:
        rows = list(ex.map(lambda i: one_call(client, args, i), range(args.n)))
    rows.sort(key=lambda r: r["run"])

    for r in rows:
        if r["ok"]:
            print(f"  run {r['run']:>3}: {r['sha256'][:16]}  "
                  f"({r['seconds']:.2f}s)", file=sys.stderr)
        else:
            print(f"  run {r['run']:>3}: ERROR {r['error']}", file=sys.stderr)

    ok = [r for r in rows if r["ok"]]
    hashes = {r["sha256"] for r in ok}
    print("\n" + "=" * 60)
    if len(hashes) <= 1:
        print(f"Gemini agreed across {len(ok)} runs (1 hash). "
              "Raise --n/--max-tokens or try gemini-2.5-pro.")
    else:
        print(f"Gemini NON-DETERMINISTIC: {len(hashes)} distinct outputs "
              f"from {len(ok)} identical requests.")
    print("=" * 60)

    out = Path(args.out)
    write_report(out, args, rows)
    print(f"\nProof written to {out}")


if __name__ == "__main__":
    main()
