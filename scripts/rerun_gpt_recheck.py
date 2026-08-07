#!/usr/bin/env python
"""Re-run the GPT-4o re-check over an eval that already has its generations.

``eval_vllm_math.py`` warns and continues when the judge call fails, so a run
that hit HTTP 429 still produces a ``summary.json`` -- with ``gpt_recheck:
true`` and ``gpt_flips: 0``, which is indistinguishable from "the judge agreed
with math_verify everywhere" unless you look. Two eval fan-outs sharing one API
key did exactly that: every ablation arm scored pre-GPT while the control it
was being compared against scored post-GPT, a ~12 point artefact.

The generations are already on disk in ``details.jsonl``, so nothing needs a
GPU here. This walks the score-0 rows, asks the judge, and rewrites
``details.jsonl`` + ``summary.json`` in place.

    python scripts/rerun_gpt_recheck.py rq_output/<run>/global_step_*/eval/*

Default concurrency is deliberately low. The failure being repaired was a rate
limit, and finishing an hour later is better than silently scoring nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# scripts/ is not a package, so import the sibling module by name.
# _load_dotenv lives inside that module's main(), so importing GPTRechecker
# alone leaves OPENAI_API_KEY unset and the constructor raises.
from eval_vllm_math import GPTRechecker, _load_dotenv  # noqa: E402


def redo(bench_dir: Path, rechecker: GPTRechecker, force: bool) -> str:
    summary_path = bench_dir / "summary.json"
    details_path = bench_dir / "details.jsonl"
    if not (summary_path.is_file() and details_path.is_file()):
        return f"{bench_dir}: missing summary.json or details.jsonl"

    summary = json.loads(summary_path.read_text())
    name = next(iter(summary.get("benchmarks", {})), None)
    if name is None:
        return f"{bench_dir}: no benchmark entry"
    entry = summary["benchmarks"][name]
    if entry.get("gpt_flips", 0) > 0 and not force:
        return f"{bench_dir.parent.parent.name}/{name}: already has flips, skipped"

    rows = [json.loads(l) for l in details_path.read_text().splitlines()]
    # pass_at_1_pre_gpt is the authority: score may already carry flips from a
    # partially successful earlier pass, and re-flipping would double-count.
    for r in rows:
        r["score"] = int(r.get("score_pre_gpt", r["score"]))
        r["score_pre_gpt"] = r["score"]
        r.pop("gpt_rechecked", None)

    n = len(rows)
    pre = sum(r["score"] for r in rows)
    flips = rechecker.recheck(rows)
    post = sum(r["score"] for r in rows)

    with details_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    entry["pass_at_1"] = post / n if n else 0.0
    entry["pass_at_1_pre_gpt"] = pre / n if n else 0.0
    entry["gpt_flips"] = flips
    summary["gpt_recheck"] = True
    summary["gpt_recheck_rerun"] = True
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    step = bench_dir.parent.parent.name
    return (f"{step}/{name}: {pre/n*100:.2f} -> {post/n*100:.2f}  (+{flips} flips)")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+", help="benchmark output dirs (with summary.json)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--force", action="store_true", help="redo even if flips > 0")
    args = p.parse_args()

    _load_dotenv(ROOT / ".env")
    rechecker = GPTRechecker(workers=args.workers)
    for d in args.dirs:
        print(redo(Path(d), rechecker, args.force), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
