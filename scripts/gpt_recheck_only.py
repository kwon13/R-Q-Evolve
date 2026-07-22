"""Run the GPT-4o re-check on ALREADY-GENERATED eval results — no GPU/vLLM.

The step eval (analysis/eval_steps_fanout.sh) can be run with GPT_RECHECK=0, which
saves every model response + math_verify score to <bench>/details.jsonl but does
NOT touch OpenAI. This script does the GPT pass afterwards, offline:

  for each global_step_*/eval/<bench>/details.jsonl
      ask gpt-4o (R-Zero results_recheck.py port) about each math_verify-WRONG row
      bump score 0 -> 1 on a "yes" verdict
      rewrite details.jsonl (adds "gpt_rechecked": true on bumped rows)
      update summary.json: pass_at_1 (final), pass_at_1_pre_gpt, gpt_flips

It REUSES GPTRechecker from scripts/eval_vllm_math.py, so the prompt, model
(gpt-4o, temp 0.1), yes/no rule, (gt,response) caching, and concurrency are
byte-identical to the in-line recheck. The OpenAI key is loaded from
R-Q-Evolve/.env (values never printed). pass@1 is unchanged when the score-0
responses don't actually match the ground truth.

Idempotent: pre-GPT accuracy is derived from rows that are correct AND not
gpt_rechecked, so re-running never double-counts earlier flips. Only score-0
rows hit the API; an already-fully-rechecked bench costs $0 the second time.

Usage:
  python analysis/gpt_recheck_only.py [BASE_DIR]
      [--steps 32,64,...] [--benches math500,gsm8k,...]
      [--workers 8] [--dry-run] [--no-collect]

  BASE_DIR default: rq_output/rq_evolve_base_entropy
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# Reuse the exact recheck implementation + .env loader from the eval script.
# (importing eval_vllm_math pulls in transformers but NOT vllm — vllm is lazy.)
from eval_vllm_math import GPTRechecker, _load_dotenv  # noqa: E402

DEFAULT_BASE = ROOT / "rq_output" / "rq_evolve_base"

BENCHES = ["math500", "gsm8k", "amc23", "aime24", "aime25", "minerva_math", "olympiadbench"]


def _discover_steps(base: Path) -> list[int]:
    steps = []
    for d in base.glob("global_step_*"):
        m = re.fullmatch(r"global_step_(\d+)", d.name)
        if m and d.is_dir():
            steps.append(int(m.group(1)))
    return sorted(steps)


def _load_rows(details_path: Path) -> list[dict]:
    return [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _pre_gpt_correct(rows: list[dict]) -> int:
    """math_verify-only correct count: correct AND not previously GPT-bumped."""
    return sum(1 for r in rows if r["score"] >= 0.5 and not r.get("gpt_rechecked"))


def recheck_bench(eval_dir: Path, bench: str, recheck: GPTRechecker, *, dry_run: bool) -> dict | None:
    details_path = eval_dir / bench / "details.jsonl"
    if not details_path.is_file():
        return None
    rows = _load_rows(details_path)
    if not rows:
        return None

    n = len(rows)
    pre_correct = _pre_gpt_correct(rows)
    targets = [r for r in rows if r["score"] < 0.5]

    if dry_run:
        uniq = len({(r["answer"], r["response"]) for r in targets})
        print(f"    {bench:14s} n={n:5d}  pre@1={pre_correct / n * 100:6.2f}%  "
              f"score0_rows={len(targets):5d}  unique_gpt_calls={uniq:5d}  [dry-run]")
        return None

    flips = recheck.recheck(rows)  # mutates score / adds gpt_rechecked on targets
    post_correct = sum(r["score"] for r in rows)

    # Rewrite details.jsonl with the bumped rows.
    with details_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Update / create summary.json for this bench.
    summary_path = eval_dir / bench / "summary.json"
    summary = {}
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
    summary.setdefault("benchmarks", {})
    summary["gpt_recheck"] = True
    summary["benchmarks"][bench] = {
        **summary["benchmarks"].get(bench, {}),
        "num_examples": n,
        "pass_at_1": post_correct / n,
        "pass_at_1_pre_gpt": pre_correct / n,
        "gpt_flips": post_correct - pre_correct,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"    {bench:14s} n={n:5d}  pre@1={pre_correct / n * 100:6.2f}%  "
          f"-> final={post_correct / n * 100:6.2f}%  (+{post_correct - pre_correct} flips)")
    return summary["benchmarks"][bench]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", nargs="?", default=str(DEFAULT_BASE), help="run base dir (has global_step_*/)")
    ap.add_argument("--steps", default="", help="comma-separated steps; default = all found")
    ap.add_argument("--benches", default=",".join(BENCHES), help="comma-separated benchmark names")
    ap.add_argument("--workers", type=int, default=8, help="gpt-4o concurrency (default 8)")
    ap.add_argument("--dry-run", action="store_true", help="report #API calls per bench, change nothing")
    ap.add_argument("--no-collect", action="store_true", help="skip running collect_scores.py at the end")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    if not base.is_dir():
        sys.exit(f"base dir not found: {base}")

    _load_dotenv(ROOT / ".env")

    steps = [int(s) for s in args.steps.split(",") if s.strip()] or _discover_steps(base)
    benches = [b.strip() for b in args.benches.split(",") if b.strip()]
    if not steps:
        sys.exit(f"no global_step_* dirs under {base}")

    print(f"base   : {base}")
    print(f"steps  : {steps}")
    print(f"benches: {benches}")
    print(f"mode   : {'DRY-RUN (no API calls, no writes)' if args.dry_run else f'live, {args.workers} gpt-4o workers'}\n")

    recheck = None
    if not args.dry_run:
        recheck = GPTRechecker(workers=args.workers)  # raises if OPENAI_API_KEY missing

    for step in steps:
        eval_dir = base / f"global_step_{step}" / "eval"
        if not eval_dir.is_dir():
            print(f"global_step_{step}: no eval/ dir, skipping")
            continue
        print(f"global_step_{step}:")
        for bench in benches:
            recheck_bench(eval_dir, bench, recheck, dry_run=args.dry_run)
        print()

    if args.dry_run or args.no_collect:
        return

    # Refresh the aggregate scores.md table.
    collect = ROOT / "analysis" / "collect_scores.py"
    if collect.is_file():
        r = subprocess.run([sys.executable, str(collect), str(base)], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"wrote {base / 'scores.md'}")
        else:
            print(f"WARN: collect_scores.py failed:\n{r.stderr[-500:]}")


if __name__ == "__main__":
    main()
