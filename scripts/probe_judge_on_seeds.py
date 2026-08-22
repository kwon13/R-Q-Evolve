#!/usr/bin/env python
"""Calibration: run the judge over the hand-written seed corpus.

The seeds' GROUP/SKILL are written by hand and treated as ground truth
everywhere else in the pipeline, so they are the one place where a judge
disagreement is unambiguously the judge's. If the gate rejects seeds, a zero
acceptance rate on generated children says nothing about the generator.

    python scripts/probe_judge_on_seeds.py
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.openai_evaluator import (  # noqa: E402
    OpenAIEvaluatorConfig,
    evaluate_messages_with_openai,
    load_project_dotenv,
)
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import (  # noqa: E402
    build_judge_messages,
    judge_accepts,
    parse_judge_verdict,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--model", default="gpt-5.4-mini")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--max-output-tokens", type=int, default=1500)
    ap.add_argument("--instance-seed", type=int, default=0)
    ap.add_argument("--out", default="rq_output/judge_on_seeds.json")
    args = ap.parse_args()

    load_project_dotenv(ROOT)
    cfg = OpenAIEvaluatorConfig(
        model=args.model,
        reasoning_effort=args.effort,
        timeout_s=180.0,
        max_output_tokens=args.max_output_tokens,
    )

    jobs = []
    for path in sorted(Path(args.seed_dir).glob("*.py")):
        program = ProblemProgram.from_file(path)
        inst = program.execute(seed=args.instance_seed)
        if inst is None:
            print(f"{path.name}: EXECUTE FAILED")
            continue
        jobs.append((path.name, program, inst))

    def run(job):
        name, program, inst = job
        text = evaluate_messages_with_openai(
            build_judge_messages(inst.problem, inst.answer), cfg
        )
        verdict = parse_judge_verdict(text)
        ok, reason = judge_accepts(
            verdict, program.declared_group(), program.declared_skill()
        )
        return {
            "program": name,
            "declared_group": program.declared_group(),
            "declared_skill": program.declared_skill(),
            "judged_group": verdict.group,
            "judged_skill": verdict.skill,
            "accepted": ok,
            "reason": reason,
            "failure_reason": verdict.failure_reason,
            "raw": text,
        }

    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(run, jobs))

    print(f"\n{'seed program':<52}{'declared':>34}{'judged':>34}  ok")
    print("-" * 126)
    for r in rows:
        dec = f"{r['declared_group']}/{r['declared_skill']}"
        jud = f"{r['judged_group']}/{r['judged_skill']}"
        print(f"{r['program'][:50]:<52}{dec:>34}{jud:>34}  {'YES' if r['accepted'] else 'no'}")

    n_ok = sum(1 for r in rows if r["accepted"])
    n_group = sum(1 for r in rows if r["judged_group"] == r["declared_group"])
    n_skill = sum(1 for r in rows if r["judged_skill"] == r["declared_skill"])
    print(f"\nboth axes agree : {n_ok}/{len(rows)}")
    print(f"GROUP agrees    : {n_group}/{len(rows)}")
    print(f"SKILL agrees    : {n_skill}/{len(rows)}")
    print("\nwhy the rejections happened:")
    for r in rows:
        if not r["accepted"]:
            print(f"  {r['program'][:46]:<48} {str(r['failure_reason'])[:78]}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
