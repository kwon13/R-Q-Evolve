#!/usr/bin/env python
"""Freeze one (problem, answer, declared GROUP, declared SKILL) corpus to judge.

Two sources, and the difference matters. The seed programs are hand-labelled,
so a disagreement there is the judge's. The children are labelled by the Evolver
itself, so a disagreement there is the question the gate exists to answer. Both
are judged by the same rubrics in the same run, which is what makes the two
numbers comparable.

    python scripts/build_judge_corpus.py --sweep rq_output/temp_sweep
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive  # noqa: E402
from rq_evolve.code_utils import extract_generator_code  # noqa: E402
from rq_evolve.config import EvolutionConfig  # noqa: E402
from rq_evolve.evolution import RQEvolver  # noqa: E402
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import MUTATION_OP  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--sweep", default="rq_output/temp_sweep")
    ap.add_argument("--out", default="rq_output/judge_corpus.json")
    args = ap.parse_args()

    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=None,
        evolution_config=EvolutionConfig(),
    )
    rows: list[dict] = []
    seen_sources: set[str] = set()

    for path in sorted(Path(args.seed_dir).glob("*.py")):
        program = ProblemProgram.from_file(path)
        inst = program.execute(seed=0)
        if inst is None:
            continue
        rows.append({
            "source": "seed",
            "name": path.name,
            "problem": inst.problem,
            "answer": inst.answer,
            "declared_group": program.declared_group(),
            "declared_skill": program.declared_skill(),
        })

    for log in sorted(Path(args.sweep).glob("*/generations.jsonl")):
        arm = log.parent.name
        for i, line in enumerate(log.read_text().splitlines()):
            if not line.strip():
                continue
            payload = json.loads(line)
            if payload.get("stage") != "code":
                continue
            code = extract_generator_code(payload.get("text") or "")
            if code is None or code in seen_sources:
                continue
            child = ProblemProgram(source_code=code, metadata={"op": MUTATION_OP})
            inst, _ = evolver.verify_program(child)
            if inst is None:
                continue
            seen_sources.add(code)
            rows.append({
                "source": "child",
                "name": f"{arm}#{i}",
                "problem": inst.problem,
                "answer": inst.answer,
                "declared_group": child.declared_group(),
                "declared_skill": child.declared_skill(),
            })

    n_seed = sum(1 for r in rows if r["source"] == "seed")
    print(f"corpus: {n_seed} seeds + {len(rows) - n_seed} verified children = {len(rows)}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
