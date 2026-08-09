"""Collect problem texts from all unique champions in an R_Q-Evolve run."""
import argparse, json, re
from pathlib import Path

from rq_evolve.program import ProblemProgram
from viz_common import load_snapshots, operator_of

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rq-output", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--max-outer-iteration", type=int, default=None)
    args = ap.parse_args()
    archive = args.rq_output / "rq_archive"
    snapshots = load_snapshots(archive, args.max_outer_iteration)
    unique = {}
    for _iteration, payload in snapshots:
        for program in payload.get("champions", []):
            unique.setdefault(program["program_id"], program)
    programs = list(unique.values())
    for program in programs:
        candidate = ProblemProgram.from_dict(program)
        texts = []
        for seed in args.seeds:
            instance = candidate.execute(seed, timeout=args.timeout)
            if instance is not None:
                texts.append(instance.problem)
        program["problem_texts"] = texts
        program["op"] = operator_of(program)
        program["group"] = (program.get("metadata") or {}).get("concept_group", "")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(programs, ensure_ascii=False), encoding="utf-8")
    print(f"saved {sum(bool(p['problem_texts']) for p in programs)}/{len(programs)} programs to {args.output}")

if __name__ == "__main__":
    main()
