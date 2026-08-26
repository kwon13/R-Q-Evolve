#!/usr/bin/env python
"""Run the PRODUCTION admission stack over the operator probe's children.

probe_operators_report.py calls a child valid when it lints and executes on
seeds 0-2. The run does not: with ``use_evaluator: false`` and
``ast_contract: enforce`` (configs/rq_evolve_4b_4gpu.yaml), a mutation child
must clear, in this order,

  1. lint_generator_source + lint_mutation_generator_source
  2. check_generator_contract          (ast_contract: enforce -> rejection)
  3. declared GROUP / SKILL validation
  4. execution on verify_seeds=5 seeds, integer answer each time
  5. archive._passes_seed_variation    (visible instances must differ)
  6. duplicate_behavior / duplicate_template / near_duplicate_template
     against the LIVE archive

and only then is a fresh instance drawn and 10 rollouts spent on it. Every
child that dies at 1-6 costs an operator a slot and never reaches a solver, so
an operator's usable yield is the survival rate through this stack -- not the
'exec' column.

    python scripts/probe_operator_gates.py --rows rq_output/probe_operators.jsonl
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.archive import MAPElitesArchive                    # noqa: E402
from rq_evolve.ast_contract import check_generator_contract       # noqa: E402
from rq_evolve.code_utils import (lint_generator_source,          # noqa: E402
                                  lint_mutation_generator_source,
                                  set_label_declarations)
from rq_evolve.concepts import validate_label_decl                # noqa: E402
from rq_evolve.program import ProblemProgram                      # noqa: E402


class _Sk(ast.NodeVisitor):
    def __init__(self): self.seq = []
    def generic_visit(self, n): self.seq.append(type(n).__name__); super().generic_visit(n)


def skel(src):
    try:
        t = ast.parse(src)
    except Exception:
        return None
    v = _Sk(); v.visit(t); return tuple(v.seq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="rq_output/probe_operators.jsonl")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--verify-seeds", type=int, default=5)
    ap.add_argument("--out", default="rq_output/probe_operator_gates.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (ROOT / args.rows).read_text().splitlines() if l.strip()]
    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    arch = MAPElitesArchive()
    arch.load(snaps[-1])
    live = [n.champion for n in arch.grid.values() if n.champion is not None]
    arch_sk = [s for s in (skel(p.source_code) for p in live) if s]

    STAGES = ["stage1", "stage2", "lint", "ast_contract", "labels",
              "executes", "seed_variation", "not_duplicate"]
    tally = {op: collections.Counter() for op in dict.fromkeys(r["op"] for r in rows)}
    out_rows = []

    for r in rows:
        op = r["op"]; t = tally[op]; t["n"] += 1
        rec = {"op": op, "i": r.get("i"), "died": None}
        if not r.get("stage1_ok"):
            rec["died"] = "stage1"; out_rows.append(rec); continue
        t["stage1"] += 1
        code = r.get("code")
        if not code:
            rec["died"] = "stage2"; out_rows.append(rec); continue
        t["stage2"] += 1
        # evolution.py:878 does this and probe_operators.py does not: stage 2 is
        # told NOT to write GROUP/SKILL, and the run appends them from the
        # stage-1 plan afterwards. Auditing the probe's raw `code` therefore
        # fails the label gate on 100% of children for a reason the run does not
        # have. Restore the step before any gate runs.
        code = set_label_declarations(code, r.get("plan_group", ""),
                                      r.get("plan_skill", ""))

        errs = lint_generator_source(code)
        errs = list(errs) + list(lint_mutation_generator_source(
            code, require_assert=False, reject_trivial_assert=False,
            reject_unbounded_sampling=False, require_answer_routes=False,
            require_canonical_instance_data=False, require_mechanical_shape=False))
        if errs:
            rec["died"] = "lint"; rec["why"] = errs[:2]; out_rows.append(rec); continue
        t["lint"] += 1

        findings = check_generator_contract(code)
        if findings:
            rec["died"] = "ast_contract"; rec["why"] = [str(f) for f in findings[:2]]
            out_rows.append(rec); continue
        t["ast_contract"] += 1

        prog = ProblemProgram(source_code=code, metadata={"op": "mutation"})
        lab = validate_label_decl(prog.declared_group(), prog.declared_skill())
        if lab:
            rec["died"] = "labels"; rec["why"] = lab[:2]; out_rows.append(rec); continue
        t["labels"] += 1

        insts = [prog.execute(seed=z) for z in range(args.verify_seeds)]
        if any(i is None for i in insts):
            rec["died"] = "executes"; rec["why"] = [prog.last_execution_error or ""]
            out_rows.append(rec); continue
        t["executes"] += 1

        if not arch._passes_seed_variation(prog, n_seeds=args.verify_seeds):
            rec["died"] = "seed_variation"; out_rows.append(rec); continue
        t["seed_variation"] += 1

        dup = (arch._find_duplicate_behavior(prog) or arch._find_duplicate_template(prog))
        near = arch._find_near_duplicate_template(prog)
        if dup is not None or near is not None:
            rec["died"] = "not_duplicate"
            rec["why"] = ["behavior/template dup" if dup is not None
                          else f"near-dup ratio {near[1]:.3f}"]
            out_rows.append(rec); continue
        t["not_duplicate"] += 1

        s = skel(code); p = skel(r["parent_src"])
        rec["skel"] = None if s is None else hash(s)
        rec["sim_parent"] = (difflib.SequenceMatcher(None, s, p, autojunk=False).ratio()
                             if s and p else None)
        rec["sim_archive_max"] = (max((difflib.SequenceMatcher(None, s, a,
                                       autojunk=False).ratio() for a in arch_sk),
                                      default=0.0) if s else None)
        rec["cell_moved"] = (str(r.get("plan_skill")) != str(
            arch.cell_labels(tuple(r["parent_cell"]))[1]))
        out_rows.append(rec)

    hdr = f"{'operator':<14}{'n':>4}" + "".join(f"{s[:9]:>11}" for s in STAGES) + f"{'yield':>8}"
    print(hdr); print("-" * len(hdr))
    for op, t in tally.items():
        print(f"{op:<14}{t['n']:>4}" + "".join(f"{t[s]:>11}" for s in STAGES)
              + f"{t['not_duplicate']/max(1,t['n']):>7.0%}")
    print("\ndeath causes:")
    causes = collections.Counter(r["died"] for r in out_rows if r["died"])
    for c, k in causes.most_common():
        print(f"  {c:<18}{k:>4}")
    surv = [r for r in out_rows if r["died"] is None]
    print(f"\n{len(surv)}/{len(out_rows)} children clear the production stack "
          f"(probe_operators_report.py counted 121 as 'valid')")
    (ROOT / args.out).write_text(json.dumps(out_rows, indent=2))
    print(f"written to {ROOT/args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
