#!/usr/bin/env python
"""Run ONE operator on its own offspring for k rounds and watch the forms run out.

A single-generation probe cannot see the failure being attacked. The collapse it
is aimed at took the live run ten generations: new AST skeletons per iteration
fell to zero by iteration ~10 and the archive has added 67 distinct skeletons in
194 iterations while holding 48 champions. An operator that looks structurally
free on one step can still be revisiting the same small set of forms -- which is
the "Mutation Without Variation" claim, and it is a claim about CHAINS.

So this closes the loop. Each round mutates the operator's OWN survivors with
the operator under test, applies the run's real admission stack (lint,
ast_contract, labels, execution on verify_seeds, seed variation, and the
archive's behaviour/template/near-duplicate gates -- against this operator's own
accumulated population, not the shipped archive), and, unless --no-select, spends
the archive's own scoring budget of ONE fresh instance x group_size rollouts so
the population is selected on real R_Q.

Sizing, from the live run rather than taste:
  --rounds 10  the deepest champion in rq_evolve_4b_4gpu is generation 11 and
               the median is 5-6, so a chain still producing new forms at round
               10 has already beaten the run.
  --pop 16     the archive held 13-20 champions across iterations 8-14, which
               is where its new-form rate hit zero. Collapse is a POPULATION
               property; a chain of one parent cannot show it.
  --children 64  the run generates 32 candidates an iteration and 11% of them
               clear the stack (scripts/probe_operator_gates.py), so 64 buys
               ~7 survivors a round -- enough that "0 new forms" is a
               measurement rather than a small-sample artifact.

--no-select is the control that separates the two candidate causes. With
selection off, admission is the gate stack alone and any collapse is the
OPERATOR revisiting forms. With it on, R_Q-greedy selection is also pulling the
population together. Run both on a finalist and the difference is the answer.

    python scripts/probe_operator_chain.py --op structural --rounds 10 \
        --pop 16 --children 64 --port 8701 --model step160
"""
from __future__ import annotations

import argparse
import ast
import collections
import difflib
import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

# The operator prompts come from the probe itself, so the chain and the
# single-generation screen are literally the same operator.
import probe_operators as PO                                       # noqa: E402
from rq_evolve.archive import MAPElitesArchive                     # noqa: E402
from rq_evolve.ast_contract import check_generator_contract        # noqa: E402
from rq_evolve.code_utils import (lint_generator_source,           # noqa: E402
                                  lint_mutation_generator_source,
                                  set_label_declarations)
from rq_evolve.concepts import GROUPS, SKILLS, validate_label_decl  # noqa: E402
from rq_evolve.program import ProblemProgram                       # noqa: E402
from rq_evolve.prompts import build_solver_messages                # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed          # noqa: E402
from rq_evolve.scoring import compute_rq_program, score_seed       # noqa: E402
from rq_evolve.solver_trace import (SOLVER_CHAT_BOUNDARY_STOPS,    # noqa: E402
                                    sanitize_solver_trace)


class _Sk(ast.NodeVisitor):
    def __init__(self): self.seq = []
    def generic_visit(self, n): self.seq.append(type(n).__name__); super().generic_visit(n)


def skel(src):
    try:
        t = ast.parse(src)
    except Exception:
        return None
    v = _Sk(); v.visit(t); return tuple(v.seq)


def sim(a, b):
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def gate(code: str, arch: MAPElitesArchive, verify_seeds: int):
    """The run's admission stack. Returns (program|None, death_reason|None)."""
    errs = list(lint_generator_source(code)) + list(lint_mutation_generator_source(
        code, require_assert=False, reject_trivial_assert=False,
        reject_unbounded_sampling=False, require_answer_routes=False,
        require_canonical_instance_data=False, require_mechanical_shape=False))
    if errs:
        return None, "lint"
    if check_generator_contract(code):
        return None, "ast_contract"
    prog = ProblemProgram(source_code=code, metadata={"op": "mutation"})
    if validate_label_decl(prog.declared_group(), prog.declared_skill()):
        return None, "labels"
    if any(prog.execute(seed=z) is None for z in range(verify_seeds)):
        return None, "executes"
    if not arch._passes_seed_variation(prog, n_seeds=verify_seeds):
        return None, "seed_variation"
    if (arch._find_duplicate_behavior(prog) or arch._find_duplicate_template(prog)
            or arch._find_near_duplicate_template(prog)):
        return None, "duplicate"
    return prog, None


def solve(port, model, problem, m, tokens, timeout=900):
    import urllib.request
    body = json.dumps({"model": model, "messages": build_solver_messages(problem),
                       "max_tokens": tokens, "temperature": 1.0, "top_p": 0.95,
                       "n": m, "logprobs": True,
                       "stop": list(SOLVER_CHAT_BOUNDARY_STOPS)}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception:
        return [], []
    texts, us = [], []
    for ch in r["choices"]:
        texts.append(ch["message"]["content"] or "")
        lp = (ch.get("logprobs") or {}).get("content") or []
        vals = [t["logprob"] for t in lp if t.get("logprob") is not None]
        us.append(-statistics.fmean(vals) if vals else 0.0)
    return texts, us


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", required=True, choices=list(PO.OPS))
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--pop", type=int, default=16, help="founder population")
    ap.add_argument("--children", type=int, default=64, help="B: mutations per round")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--threads", type=int, default=96)
    ap.add_argument("--rollouts", type=int, default=10, help="m: = evolution.group_size")
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--verify-seeds", type=int, default=5)
    ap.add_argument("--seed-base", type=int, default=5000)
    ap.add_argument("--no-select", action="store_true",
                    help="admit on the gate stack alone; spend no rollouts. "
                         "The control that separates operator-driven collapse "
                         "from selection-driven collapse.")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = ROOT / (args.out or f"rq_output/chain_{args.op}"
                  f"{'_noselect' if args.no_select else ''}.json")

    rng = random.Random(args.seed)
    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    src = MAPElitesArchive(); src.load(snaps[-1])
    founders = [n.champion for n in src.grid.values() if n.champion is not None]
    # Same founders for every operator: the chains are blocked on their start.
    founders = sorted(founders, key=lambda p: p.program_id)
    rng.shuffle(founders)
    founders = founders[:args.pop]

    arch = MAPElitesArchive()
    for p in founders:
        arch.try_insert(p, float(p.u_score or 0.0), float(p.rq_score or 0.0))
    seen = {s for s in (skel(p.source_code) for p in arch.champions()) if s}
    print(f"op={args.op}  founders={len(arch.champions())}  distinct skeletons={len(seen)}  "
          f"select={'off' if args.no_select else 'R_Q'}", flush=True)

    seed_cursor = args.seed_base
    log = []
    hdr = (f"{'round':>6}{'pop':>5}{'distinct':>9}{'NEW':>5}{'cum':>5}"
           f"{'gatepass':>9}{'simPar':>8}{'simFnd':>8}{'cells':>6}{'meanRQ':>8}{'sec':>7}")
    print(hdr); print("-" * len(hdr))

    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        pool = arch.champions()
        if not pool:
            print(f"{rnd:>6}  population extinct"); break
        jobs = []
        for _ in range(args.children):
            par = pool[rng.randrange(len(pool))]
            don = pool[rng.randrange(len(pool))]
            jobs.append((par, don, rng.randint(0, 10 ** 9)))

        def mutate(job):
            par, don, sd = job
            r = random.Random(sd)
            gi = par.niche_group if isinstance(par.niche_group, int) else 0
            grp = GROUPS[gi] if gi < len(GROUPS) else GROUPS[0]
            msgs = PO.stage1(args.op, par, grp, don, r,
                             float(par.s_hat or 0.0), float(par.rq_score or 0.0),
                             len(pool))
            plan = PO.parse_plan(PO.chat(args.port, args.model, msgs, 1200))
            if not plan:
                return par, None, "stage1"
            code = PO.code_of(PO.chat(args.port, args.model,
                                      PO.stage2(par, plan), 2200))
            if not code:
                return par, None, "stage2"
            # evolution.py:878 -- stage 2 is told not to write the labels and the
            # run appends them from the stage-1 plan.
            return par, set_label_declarations(code, plan["GROUP"], plan["SKILL"]), None

        with ThreadPoolExecutor(args.threads) as ex:
            made = list(ex.map(mutate, jobs))

        deaths = collections.Counter()
        passed = []
        for par, code, why in made:
            if why:
                deaths[why] += 1; continue
            prog, why = gate(code, arch, args.verify_seeds)
            if why:
                deaths[why] += 1; continue
            passed.append((par, prog))

        # Score on the archive's own budget: ONE fresh instance x m rollouts.
        scored = []
        if passed and not args.no_select:
            insts = []
            for par, prog in passed:
                inst = None
                for z in range(seed_cursor, seed_cursor + 5):
                    inst = prog.execute(seed=z)
                    if inst is not None:
                        break
                insts.append(inst)
            seed_cursor += 5
            with ThreadPoolExecutor(args.threads) as ex:
                outs = list(ex.map(
                    lambda i: solve(args.port, args.model, i.problem, args.rollouts,
                                    args.tokens) if i is not None else ([], []),
                    insts))
            for (par, prog), inst, (texts, us) in zip(passed, insts, outs):
                if inst is None or not texts:
                    deaths["rollout_failed"] += 1; continue
                flags = []
                for t in texts:
                    pred = extract_boxed(sanitize_solver_trace(t))
                    flags.append(bool(pred is not None
                                      and answers_match(pred, inst.answer)))
                st = score_seed(seed=int(inst.seed), correct_flags=flags,
                                rollout_entropies=us)
                res = compute_rq_program([st])
                prog.s_hat = res.s_hat
                scored.append((par, prog, res.u_score, res.rq_score))
        else:
            scored = [(par, prog, 1.0, 0.0) for par, prog in passed]

        ins = 0
        for par, prog, u, rq in scored:
            prog.generation = int(getattr(par, "generation", 0) or 0) + 1
            prog.parent_id = par.program_id
            if arch.try_insert(prog, u, rq):
                ins += 1
            else:
                deaths["not_elite"] += 1

        champs = arch.champions()
        sks = {s for s in (skel(p.source_code) for p in champs) if s}
        new = len(sks - seen); seen |= sks
        sp = [sim(skel(pr.source_code), skel(pa.source_code))
              for pa, pr, _, _ in scored
              if skel(pr.source_code) and skel(pa.source_code)]
        fnd = [max(sim(skel(pr.source_code), skel(f.source_code)) for f in founders)
               for _, pr, _, _ in scored if skel(pr.source_code)]
        rqs = [float(p.rq_score or 0.0) for p in champs]
        row = {"round": rnd, "pop": len(champs), "distinct": len(sks), "new": new,
               "cum": len(seen), "gate_pass": len(passed), "inserted": ins,
               "deaths": dict(deaths),
               "sim_parent_med": statistics.median(sp) if sp else None,
               "sim_founder_med": statistics.median(fnd) if fnd else None,
               "cells": len({(p.niche_group, p.niche_skill) for p in champs}),
               "mean_rq": statistics.fmean(rqs) if rqs else 0.0}
        log.append(row)
        print(f"{rnd:>6}{len(champs):>5}{len(sks):>9}{new:>5}{len(seen):>5}"
              f"{len(passed):>9}"
              f"{(statistics.median(sp) if sp else float('nan')):>8.3f}"
              f"{(statistics.median(fnd) if fnd else float('nan')):>8.3f}"
              f"{row['cells']:>6}{row['mean_rq']:>8.4f}{time.time()-t0:>7.0f}",
              flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"args": vars(args), "rounds": log,
                               "final_sources": [p.source_code for p in arch.champions()]},
                              indent=2))
    # The headline: the round at which the operator stops producing forms.
    stall = next((r["round"] for r in log
                  if all(x["new"] == 0 for x in log[r["round"] - 1:r["round"] + 2])), None)
    print(f"\nnew forms per round: {[r['new'] for r in log]}")
    print(f"first round of a 3-round zero-new-form run: {stall}")
    print(f"written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
