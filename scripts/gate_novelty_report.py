#!/usr/bin/env python3
"""Second pass: do the generated children actually run, and are they NEW?

Runs every arm's stage-2 code in the repo's own sandbox, then puts each child
through the exact archive gates (seed-variation, behavior duplicate, template
duplicate, near-duplicate) against the live 22-champion MAP. This is the test
the confusion matrix cannot do: a child can pass the SKILL gate and still be a
byte-identical clone of a champion in another cell.
"""
import argparse, json, os, sys, collections
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq_evolve.archive import MAPElitesArchive
from rq_evolve.program import ProblemProgram
from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.code_utils import set_label_declarations, extract_generator_code


def build_archive(path):
    a = MAPElitesArchive()
    for c in json.load(open(path))["champions"]:
        p = ProblemProgram(source_code=c["source_code"], program_id=c["program_id"])
        p.niche_group, p.niche_skill = c["niche_group"], c["niche_skill"]
        p.rq_score = c.get("rq_score") or 0.0
        p.s_hat = c.get("s_hat") or 0.0
        p.u_score = c.get("u_score") or 0.0
        a.grid[(c["niche_group"], c["niche_skill"])].champion = p
        a.grid[(c["niche_group"], c["niche_skill"])].champion_rq = p.rq_score
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rq_output/gate_experiment")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive_iter243.json")
    ap.add_argument("--arms", default="baseline,target_cell,target_rotate")
    ap.add_argument("--tag", default="")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    arch = build_archive(args.archive)
    occupied = {(GROUPS[g], SKILLS[s]) for (g, s), n in arch.grid.items() if n.champion}
    out_summary = {}

    print(f"{'ARM':16s}{'gate-passed':>12s}{'runs':>8s}{'seed-var':>10s}{'dup-beh':>9s}"
          f"{'dup-tmpl':>10s}{'near-dup':>10s}{'CLEAN':>8s}{'->EMPTY':>9s}{'cells':>7s}")
    for arm in args.arms.split(","):
        p = os.path.join(args.dir, f"raw_{arm}{args.tag}.jsonl")
        if not os.path.exists(p):
            continue
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        # children that would actually be handed to verify in the real pipeline:
        # stage-1 parsed, code extracted, SKILL gate matched
        live = [r for r in rows
                if r["stage1_parsed"] and r["code_ok"]
                and r["inferred_skill"] == r["plan_skill"]]

        def probe(r):
            code = extract_generator_code(r.get("stage2_reply") or "")
            src = set_label_declarations(code, r["plan_group"], r["plan_skill"]) if code else None
            res = {"i": r["i"], "cell": (r["plan_group"], r["plan_skill"])}
            if not src:
                res["runs"] = False; res["err"] = "NoCode: reply truncated in log"
                return res
            prog = ProblemProgram(source_code=src, program_id=f"{arm}-{r['i']}")
            inst = prog.execute(seed=0)
            res["runs"] = inst is not None
            if not res["runs"]:
                res["err"] = (prog.last_execution_error or "")[:120]
                return res
            res["seed_var"] = bool(arch._passes_seed_variation(prog))
            if not res["seed_var"]:
                return res
            d = arch._find_duplicate_behavior(prog)
            res["dup_beh"] = None if d is None else (GROUPS[d.niche_group], SKILLS[d.niche_skill])
            if d is not None:
                return res
            d = arch._find_duplicate_template(prog)
            res["dup_tmpl"] = None if d is None else (GROUPS[d.niche_group], SKILLS[d.niche_skill])
            if d is not None:
                return res
            nd = arch._find_near_duplicate_template(prog)
            res["near"] = None if nd is None else (GROUPS[nd[0].niche_group], SKILLS[nd[0].niche_skill], round(nd[1], 3))
            res["clean"] = nd is None
            return res

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            probes = list(ex.map(probe, live))

        runs = [x for x in probes if x.get("runs")]
        sv_fail = [x for x in runs if not x.get("seed_var")]
        db = [x for x in runs if x.get("dup_beh")]
        dt = [x for x in runs if x.get("dup_tmpl")]
        nd = [x for x in runs if x.get("near")]
        clean = [x for x in runs if x.get("clean")]
        clean_empty = [x for x in clean if x["cell"] not in occupied]
        cells = {x["cell"] for x in clean_empty}
        print(f"{arm:16s}{len(live):12d}{len(runs):8d}{len(sv_fail):10d}{len(db):9d}"
              f"{len(dt):10d}{len(nd):10d}{len(clean):8d}{len(clean_empty):9d}{len(cells):7d}")
        out_summary[arm] = {
            "n_rows": len(rows), "gate_passed": len(live), "runs": len(runs),
            "seed_variation_fail": len(sv_fail), "dup_behavior": len(db),
            "dup_template": len(dt), "near_dup": len(nd), "clean": len(clean),
            "clean_into_empty_cell": len(clean_empty),
            "distinct_new_cells": sorted("/".join(c) for c in cells),
            "dup_partners": collections.Counter(
                "/".join(x["dup_beh"] or x["dup_tmpl"] or x["near"][:2])
                for x in db + dt + nd).most_common(),
            "run_errors": collections.Counter(
                (x.get("err") or "").split(":")[0] for x in probes if not x.get("runs")).most_common(6),
        }
    json.dump(out_summary, open(os.path.join(args.dir, f"novelty_summary{args.tag}.json"), "w"), indent=1)
    print()
    for arm, s in out_summary.items():
        print(f"\n[{arm}]  new distinct cells reached: {s['distinct_new_cells']}")
        print(f"          duplicate partners: {s['dup_partners'][:6]}")
        print(f"          execution errors  : {s['run_errors']}")


if __name__ == "__main__":
    main()
