#!/usr/bin/env python3
"""Confusion matrices + coverage metrics for the three prompt arms."""
import json, os, re, sys, collections, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq_evolve.concepts import GROUPS, SKILLS

SH = {s: s[:6] for s in SKILLS}
FEWSHOT = {
    "EX1 |x^2-b|=c": r"x\^?2\s*-\s*b|\|\s*x.{0,4}2\s*-\s*b",
    "EX2 n lines->regions": r"no three pass through one point|regions do (these|the).{0,12}lines|lines.{0,30}divide the plane",
    "EX3 proper divisor": r"proper divisor",
    "EX4 a+b+ab board": r"a \+ b \+ a\s*\*?\s*b|blackboard",
    "EX5 largest >= L": r"largest of the numbers is at least",
    "EX6 derangement": r"own hat|derangement",
    "EX7 xy+x+y=N": r"xy \+ x \+ y|x\s*\*\s*y \+ x \+ y",
    "EX8 triangle-free": r"contains no triangle|triangle-free",
}


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rq_output/gate_experiment")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive_iter243.json")
    ap.add_argument("--arms", default="baseline,target_cell,target_rotate")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    arch = json.load(open(args.archive))
    occupied = {(GROUPS[c["niche_group"]], SKILLS[c["niche_skill"]]) for c in arch["champions"]}

    summary = {}
    for arm in args.arms.split(","):
        p = os.path.join(args.dir, f"raw_{arm}{args.tag}.jsonl")
        if not os.path.exists(p):
            print(f"!! missing {p}"); continue
        rows = load(p)
        n = len(rows)
        parsed = [r for r in rows if r["stage1_parsed"]]
        coded = [r for r in parsed if r["code_ok"]]

        print("\n" + "=" * 104)
        print(f"ARM: {arm}    n={n}")
        print(f"  stage-1 parsed      : {len(parsed):4d}/{n}  ({100*len(parsed)/n:.1f}%)")
        print(f"  stage-2 code ok     : {len(coded):4d}/{n}  ({100*len(coded)/n:.1f}%)")

        # ---- gate outcome ----
        match = mism = pfail = 0
        cm = collections.Counter()
        for r in coded:
            inf = r["inferred_skill"]
            if inf is None:
                pfail += 1
                cm[(r["plan_skill"], "PARSE_FAIL")] += 1
            elif inf == r["plan_skill"]:
                match += 1; cm[(r["plan_skill"], inf)] += 1
            else:
                mism += 1; cm[(r["plan_skill"], inf)] += 1
        gate_pool = match + mism
        print(f"  gate match          : {match:4d}   mismatch {mism:4d}   parse_fail {pfail:3d}"
              f"   -> match rate {100*match/max(gate_pool,1):.1f}%")
        print(f"  end-to-end survival : {100*match/n:.1f}%  (of all {n} attempts)")

        # ---- confusion matrix ----
        print(f"\n  CONFUSION MATRIX  rows = stage-1 plan SKILL, cols = stage-2 INFERRED_SKILL")
        hdr = "  " + "plan / inferred".ljust(22) + "".join(f"{SH[s]:>8s}" for s in SKILLS) + f"{'PARSE':>8s}{'n':>7s}{'pass%':>8s}"
        print(hdr); print("  " + "-" * (len(hdr) - 2))
        for ps in SKILLS:
            tot = sum(v for (a, b), v in cm.items() if a == ps)
            if tot == 0:
                print(f"  {ps:22s}" + "".join(f"{'.':>8s}" for _ in SKILLS) + f"{'.':>8s}{0:7d}{'-':>8s}")
                continue
            cells = "".join(f"{cm[(ps,s)] or '.':>8}" for s in SKILLS)
            pf = cm[(ps, "PARSE_FAIL")]
            denom = tot - pf
            rate = f"{100*cm[(ps,ps)]/denom:.0f}%" if denom else "-"
            print(f"  {ps:22s}{cells}{pf or '.':>8}{tot:7d}{rate:>8s}")

        colm = {s: sum(v for (a, b), v in cm.items() if b == s) for s in SKILLS}
        tot_inf = sum(colm.values())
        print("  " + "INFERRED total (col)".ljust(22)
              + "".join(f"{colm[s] or '.':>8}" for s in SKILLS)
              + f"{'':>8}{tot_inf:7d}")
        print("  " + "plan total (row)".ljust(22)
              + "".join(f"{sum(v for (a,b),v in cm.items() if a==s) or '.':>8}" for s in SKILLS))

        # ---- plan-side distributions ----
        pg = collections.Counter(r["plan_group"] for r in parsed)
        pk = collections.Counter(r["plan_skill"] for r in parsed)
        print(f"\n  stage-1 plan GROUP : " + "  ".join(f"{g[:6]}={100*pg[g]/len(parsed):.0f}%" for g in GROUPS))
        print(f"  stage-1 plan SKILL : " + "  ".join(f"{s[:6]}={100*pk[s]/len(parsed):.0f}%" for s in SKILLS))

        # ---- target compliance ----
        if any(r["target_cell"] for r in parsed):
            tg = [r for r in parsed if r["target_cell"]]
            gc = sum(1 for r in tg if r["plan_group"] == r["target_cell"][0])
            sc = sum(1 for r in tg if r["plan_skill"] == r["target_cell"][1])
            bc = sum(1 for r in tg if [r["plan_group"], r["plan_skill"]] == r["target_cell"])
            print(f"  target compliance  : GROUP {100*gc/len(tg):.0f}%   SKILL {100*sc/len(tg):.0f}%   BOTH {100*bc/len(tg):.0f}%")

        # ---- coverage yield ----
        def cell(r): return (r["plan_group"], r["plan_skill"])
        empty_plan = [r for r in parsed if cell(r) not in occupied]
        empty_surv = [r for r in coded if cell(r) not in occupied and r["inferred_skill"] == r["plan_skill"]]
        distinct_empty = {cell(r) for r in empty_surv}
        print(f"\n  >> plans aimed at an EMPTY cell        : {len(empty_plan):4d}/{len(parsed)}  ({100*len(empty_plan)/len(parsed):.1f}%)")
        print(f"  >> ... that also SURVIVE the gate      : {len(empty_surv):4d}/{n}       ({100*len(empty_surv)/n:.1f}% of attempts)")
        print(f"  >> distinct empty cells reached        : {len(distinct_empty):4d} / 26")

        # ---- few-shot leakage ----
        hit = collections.Counter(); anyh = 0
        for r in parsed:
            t = r["child_family"] or ""
            h = False
            for k, pat in FEWSHOT.items():
                if re.search(pat, t, re.I): hit[k] += 1; h = True
            anyh += h
        print(f"  >> few-shot example copied in CHILD    : {anyh:4d}/{len(parsed)}  ({100*anyh/len(parsed):.1f}%)"
              + ("   top: " + ", ".join(f"{k.split()[0]}={v}" for k, v in hit.most_common(3)) if hit else ""))
        norm = collections.Counter(re.sub(r"\s+", " ", (r["child_family"] or "").lower())[:120] for r in parsed)
        print(f"  >> distinct child families             : {len(norm):4d}/{len(parsed)}"
              f"   top1 share {100*norm.most_common(1)[0][1]/len(parsed):.1f}%")

        summary[arm] = {
            "n": n, "stage1_parsed": len(parsed), "code_ok": len(coded),
            "gate_match": match, "gate_mismatch": mism, "gate_parse_fail": pfail,
            "gate_match_rate": match / max(gate_pool, 1),
            "e2e_survival": match / n,
            "empty_plans": len(empty_plan), "empty_survivors": len(empty_surv),
            "distinct_empty_cells": len(distinct_empty),
            "fewshot_copy_rate": anyh / len(parsed),
            "distinct_children": len(norm),
            "plan_skill": dict(pk), "plan_group": dict(pg),
            "confusion": {f"{a}->{b}": v for (a, b), v in cm.items()},
        }

    print("\n" + "=" * 104)
    print(f"{'ARM':16s}{'gate pass':>11s}{'e2e surv':>11s}{'empty-plan':>12s}{'empty-surv':>12s}{'cells/26':>10s}{'fewshot':>10s}")
    for arm, s in summary.items():
        print(f"{arm:16s}{100*s['gate_match_rate']:10.1f}%{100*s['e2e_survival']:10.1f}%"
              f"{100*s['empty_plans']/s['stage1_parsed']:11.1f}%{100*s['empty_survivors']/s['n']:11.1f}%"
              f"{s['distinct_empty_cells']:10d}{100*s['fewshot_copy_rate']:9.1f}%")
    json.dump(summary, open(os.path.join(args.dir, f"summary{args.tag}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
