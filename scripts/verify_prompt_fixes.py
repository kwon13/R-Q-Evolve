#!/usr/bin/env python3
"""End-to-end check that the SHIPPED prompt path reproduces the experiment.

Calls prompts.build_family_task / build_generator_task and archive.sample_target_cell
directly -- not the experiment harness -- so a regression in the production code
shows up here.
"""
import argparse, json, os, random, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq_evolve.archive import MAPElitesArchive
from rq_evolve.prompts import (build_family_task, build_generator_task,
                               parse_family_plan, parse_inferred_labels)
from rq_evolve.code_utils import extract_generator_code
from rq_evolve.program import ProblemProgram
from rq_evolve.concepts import GROUPS, SKILLS
import requests

_tl = threading.local()


def chat(url, model, messages, temperature, top_p, max_tokens):
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session(); _tl.s = s
    for k in range(3):
        try:
            r = s.post(f"{url}/chat/completions", timeout=1200, json={
                "model": model, "messages": messages, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception:
            if k == 2: return None
            time.sleep(2 * (k + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive_iter243.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8311/v1")
    ap.add_argument("--model", default="qwen3-4b-base")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rq_output/gate_experiment/verify_shipped.jsonl")
    args = ap.parse_args()

    snap = json.load(open(args.archive))
    arch = MAPElitesArchive()
    parents = []
    for c in snap["champions"]:
        p = ProblemProgram(source_code=c["source_code"], program_id=c["program_id"])
        p.niche_group, p.niche_skill = c["niche_group"], c["niche_skill"]
        parents.append(p)
        arch.grid[(c["niche_group"], c["niche_skill"])].champion = p
    occupied = {(GROUPS[g], SKILLS[s]) for (g, s), n in arch.grid.items() if n.champion}
    print(f"[verify] {len(parents)} parents, {len(occupied)} occupied, {48-len(occupied)} empty")

    rng = random.Random(args.seed)
    random.seed(args.seed)
    jobs = [{"i": i, "parent": rng.choice(parents), "target": arch.sample_target_cell()}
            for i in range(args.n)]

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        def s1(j):
            t = build_family_task(j["parent"], temperature=0.4, top_p=0.95,
                                  target_cell=j["target"], rotate_shots=True)
            j["r1"] = chat(args.base_url, args.model, t.messages, 0.4, 0.95, 5000)
            j["plan"] = parse_family_plan(j["r1"] or "")
            return j
        jobs = list(ex.map(s1, jobs))
    live = [j for j in jobs if j["plan"]]
    print(f"[verify] stage-1 parsed {len(live)}/{len(jobs)}")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        def s2(j):
            t = build_generator_task(j["parent"], j["plan"], temperature=0.0,
                                     top_p=1.0, rotate_shots=True)
            j["r2"] = chat(args.base_url, args.model, t.messages, 0.0, 1.0, 5000)
            j["code"] = extract_generator_code(j["r2"] or "")
            _g, j["inf"] = parse_inferred_labels(j["r2"] or "")
            return j
        live = list(ex.map(s2, live))

    match = sum(1 for j in live if j["code"] and j["inf"] == j["plan"]["SKILL"])
    decided = sum(1 for j in live if j["code"] and j["inf"] is not None)
    tgt_skill = sum(1 for j in live if j["plan"]["SKILL"] == j["target"][1])
    tgt_group = sum(1 for j in live if j["plan"]["GROUP"] == j["target"][0])
    cells = [(j["plan"]["GROUP"], j["plan"]["SKILL"]) for j in live]
    empty_plan = sum(1 for c in cells if c not in occupied)
    empty_surv = sum(1 for j in live
                     if (j["plan"]["GROUP"], j["plan"]["SKILL"]) not in occupied
                     and j["code"] and j["inf"] == j["plan"]["SKILL"])
    distinct = {c for j, c in zip(live, cells)
                if c not in occupied and j["code"] and j["inf"] == j["plan"]["SKILL"]}
    sk = Counter(j["plan"]["SKILL"] for j in live)
    gr = Counter(j["plan"]["GROUP"] for j in live)
    import re
    FEW = r'contains no triangle|xy \+ x \+ y|no three pass through one point|largest of the numbers is at least|own hat|proper divisor|blackboard|x\^?2\s*-\s*b'
    few = sum(1 for j in live if re.search(FEW, j["plan"]["CHILD FAMILY"], re.I))

    print(f"\n[SHIPPED CODE] n={args.n}")
    print(f"  stage-1 parsed          {len(live)}/{args.n} ({100*len(live)/args.n:.1f}%)")
    print(f"  gate match rate         {100*match/max(decided,1):.1f}%")
    print(f"  target SKILL compliance {100*tgt_skill/len(live):.0f}%   GROUP {100*tgt_group/len(live):.0f}%")
    print(f"  plan GROUP  " + "  ".join(f"{g[:6]}={100*gr[g]/len(live):.0f}%" for g in GROUPS))
    print(f"  plan SKILL  " + "  ".join(f"{s[:6]}={100*sk[s]/len(live):.0f}%" for s in SKILLS))
    print(f"  >> empty-cell plans     {empty_plan}/{len(live)} ({100*empty_plan/len(live):.1f}%)")
    print(f"  >> empty-cell survivors {empty_surv}/{args.n} ({100*empty_surv/args.n:.1f}%)")
    print(f"  >> distinct empty cells {len(distinct)} / {48-len(occupied)}")
    print(f"  >> few-shot copied      {few}/{len(live)} ({100*few/len(live):.1f}%)")
    with open(args.out, "w", encoding="utf-8") as f:
        for j in live:
            f.write(json.dumps({"i": j["i"], "target": list(j["target"]),
                                "plan_group": j["plan"]["GROUP"], "plan_skill": j["plan"]["SKILL"],
                                "child_family": j["plan"]["CHILD FAMILY"][:600],
                                "code_ok": bool(j["code"]), "inferred_skill": j["inf"],
                                "stage2_reply": (j["r2"] or "")[:12000]}, ensure_ascii=False) + "\n")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
