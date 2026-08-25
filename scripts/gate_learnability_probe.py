#!/usr/bin/env python3
"""Do the newly-opened cells actually contain problems the policy cannot solve?

The coverage experiments stopped at "archive-admissible". This one scores the
survivors with real solver rollouts and reports the s_hat distribution, next to
the live champions as a control. Only 0 < s_hat < 1 feeds training, so this is
the link between "a new cell opened" and "the frontier refilled".
"""
import argparse, json, os, re, sys, threading, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq_evolve.prompts import build_solver_messages
from rq_evolve.program import ProblemProgram
from rq_evolve.code_utils import extract_generator_code, set_label_declarations
from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.math_eval import grade_eval
import requests

_tl = threading.local()


def chat(base_url, model, messages, n, temperature, top_p, max_tokens, retries=3):
    s = getattr(_tl, "s", None)
    if s is None:
        s = requests.Session(); _tl.s = s
    payload = {"model": model, "messages": messages, "n": n,
               "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    for k in range(retries):
        try:
            r = s.post(f"{base_url}/chat/completions", json=payload, timeout=1800)
            r.raise_for_status()
            return [c["message"]["content"] for c in r.json()["choices"]]
        except Exception:
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))
    return None


def s_hat(base_url, model, prog, n_seeds, n_rollouts, max_tokens):
    """Fraction of correct rollouts over n_seeds fresh instances x n_rollouts."""
    total = ok = 0
    for seed in range(n_seeds):
        inst = prog.execute(seed=seed)
        if inst is None:
            continue
        outs = chat(base_url, model, build_solver_messages(inst.problem),
                    n_rollouts, 1.0, 0.95, max_tokens)
        if not outs:
            continue
        for o in outs:
            total += 1
            ok += bool(grade_eval(o or "", inst.answer))
    return (ok / total if total else None), total


def band(v):
    if v is None: return "no rollouts"
    if v <= 0.0: return "s_hat = 0   (unsolvable)"
    if v >= 1.0: return "s_hat = 1   (already solved)"
    return "0 < s_hat < 1  (LEARNABLE)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="rq_output/gate_experiment")
    ap.add_argument("--arms", default="target_rotate_full,target_cell")
    ap.add_argument("--tag", default="")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive_iter243.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8312/v1")
    ap.add_argument("--model", default="qwen3-4b-step224")
    ap.add_argument("--n-seeds", type=int, default=2)
    ap.add_argument("--n-rollouts", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=3000)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--out", default="learnability.json")
    args = ap.parse_args()

    arch = json.load(open(args.archive))
    occupied = {(GROUPS[c["niche_group"]], SKILLS[c["niche_skill"]]) for c in arch["champions"]}

    jobs = []
    for c in arch["champions"]:
        jobs.append(("CHAMPION (control)",
                     f'{GROUPS[c["niche_group"]]}/{SKILLS[c["niche_skill"]]}',
                     ProblemProgram(source_code=c["source_code"], program_id=c["program_id"])))
    for arm in args.arms.split(","):
        p = os.path.join(args.dir, f"raw_{arm}{args.tag}.jsonl")
        if not os.path.exists(p):
            continue
        seen = 0
        for line in open(p, encoding="utf-8"):
            r = json.loads(line)
            if not (r["stage1_parsed"] and r["code_ok"] and r["inferred_skill"] == r["plan_skill"]):
                continue
            cell = (r["plan_group"], r["plan_skill"])
            if cell in occupied:
                continue
            code = extract_generator_code(r.get("stage2_reply") or "")
            if not code:
                continue
            src = set_label_declarations(code, r["plan_group"], r["plan_skill"])
            prog = ProblemProgram(source_code=src, program_id=f'{arm}{args.tag}-{r["i"]}')
            if prog.execute(seed=0) is None:
                continue
            jobs.append((f"NEW CELL {arm}{args.tag}", "/".join(cell), prog))
            seen += 1
            if seen >= args.limit:
                break
    print(f"[probe] {len(jobs)} programs "
          f"({sum(1 for j in jobs if j[0].startswith('CHAMPION'))} champions, "
          f"{sum(1 for j in jobs if j[0].startswith('NEW'))} new-cell children)")
    print(f"[probe] {args.n_seeds} seeds x {args.n_rollouts} rollouts against {args.model}")

    def run(j):
        grp, cell, prog = j
        v, n = s_hat(args.base_url, args.model, prog, args.n_seeds, args.n_rollouts, args.max_tokens)
        return dict(group=grp, cell=cell, program_id=prog.program_id, s_hat=v, rollouts=n)

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(run, jobs))
    print(f"[probe] done in {time.time()-t0:.0f}s\n")

    for grp in sorted({r["group"] for r in res}):
        v = [r for r in res if r["group"] == grp]
        c = Counter(band(r["s_hat"]) for r in v)
        learn = c["0 < s_hat < 1  (LEARNABLE)"]
        print(f'{grp:34s} n={len(v):3d}   LEARNABLE {learn:3d} ({100*learn/len(v):5.1f}%)')
        for k, n in c.most_common():
            print(f'      {k:32s} {n:3d}')
    json.dump(res, open(os.path.join(args.dir, args.out), "w"), indent=1)
    print(f'\nwrote {os.path.join(args.dir, args.out)}')


if __name__ == "__main__":
    main()
