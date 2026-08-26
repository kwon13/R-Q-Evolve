#!/usr/bin/env python
"""Separate the stage-1 contribution from the stage-2 contribution.

probe_operators.py holds stage 2 fixed and calls every difference between arms
an operator effect. That only follows if stage 2 contributes a CONSTANT amount.
It does not: stage 2 is handed the parent's source, and what it is told the
parent IS decides how much of it comes back.

This runs ONE stage-1 plan per parent per operator and then implements that SAME
plan four ways, so stage 1 is literally identical across the four columns and
every difference is stage 2:

    prod   parent family + parent generator, production framing (gen_program_user_prompt.txt)
    show   probe_operators.py's own "PARENT SOURCE ... NEW PROBLEM FAMILY" framing
    fixed  a FIXED unrelated seed generator as the reference pair
    none   no reference generator at all

    python scripts/probe_stage2_ablation.py --n 20 --port 8701 --model step160
"""
from __future__ import annotations
import argparse, ast, difflib, json, random, re, sys, urllib.request, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.code_utils import (extract_problem_template, strip_label_declarations,   # noqa: E402
                                  strip_module_docstring)
from rq_evolve.concepts import GROUPS, SKILLS                                           # noqa: E402
from rq_evolve.program import ProblemProgram                                            # noqa: E402

TPL = ROOT / "prompt_templates"
T = lambda n: (TPL / n).read_text(encoding="utf-8")
D = lambda n: T(n).strip()

OPS = {
"full_rewrite": """
Do not edit the parent. Write a NEW family from scratch that belongs to the same
mathematical domain and asks a question the parent does not. The parent is shown
only so you know which domain to stay in; do not reuse its objects, its bounds,
or the shape of its question.
""",
"parameter": """
Keep the parent's statement structure exactly. Change ONLY the numeric ranges
and constants its parameters are drawn from, so that instances differ in size
but the mathematics and the solution route are unchanged.
""",
}
TAIL = """
Hold GROUP fixed at the parent's own value: {group}. Choose SKILL freely from
the allowed list, as a DESCRIPTION of what your child actually demands -- never
as a target to write toward.
"""
_KEYS = "STRUCTURAL MUTATION|CHILD FAMILY|WHY FINITE|GROUP|SKILL"

def _prob(p):
    i = p.execute(seed=0)
    return i.problem.strip() if i else "(parent did not run)"

def _field(text, key):
    best = ""
    for m in re.finditer(rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_KEYS})[ \t]*:|\Z)",
                         text, re.M | re.S):
        v = m.group(1).strip()
        if v and not v.startswith("<"): best = v
    return best

def parse_plan(reply):
    out = {"CHILD FAMILY": _field(reply, "CHILD FAMILY")}
    for k, vocab in (("GROUP", GROUPS), ("SKILL", SKILLS)):
        v = _field(reply, k).strip().strip("\"'").split()
        out[k] = (v[0].strip(".,") if v else "")
        if out[k] not in vocab: out[k] = ""
    return out if all(out.values()) else None

def stage1(op, parent, group):
    user = (f"PARENT PROBLEM FAMILY\n\n{extract_problem_template(parent.source_code) or _prob(parent)}\n\n"
            f"One instance of it, for reference:\n\n{_prob(parent)}\n"
            "\nAllowed SKILLS:\n" + D("skill_definitions.txt")
            + "\n\nOPERATOR\n" + OPS[op] + TAIL.format(group=group))
    return [{"role": "system", "content": T("diff_problem_system_prompt.txt")},
            {"role": "user", "content": user}]

_FIXED = None
def stage2(parent, plan, mode):
    global _FIXED
    sys_text = T("gen_program_system_prompt.txt").replace("$skill_definitions", D("skill_definitions.txt"))
    ref = parent
    if mode == "fixed":
        if _FIXED is None:
            _FIXED = ProblemProgram.from_file(sorted((ROOT / "seed_programs").glob("*.py"))[0])
        ref = _FIXED
    if mode == "show":
        user = (f"PARENT SOURCE\n\n```python\n"
                f"{strip_label_declarations(strip_module_docstring(parent.source_code))}\n```\n\n"
                f"NEW PROBLEM FAMILY TO IMPLEMENT\n\n{plan['CHILD FAMILY']}\n\n"
                f"GROUP: {plan['GROUP']}\nSKILL: {plan['SKILL']}\n\n"
                "Write the complete generator for the NEW PROBLEM FAMILY.")
    elif mode == "none":
        user = f"NOW WRITE THE GENERATOR FOR THIS FAMILY\n\n{plan['CHILD FAMILY']}\n\nCORRECT OUTPUT:\n"
    else:
        user = Template(T("gen_program_user_prompt.txt")).safe_substitute(
            parent_template=extract_problem_template(ref.source_code) or _prob(ref),
            parent_source=strip_label_declarations(strip_module_docstring(ref.source_code)),
            new_problem=plan["CHILD FAMILY"])
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]

def chat(port, model, messages, max_tokens, temp, top_p, seed):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temp, "top_p": top_p, "seed": seed}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    last = ""
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            c = r["choices"][0]
            return (c["message"]["content"] or ""), (c.get("finish_reason") or "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
    return "", last

def code_of(reply):
    m = re.findall(r"```python\n(.*?)```", reply, re.S)
    return max(m, key=len) if m else None

class _Sk(ast.NodeVisitor):
    def __init__(self): self.seq = []
    def generic_visit(self, n): self.seq.append(type(n).__name__); super().generic_visit(n)
def skel(src):
    try: t = ast.parse(src)
    except Exception: return None
    v = _Sk(); v.visit(t); return tuple(v.seq)
def sim(a, b): return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

MODES = ("prod", "show", "fixed", "none")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--out", default="rq_output/probe_stage2_ablation.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--temp1", type=float, default=0.4)
    ap.add_argument("--temp2", type=float, default=0.0)
    args = ap.parse_args()

    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    champs = json.loads(snaps[-1].read_text())["champions"]
    champs = list(champs.values() if isinstance(champs, dict) else champs)
    pool = [dict(prog=ProblemProgram(source_code=c["source_code"]),
                 group=GROUPS[c["niche_group"]], src=c["source_code"])
            for c in champs if isinstance(c.get("niche_group"), int)]
    rng = random.Random(args.seed)
    block = [(i, rng.randrange(len(pool)), rng.randint(0, 10**9)) for i in range(args.n)]

    plans = {}
    def do1(job):
        op, (i, ai, sd) = job
        P = pool[ai]
        r1, _f = chat(args.port, args.model, stage1(op, P["prog"], P["group"]), 2048, args.temp1, 0.95, sd)
        return (op, i, ai, sd, parse_plan(r1))
    jobs1 = [(op, b) for op in OPS for b in block]
    with ThreadPoolExecutor(8) as ex: out1 = list(ex.map(do1, jobs1))
    for op, i, ai, sd, plan in out1:
        if plan: plans[(op, i)] = (ai, sd, plan)
    print(f"stage 1: {len(plans)}/{len(jobs1)} plans parsed", flush=True)

    def do2(job):
        (op, i), mode = job
        ai, sd, plan = plans[(op, i)]
        P = pool[ai]
        r2, fin = chat(args.port, args.model, stage2(P["prog"], plan, mode), 3000, args.temp2, 1.0, sd)
        c = code_of(r2)
        return {"op": op, "i": i, "mode": mode, "parent_src": P["src"],
                "child_family": plan["CHILD FAMILY"], "code": c, "finish": fin, "rawlen": len(r2)}
    jobs2 = [(k, m) for k in plans for m in MODES]
    with ThreadPoolExecutor(8) as ex: rows = list(ex.map(do2, jobs2))
    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    print(f"\n{'operator':<14}{'mode':<8}{'n':>4}{'skelSim':>9}{'ident%':>8}{'famSim':>9}")
    for op in OPS:
        for mode in MODES:
            g = [r for r in rows if r["op"] == op and r["mode"] == mode and r["code"]]
            ss, fs = [], []
            for r in g:
                c = skel(r["code"]); p = skel(strip_label_declarations(strip_module_docstring(r["parent_src"])))
                if c and p: ss.append(sim(c, p))
                pt = extract_problem_template(r["parent_src"])
                if pt: fs.append(sim(r["child_family"], pt))
            m = lambda v: statistics.median(v) if v else float("nan")
            ident = 100 * sum(1 for x in ss if x >= 0.999) / max(len(ss), 1)
            print(f"{op:<14}{mode:<8}{len(g):>4}{m(ss):>9.3f}{ident:>7.0f}%{m(fs):>9.3f}")
    import collections as _c
    print("\nstage-2 outcomes:", dict(_c.Counter((r["mode"], r["finish"] or "err", bool(r["code"])) for r in rows)))
    print(f"\nwritten to {ROOT/args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
