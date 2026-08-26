#!/usr/bin/env python
"""One mutation operator per arm, stage 2 held fixed, so the operator is the only variable.

Why this exists. Measured on the 4B run's own archive: 145 champions carry 67
distinct AST skeletons, parent->child structural similarity has median 0.996,
45% of children reproduce the parent's skeleton EXACTLY, and new structural
forms per generation reach zero by generation 10. Text-level novelty rejection
cannot see that -- the statement changes while the skeleton does not -- so the
question is which mutation OPERATOR produces structural variety at all.

Every arm holds the parent's GROUP (the archive's group axis is inherited, not
declared) and is free on SKILL. Stage 2 is identical across arms: it is handed
the child family and the parent source and asked for the generator. Any
difference in the offspring is therefore attributable to the stage-1 operator.

    python scripts/probe_operators.py --n 30 --port 8701 --model step160
"""
from __future__ import annotations
import argparse, ast, difflib, json, random, re, sys, urllib.request, statistics, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.code_utils import (extract_problem_template, strip_label_declarations,      # noqa: E402
                                  strip_module_docstring, lint_generator_source)
from rq_evolve.program import ProblemProgram                                               # noqa: E402

TPL = ROOT / "prompt_templates"
def T(n): return (TPL / n).read_text(encoding="utf-8")
def D(n): return T(n).strip()

# --------------------------------------------------------------------------
# The ten operators. Each supplies the INSTRUCTION that replaces the shipped
# "make a structurally different child" order, and says what extra material the
# user turn carries. Everything else is shared.
# --------------------------------------------------------------------------
OPS = {
"local_diff": """
Change EXACTLY ONE clause of the parent family and nothing else. Name the clause
you are replacing and give its replacement. The rest of the statement -- objects,
bounds, the quantity asked for -- must survive word for word.
""",
"full_rewrite": """
Do not edit the parent. Write a NEW family from scratch that belongs to the same
mathematical domain and asks a question the parent does not. The parent is shown
only so you know which domain to stay in; do not reuse its objects, its bounds,
or the shape of its question.
""",
"directed": """
The parent is measured at a solver success rate of {s_hat:.2f}. The target band
is 0.30 to 0.70. Change the DECISIVE REASONING MOVE of the shortest clean
solution so the child lands in that band: {direction}. Keep the domain.
""",
"parameter": """
Keep the parent's statement structure exactly. Change ONLY the numeric ranges
and constants its parameters are drawn from, so that instances differ in size
but the mathematics and the solution route are unchanged.
""",
"simplify": """
Find every condition, object, or bound in the parent that the answer does NOT
depend on, and delete it. The child asks the same underlying question with the
redundant material removed. If nothing is redundant, say so and instead remove
the single least load-bearing condition and adjust the question to stay
well-posed.
""",
"structural": """
Keep the parent's mathematical domain and its kind of object. Change the
DECISIVE SOLUTION STRUCTURE: the move that the shortest clean solution turns on
must become a different move. Changing parameters, bounds, wording, the
direction of a predicate, or the size of the object does NOT count -- a solver
who knows the parent's method must be unable to finish your child with it.
""",
"scale": """
Keep the parent's structure and its decisive reasoning move exactly. Change only
the SIZE of the object the question is about -- more elements, a larger index, a
higher dimension -- so the same argument has to be carried further.
""",
"adaptive": """
Apply a {strength} mutation to the parent.
  small   : one clause changes; the solution route is the parent's.
  medium  : the question asked changes; the objects stay.
  radical : the objects and the decisive move both change; only the domain stays.
""",
"feedback": """
This parent was measured. Solver success rate {s_hat:.2f} over {n_roll} rollouts;
learnability R_Q {rq:.3f}; the archive already holds {n_cell} programs in its
cell. Diagnosis: {diagnosis}

Mutate the parent to fix what the diagnosis names. Do not change anything the
diagnosis does not implicate.
""",
"donor": """
A DONOR family from a different archive cell is shown after the parent. The
child must remain a mutation OF THE PARENT -- the parent's domain and objects
stay -- but it must import exactly ONE idea from the donor: a constraint, a
decisive move, or a way of counting. Name which idea you took. Copying the
donor, or ignoring it, both fail.
""",
}

TAIL = """
Hold GROUP fixed at the parent's own value: {group}. Choose SKILL freely from
the allowed list, as a DESCRIPTION of what your child actually demands -- never
as a target to write toward.
"""

def _prob(p):
    i = p.execute(seed=0)
    return i.problem.strip() if i else "(parent did not run)"

def stage1(op, parent, group, donor, rng, s_hat, rq, n_cell):
    sys_text = T("diff_problem_system_prompt.txt")
    diag = ("solvers never finish it -- make the decisive step reachable"
            if s_hat < 0.15 else
            "solvers finish it every time -- the decisive step is missing"
            if s_hat > 0.85 else
            "it sits in band; move it to an unoccupied part of the cell")
    instr = OPS[op].format(
        s_hat=s_hat, rq=rq, n_roll=10, n_cell=n_cell, diagnosis=diag,
        direction=("make the decisive step reachable" if s_hat < 0.5 else "add a step that a direct route cannot skip"),
        strength=rng.choice(["small", "medium", "radical"]),
    )
    body = (f"PARENT PROBLEM FAMILY\n\n{extract_problem_template(parent.source_code) or _prob(parent)}\n\n"
            f"One instance of it, for reference:\n\n{_prob(parent)}\n")
    if op == "donor":
        body += (f"\nDONOR FAMILY (different cell)\n\n"
                 f"{extract_problem_template(donor.source_code) or _prob(donor)}\n")
    user = (body + "\nAllowed SKILLS:\n" + D("skill_definitions.txt")
            + "\n\nOPERATOR\n" + instr + TAIL.format(group=group))
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]

def stage2(parent, plan):
    sys_text = T("gen_program_system_prompt.txt").replace("$skill_definitions", D("skill_definitions.txt"))
    user = (f"PARENT SOURCE\n\n```python\n"
            f"{strip_label_declarations(strip_module_docstring(parent.source_code))}\n```\n\n"
            f"NEW PROBLEM FAMILY TO IMPLEMENT\n\n{plan['CHILD FAMILY']}\n\n"
            f"GROUP: {plan['GROUP']}\nSKILL: {plan['SKILL']}\n\n"
            "Write the complete generator for the NEW PROBLEM FAMILY.")
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]

def chat(port, model, messages, max_tokens, temp=0.7):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": temp, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            return r["choices"][0]["message"]["content"] or ""
        except Exception:
            pass
    return ""

def parse_plan(reply):
    out = {}
    for k in ("CHILD FAMILY", "GROUP", "SKILL"):
        m = re.search(rf"^{re.escape(k)}\s*:\s*(.+?)\s*$", reply, re.M)
        if m: out[k] = m.group(1).strip()
    return out if len(out) == 3 else None

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--out", default="rq_output/probe_operators.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    champs = json.loads(snaps[-1].read_text())["champions"]
    champs = list(champs.values() if isinstance(champs, dict) else champs)
    GNAMES = [l.split(":")[0].strip() for l in D("group_definitions.txt").split("\n") if l.strip()]
    pool = []
    for c in champs:
        gi = c.get("niche_group")
        pool.append(dict(prog=ProblemProgram(source_code=c["source_code"]),
                         group=GNAMES[gi] if isinstance(gi, int) and gi < len(GNAMES) else GNAMES[0],
                         cell=(c.get("niche_group"), c.get("niche_skill")),
                         s_hat=float(c.get("s_hat") or 0.0), rq=float(c.get("rq_score") or 0.0),
                         src=c["source_code"]))
    arch_sk = [s for s in (skel(p["src"]) for p in pool) if s]
    cellcount = collections.Counter(p["cell"] for p in pool)
    rng = random.Random(args.seed)

    jobs = []
    for op in OPS:
        for i in range(args.n):
            a = rng.randrange(len(pool))
            d = rng.randrange(len(pool))
            while pool[d]["cell"] == pool[a]["cell"] and len(pool) > 1: d = rng.randrange(len(pool))
            jobs.append((op, i, a, d, rng.randint(0, 10**9)))

    def run(job):
        op, i, ai, di, sd = job
        P, DN = pool[ai], pool[di]
        r = random.Random(sd)
        row = {"op": op, "i": i, "parent_src": P["src"], "parent_cell": P["cell"], "parent_s_hat": P["s_hat"]}
        r1 = chat(args.port, args.model,
                  stage1(op, P["prog"], P["group"], DN["prog"], r, P["s_hat"], P["rq"], cellcount[P["cell"]]), 1200)
        plan = parse_plan(r1)
        row["stage1_ok"] = plan is not None
        if not plan: return row
        row.update(child_family=plan["CHILD FAMILY"], plan_group=plan["GROUP"], plan_skill=plan["SKILL"])
        r2 = chat(args.port, args.model, stage2(P["prog"], plan), 2200)
        c = code_of(r2)
        row["stage2_ok"] = c is not None
        if c: row["code"] = c
        return row

    print(f"{len(OPS)} operators x n={args.n}  model={args.model}:{args.port}", flush=True)
    with ThreadPoolExecutor(24) as ex: rows = list(ex.map(run, jobs))
    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"written to {ROOT/args.out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
