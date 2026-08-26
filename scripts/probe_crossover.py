#!/usr/bin/env python
"""Does giving the mutation TWO parents break the structural copying?

Measured on the 4B run's own archive: 145 champions carry only 67 distinct AST
skeletons, parent->child structural similarity has median 0.996, and 45% of
children reproduce their parent's skeleton EXACTLY. New structural forms per
generation fall to zero by generation 10. Text-level novelty rejection cannot
see this -- the statement changes while the skeleton does not.

The hypothesis this probe tests is that the copying is taught by the prompt:
stage 2 is handed the parent's SOURCE as the worked example of the
statement-to-program mapping, so the cheapest correct reply is that source with
the constants moved. Showing TWO parents from different cells removes the
single template to copy.

Arms:
  single        one parent + target cell            (the shipped behaviour)
  cross         two parents from different cells, both shown, target cell kept
  cross_free    two parents, no target cell         (crossover alone as the driver)

    python scripts/probe_crossover.py --n 40 --port 8701 --model step160
"""
from __future__ import annotations
import argparse, ast, difflib, json, random, re, sys, urllib.request, collections, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.code_utils import (extract_problem_template, strip_label_declarations,  # noqa: E402
                                  strip_module_docstring, lint_generator_source)
from rq_evolve.program import ProblemProgram  # noqa: E402

TPL = ROOT / "prompt_templates"
def T(name): return (TPL / name).read_text(encoding="utf-8")
def defs(name): return T(name).strip()

TARGET_BLOCK = """
TARGET NICHE (the archive cell this child must land in):
TARGET GROUP: {group}
TARGET SKILL: {skill}

The CHILD FAMILY you invent must belong to TARGET GROUP, and the decisive
reasoning move of its shortest clean solution must be TARGET SKILL. Write the
GROUP and SKILL lines as exactly these two values.
"""

CROSS_HEAD = """
You are given TWO parent families from different regions of the archive, not one.

The CHILD FAMILY must inherit something essential from EACH parent -- an object,
a constraint, or a decisive move -- and combine them into one question that
neither parent asks. A child that is recognisably one parent with the other
ignored is a failed crossover, and so is a child that merely places the two
questions side by side. The combination itself is what has to be new.
"""

def _parent_problem(p):
    inst = p.execute(seed=0)
    return inst.problem.strip() if inst else "(parent did not run)"

def stage1(parents, target, cross):
    sys_text = T("diff_problem_system_prompt.txt")
    if cross: sys_text = sys_text + "\n" + CROSS_HEAD
    body = []
    for i, p in enumerate(parents, 1):
        tag = f"PARENT {i} " if cross else "PARENT "
        body.append(f"{tag}PROBLEM FAMILY\n\n"
                    f"{extract_problem_template(p.source_code) or _parent_problem(p)}\n\n"
                    f"One instance of it, for reference:\n\n{_parent_problem(p)}\n")
    user = ("\n".join(body) + "\nAllowed GROUPS:\n" + defs("group_definitions.txt")
            + "\n\nAllowed SKILLS:\n" + defs("skill_definitions.txt"))
    if target: user += "\n" + TARGET_BLOCK.format(group=target[0], skill=target[1])
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]

def stage2(parents, plan, cross):
    sys_text = T("gen_program_system_prompt.txt").replace("$skill_definitions", defs("skill_definitions.txt"))
    src_blocks = "\n\n".join(
        f"PARENT {i} SOURCE\n\n```python\n{strip_label_declarations(strip_module_docstring(p.source_code))}\n```"
        for i, p in enumerate(parents, 1)) if cross else \
        f"PARENT SOURCE\n\n```python\n{strip_label_declarations(strip_module_docstring(parents[0].source_code))}\n```"
    user = (f"{src_blocks}\n\nNEW PROBLEM FAMILY TO IMPLEMENT\n\n{plan['CHILD FAMILY']}\n\n"
            f"GROUP: {plan['GROUP']}\nSKILL: {plan['SKILL']}\n\n"
            "Write the complete generator for the NEW PROBLEM FAMILY. The parent "
            "source above is shown only as an example of the statement-to-program "
            "mapping; do not carry its structure over.")
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]

def chat(port, model, messages, max_tokens):
    body = json.dumps({"model": model, "messages": messages, "max_tokens": max_tokens,
                       "temperature": 0.7, "top_p": 0.95}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    for _ in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=600).read())
            return r["choices"][0]["message"]["content"] or ""
        except Exception:
            pass
    return ""

def parse_plan(reply):
    out = {}
    for key in ("CHILD FAMILY", "GROUP", "SKILL"):
        m = re.search(rf"^{re.escape(key)}\s*:\s*(.+?)\s*$", reply, re.M)
        if m: out[key] = m.group(1).strip()
    return out if len(out) == 3 else None

def extract_code(reply):
    m = re.findall(r"```python\n(.*?)```", reply, re.S)
    return max(m, key=len) if m else None

class _Skel(ast.NodeVisitor):
    def __init__(self): self.seq = []
    def generic_visit(self, node):
        self.seq.append(type(node).__name__); super().generic_visit(node)
def skeleton(src):
    try: t = ast.parse(src)
    except Exception: return None
    s = _Skel(); s.visit(t); return tuple(s.seq)
def sim(a, b): return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--out", default="rq_output/probe_crossover.jsonl")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    champs = json.loads(snaps[-1].read_text())["champions"]
    champs = list(champs.values() if isinstance(champs, dict) else champs)
    pool = [(ProblemProgram(source_code=c["source_code"]),
             (c.get("niche_group"), c.get("niche_skill"))) for c in champs]
    GROUPS = defs("group_definitions.txt").split("\n"); SKILLS = defs("skill_definitions.txt").split("\n")
    gnames = [l.split(":")[0].strip() for l in GROUPS]; snames = [l.split(":")[0].strip() for l in SKILLS]
    rng = random.Random(args.seed)

    jobs = []
    for arm in ("single", "cross", "cross_free"):
        for i in range(args.n):
            a = rng.randrange(len(pool))
            b = rng.randrange(len(pool))
            while pool[b][1] == pool[a][1] and len(pool) > 1: b = rng.randrange(len(pool))
            parents = [pool[a][0]] if arm == "single" else [pool[a][0], pool[b][0]]
            tgt = None if arm == "cross_free" else (rng.choice(gnames), rng.choice(snames))
            jobs.append((arm, i, parents, tgt))

    def run(job):
        arm, i, parents, tgt = job
        r1 = chat(args.port, args.model, stage1(parents, tgt, arm != "single"), 1024)
        plan = parse_plan(r1)
        row = {"arm": arm, "i": i, "stage1_ok": plan is not None,
               "n_parents": len(parents),
               "parent_src": [p.source_code for p in parents]}
        if not plan: return row
        row.update({"child_family": plan["CHILD FAMILY"], "plan_group": plan["GROUP"], "plan_skill": plan["SKILL"]})
        r2 = chat(args.port, args.model, stage2(parents, plan, arm != "single"), 2048)
        code = extract_code(r2)
        row["stage2_ok"] = code is not None
        if code: row["code"] = code
        return row

    print(f"arms=single/cross/cross_free  n={args.n} each  model={args.model}:{args.port}", flush=True)
    with ThreadPoolExecutor(24) as ex: rows = list(ex.map(run, jobs))
    out = ROOT / args.out
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    print(f"\n{'arm':<12}{'stage1':>8}{'stage2':>8}{'parse':>7}{'lint':>7}"
          f"{'구조유사(부모최대)':>20}{'텍스트유사':>12}{'고유skeleton':>13}")
    print("-" * 92)
    for arm in ("single", "cross", "cross_free"):
        g = [r for r in rows if r["arm"] == arm]
        s1 = sum(1 for r in g if r.get("stage1_ok")); s2 = sum(1 for r in g if r.get("stage2_ok"))
        ss, ts, skels, nlint = [], [], [], 0
        for r in g:
            c = r.get("code")
            if not c: continue
            sk = skeleton(c)
            if sk is None: continue
            skels.append(sk)
            if not lint_generator_source(c): nlint += 1
            ps = [skeleton(p) for p in r["parent_src"]]
            ss.append(max(sim(sk, p) for p in ps if p))
            ts.append(max(sim(c, p) for p in r["parent_src"]))
        uniq = len({s for s in skels})
        near = sum(1 for i, a in enumerate(skels) for b in skels[i+1:] if sim(a, b) >= 0.95)
        print(f"{arm:<12}{s1:>8}{s2:>8}{len(skels):>7}{nlint:>7}"
              f"{statistics.median(ss) if ss else float('nan'):>20.3f}"
              f"{statistics.median(ts) if ts else float('nan'):>12.3f}"
              f"{uniq:>8}/{len(skels):<4}")
    print(f"\nwritten to {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
