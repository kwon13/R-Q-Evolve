#!/usr/bin/env python
"""Two-stage mutation: write the PROBLEM in prose, then write its generator.

    bash scripts/serve_calibration.sh
    python scripts/probe_diff_mutation.py --n 24

Replaces asking one call to mutate a parent program, which did not work on a
base policy: 85% of children in a live run were >95% identical to their parent,
and the rest had lost the assert or the two label lines. Both are the same
thing -- a model shown a complete program reproduces it.

Stage A never sees a program. It is given the parent's problem FAMILY (the
f-string, so the braces make "change the numbers" visibly empty) plus worked
examples of what a structural mutation looks like, and returns a new family and
its two labels.

Stage B is given that fixed family and the parent pair -- family and generator
-- as a worked example of the statement-to-program mapping. Because the child
family is already fixed and different, reproducing the example would emit the
wrong statement: child/parent source similarity fell from 0.99 to 0.17.

The labels never travel through stage B. Stage A chose them and the harness
appends them, so they cannot be dropped after `return` and cannot disagree with
the problem they describe.

Reports what happened at each stage, not just a pass rate:
  A  did the problem change (similarity to the parent's statement, cell moved)
  B  did a generator come back, and how much of the parent is in it
  V  lint + AST contract + seeds 0-4, the same gates the run uses
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from string import Template

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.code_utils import (  # noqa: E402
    extract_generator_code,
    extract_problem_template,
    lint_generator_source,
    lint_problem_instance,
    set_label_declarations,
    strip_label_declarations,
    strip_module_docstring,
)
from rq_evolve.ast_contract import check_generator_contract  # noqa: E402
from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.prompts import (  # noqa: E402
    _load_definitions,
    parse_inferred_labels,
    GROUP_DEFINITIONS_FILE,
    SKILL_DEFINITIONS_FILE,
)


TPL = ROOT / "prompt_templates"


def _tpl(name: str) -> str:
    return (TPL / name).read_text(encoding="utf-8")


# Stage 1 emits four fields. Two of them contain a space, so the lookahead
# that ends one field has to know every header -- a header it does not list is
# swallowed into the previous field's value.
_KEYS = "STRUCTURAL MUTATION|CHILD FAMILY|WHY FINITE|GROUP|SKILL"


def _field(text: str, key: str) -> str:
    """The LAST `KEY: value` whose value is not a leftover `<...>` placeholder.

    A base policy re-emits the prompt before answering it: measured here, 13 of
    24 stage-A replies opened with the template's own placeholder lines and only
    then wrote the real six. Taking the first match reads the placeholder and
    calls the reply incomplete; taking the last reads the answer. Placeholder
    values are skipped outright so a reply that echoes AFTER answering does not
    overwrite a good field with `<...>`.
    """
    best = ""
    for m in re.finditer(
        rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_KEYS})[ \t]*:|\Z)",
        text, re.M | re.S,
    ):
        value = m.group(1).strip()
        if value and not value.startswith("<"):
            best = value
    return best


def _label(value: str, allowed) -> str | None:
    v = value.strip().strip('"\'').split()[0].strip('.,') if value.strip() else ""
    return v if v in allowed else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8801/v1")
    ap.add_argument("--model", default="qwen3-8b-base")
    ap.add_argument("--n", type=int, default=24)
    # The two stages want opposite things. Stage A is invention: at 0 it
    # returns the same child for the same parent, and the parent pool is small,
    # so the batch collapses. Stage B is transcription of a fixed spec onto a
    # fixed file -- there is nothing to explore, and sampling only adds ways to
    # mistype a SEARCH block.
    ap.add_argument("--temperature-a", type=float, default=0.4)
    ap.add_argument("--top-p-a", type=float, default=0.95)
    ap.add_argument("--temperature-b", type=float, default=0.0)
    ap.add_argument("--top-p-b", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--seed-dir", default="seed_programs")
    ap.add_argument("--out", default="rq_output/probe_diff")
    ap.add_argument("--show", type=int, default=0, help="print N full replies")
    args = ap.parse_args()

    from openai import OpenAI

    client = OpenAI(base_url=args.base_url, api_key="none")

    parents = [ProblemProgram.from_file(p) for p in sorted(Path(args.seed_dir).glob("*.py"))]
    jobs = [parents[i % len(parents)] for i in range(args.n)]
    groups, skills = _load_definitions(GROUP_DEFINITIONS_FILE), _load_definitions(SKILL_DEFINITIONS_FILE)

    def ask(system: str, user: str, temperature: float, top_p: float) -> str:
        r = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=temperature, top_p=top_p, max_tokens=args.max_tokens,
        )
        return r.choices[0].message.content or ""

    # ---- stage A: the problem, in prose, with no program in sight ----------
    sys_a, usr_a = _tpl("diff_problem_system_prompt.txt"), _tpl("diff_problem_user_prompt.txt")

    def stage_a(parent):
        inst = parent.execute(seed=0)
        if inst is None:
            return parent, None, "parent did not run"
        # The family, not the instance: shown concrete numbers, the model
        # changes the numbers. Falls back to the instance if the parent has no
        # recognisable `problem = ...` assignment.
        template = extract_problem_template(parent.source_code) or inst.problem
        user = Template(usr_a).safe_substitute(
            parent_template=template, parent_problem=inst.problem,
            allowed_groups=groups, allowed_skills=skills,
        )
        return parent, inst, ask(sys_a, user, args.temperature_a, args.top_p_a)

    print(f"[diff-probe] {args.n} children, {args.model}, "
          f"A temp={args.temperature_a}, B temp={args.temperature_b}")
    with ThreadPoolExecutor(max_workers=min(args.n, 32)) as ex:
        a_out = list(ex.map(stage_a, jobs))

    plans = []
    for parent, inst, reply in a_out:
        row = {"parent": f"{parent.get_group()}/{parent.get_skill()}", "raw_a": reply}
        if inst is None:
            row["stage_a"] = "parent_failed"; plans.append((parent, inst, row, None)); continue
        p = {k: _field(reply, k) for k in ("CHILD FAMILY", "STRUCTURAL MUTATION")}
        p["GROUP"], p["SKILL"] = _label(_field(reply, "GROUP"), GROUPS), _label(_field(reply, "SKILL"), SKILLS)
        missing = [k for k, v in p.items() if not v]
        if missing:
            row["stage_a"] = "incomplete:" + ",".join(missing)
            plans.append((parent, inst, row, None)); continue
        row["stage_a"] = "ok"
        row["problem_sim"] = difflib.SequenceMatcher(
            None, p["CHILD FAMILY"], inst.problem
        ).ratio()
        row["new_cell"] = f"{p['GROUP']}/{p['SKILL']}"
        plans.append((parent, inst, row, p))

    # ---- stage B: turn the fixed child family into a program --------------
    #
    # The parent goes in as a worked example of the statement-to-program
    # mapping, not as something to mutate. The child family is already fixed
    # and different, so reproducing the example would visibly emit the wrong
    # statement -- which is what finally broke the copying: child/parent source
    # similarity fell from 0.99 under whole-program rewriting to 0.17.
    sys_b, usr_b = _tpl("gen_program_system_prompt.txt"), _tpl("gen_program_user_prompt.txt")

    def stage_b(item):
        parent, inst, row, p = item
        if p is None:
            return row
        user = Template(usr_b).safe_substitute(
            new_problem=p["CHILD FAMILY"], new_change=p["STRUCTURAL MUTATION"],
            parent_source=strip_label_declarations(
                strip_module_docstring(parent.source_code)
            ),
            parent_template=extract_problem_template(parent.source_code) or inst.problem,
        )
        row["raw_b"] = ask(sys_b, user, args.temperature_b, args.top_p_b)
        _inf_g, inf_s = parse_inferred_labels(row["raw_b"])
        row["stage1_skill"] = p["SKILL"]
        row["inferred_skill"] = inf_s
        row["skill_match"] = (p["SKILL"] == inf_s) if inf_s else False

        child_src = extract_generator_code(row["raw_b"])
        if child_src is None:
            row["stage_b"] = "no_code: no parseable generate() in the reply"
            return row
        # The labels come from stage 1: they cannot be omitted, and they cannot
        # disagree with the problem they describe.
        child_src = set_label_declarations(child_src, p["GROUP"], p["SKILL"])
        row["stage_b"] = "ok"
        row["source"] = child_src
        row["parent_sim"] = difflib.SequenceMatcher(None, child_src, parent.source_code).ratio()
        return row

    with ThreadPoolExecutor(max_workers=min(args.n, 32)) as ex:
        rows = list(ex.map(stage_b, plans))

    # ---- verify: the same gates the run applies ---------------------------
    for row in rows:
        if row.get("stage_b") != "ok":
            continue
        src = row["source"]
        errs = lint_generator_source(src) + [str(f) for f in check_generator_contract(src)]
        if errs:
            row["verify"] = "; ".join(errs[:2]); continue
        prog = ProblemProgram(source_code=src)
        g, s = prog.declared_group(), prog.declared_skill()
        if g not in GROUPS or s not in SKILLS:
            row["verify"] = f"bad labels {g}/{s}"; continue
        row["child_cell"] = f"{g}/{s}"
        texts = set()
        for seed in range(5):
            i = prog.execute(seed=seed)
            if i is None:
                row["verify"] = f"seed {seed}: {prog.last_execution_error}"; break
            if lint_problem_instance(i):
                row["verify"] = f"seed {seed} lint: {lint_problem_instance(i)[0]}"; break
            texts.add(i.problem)
        else:
            row["verify"] = "ok" if len(texts) >= 2 else "one problem text across seeds"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "probe.jsonl").write_text("\n".join(json.dumps(r) for r in rows))

    # ---- report -----------------------------------------------------------
    n = len(rows)
    a_ok = [r for r in rows if r.get("stage_a") == "ok"]
    b_ok = [r for r in rows if r.get("stage_b") == "ok"]
    v_ok = [r for r in rows if r.get("verify") == "ok"]
    print(f"\n  A  problem written    {len(a_ok)}/{n}")
    if a_ok:
        sims = [r["problem_sim"] for r in a_ok]
        print(f"       similarity to the parent's statement: mean {sum(sims)/len(sims):.2f}, "
              f"{sum(1 for s in sims if s > .8)} of {len(sims)} above 0.8")
        moved = sum(1 for r in a_ok if r["new_cell"] != r["parent"])
        print(f"       declared a different cell than the parent: {moved}/{len(a_ok)}")
    print(f"  B  generator written  {len(b_ok)}/{n}")
    if b_ok:
        print("       child/parent source similarity "
              f"{sum(r['parent_sim'] for r in b_ok)/len(b_ok):.2f}")
    
    # Self-Consistency Gate Reporting
    b_with_skill = [r for r in b_ok if r.get("inferred_skill")]
    b_matched = [r for r in b_ok if r.get("skill_match")]
    b_mismatched = [r for r in b_ok if r.get("inferred_skill") and not r.get("skill_match")]
    b_parse_failed = [r for r in b_ok if not r.get("inferred_skill")]

    print(f"  G  Skill Consistency  {len(b_matched)}/{len(b_ok)} matches ({(len(b_matched)/len(b_ok)*100 if b_ok else 0):.1f}%)")
    print(f"       mismatches: {len(b_mismatched)}, parse failed: {len(b_parse_failed)}")
    if b_mismatched:
        print("       Sample Mismatches (Stage 1 Plan != Stage 2 Inferred):")
        for r in b_mismatched[:5]:
            print(f"         - Stage 1: {r.get('stage1_skill')}  vs  Stage 2 Inferred: {r.get('inferred_skill')}")

    print(f"  V  verified           {len(v_ok)}/{n}")
    if v_ok:
        kept = sum(1 for r in v_ok if r.get("child_cell") == r["parent"])
        print(f"       still in the parent's cell: {kept}/{len(v_ok)}")
        cells = {r.get("child_cell") for r in v_ok}

        print(f"       distinct cells: {len(cells)}  {sorted(c for c in cells if c)}")

    fails = {}
    for r in rows:
        why = (r.get("verify") if r.get("stage_b") == "ok" else r.get("stage_b")) \
              if r.get("stage_a") == "ok" else r.get("stage_a")
        if why and why != "ok":
            fails[str(why)[:70]] = fails.get(str(why)[:70], 0) + 1
    if fails:
        print("\n  where they died:")
        for k, v in sorted(fails.items(), key=lambda t: -t[1])[:8]:
            print(f"    {v:3d}  {k}")
    for r in rows[: args.show]:
        print("\n" + "=" * 70 + f"\nparent {r['parent']}\n--- stage A ---\n{r.get('raw_a','')[:1200]}"
              f"\n--- stage B ---\n{r.get('raw_b','')[:1200]}")
    print(f"\n  [diff-probe] {out/'probe.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
