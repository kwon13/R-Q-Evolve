#!/usr/bin/env python3
"""Stage-1/stage-2 label experiment: three prompt arms, one confusion matrix each.

Measures what the run's logs cannot: the (plan SKILL x INFERRED_SKILL) matrix the
stage-2 self-consistency gate actually decides on, plus the plan-side label
distribution that feeds MAP coverage.

Arms
  A  baseline        prompts exactly as shipped in prompt_templates/
  B  target_cell     A + a randomly drawn EMPTY (GROUP, SKILL) cell injected into
                     stage 1 as the niche the child must fill
  C  target_rotate   B + few-shot rotation: stage 1 shows 3 of its 8 EXAMPLEs
                     (always including the target SKILL's) in random order, and
                     stage 2's 4 worked examples / 8 skill sketches / skill list
                     are shuffled per request

Parents come from an existing MAP archive snapshot; nothing is inserted anywhere.
"""
import argparse, json, os, random, re, sys, threading, time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from rq_evolve.prompts import (  # noqa: E402
    MutationTask, parse_family_plan, parse_inferred_labels,
    _load_template, _load_definitions, _render_template, _parent_problem_text,
    FAMILY_SYSTEM_PROMPT_FILE, FAMILY_USER_PROMPT_FILE,
    GENERATOR_SYSTEM_PROMPT_FILE, GENERATOR_USER_PROMPT_FILE,
)
from rq_evolve.code_utils import (  # noqa: E402
    extract_generator_code, extract_problem_template,
    strip_label_declarations, strip_module_docstring,
)
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.concepts import GROUPS, SKILLS  # noqa: E402

import requests  # noqa: E402

# --------------------------------------------------------------------------
# prompt surgery: split the shipped templates into rotatable blocks
# --------------------------------------------------------------------------
_EX_HEAD = re.compile(r"^EXAMPLE \d+ — (.+)$", re.M)


def split_family_system(text: str):
    """(head, [(skill, block)...], tail) for diff_problem_system_prompt.txt."""
    hits = list(_EX_HEAD.finditer(text))
    head = text[: hits[0].start()]
    blocks = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else None
        body = text[m.start(): end] if end else text[m.start():]
        sk = re.search(r"^SKILL:\s*(\w+)", body, re.M)
        blocks.append((sk.group(1) if sk else None, body))
    # the tail ("Now create a child family...") is glued to the last block
    last_sk, last_body = blocks[-1]
    cut = last_body.find("Now create a child family")
    tail = last_body[cut:] if cut >= 0 else ""
    blocks[-1] = (last_sk, last_body[:cut] if cut >= 0 else last_body)
    return head, blocks, tail


_WORKED = re.compile(r"^WORKED EXAMPLE \d+$", re.M)
_SKETCH = re.compile(r"^(casework|induction|contradiction|invariant|extremal_principle|"
                     r"counting|transformation|construction) -- ", re.M)


def split_generator_system(text: str):
    """(head, sketches, mid, worked_blocks, tail) for gen_program_system_prompt.txt."""
    sm = list(_SKETCH.finditer(text))
    wm = list(_WORKED.finditer(text))
    head = text[: sm[0].start()]
    sketches = []
    for i, m in enumerate(sm):
        end = sm[i + 1].start() if i + 1 < len(sm) else wm[0].start()
        sketches.append(text[m.start(): end])
    # text between last sketch and first worked example
    mid_start = sm[-1].end()
    mid = text[text.find("\n", mid_start):wm[0].start()]
    mid = text[sm[-1].start():wm[0].start()]
    mid = mid[len(sketches[-1]):]
    worked = []
    for i, m in enumerate(wm):
        end = wm[i + 1].start() if i + 1 < len(wm) else None
        worked.append(text[m.start(): end] if end else text[m.start():])
    cut = worked[-1].find("After the closing ```")
    tail = worked[-1][cut:]
    worked[-1] = worked[-1][:cut]
    return head, sketches, mid, worked, tail


EXTRA_WORKED_FILE = "gen_program_extra_examples.txt"


def _extra_worked_blocks():
    """The 4 authored WORKED EXAMPLEs covering invariant / induction /
    contradiction / construction -- the skills stage 2's shipped demo set never
    shows. Returned in the same (skill, block) shape as split_generator_system."""
    text = _load_template(EXTRA_WORKED_FILE)
    hits = list(_WORKED.finditer(text))
    out = []
    for i, m in enumerate(hits):
        end = hits[i + 1].start() if i + 1 < len(hits) else None
        block = text[m.start(): end] if end else text[m.start():]
        sk = re.search(r"^INFERRED_SKILL:\s*(\w+)", block, re.M)
        out.append((sk.group(1) if sk else None, block))
    return out


def _worked_skill(block):
    m = re.search(r"^INFERRED_SKILL:\s*(\w+)", block, re.M)
    return m.group(1) if m else None


GRAMMAR_BLOCK = """
COMPETITION-PROBLEM GRAMMAR (the CHILD FAMILY must read like a contest problem,
not like a formula lookup):

- Give at least TWO givens that have to be combined with each other. A family
  whose answer follows from substituting one parameter into a known formula
  does not qualify.
- Ask for a quantity that is NOT one of the givens: a combination of the
  unknowns, a count over a set the givens define, or an extremal value of a
  quantity derived from them.
- Prefer "Find ..." or "Determine ..." over "How many ...". Ask "How many"
  only when counting is genuinely the decisive reasoning move for TARGET SKILL.
- If the natural answer is not already a small integer, close in the contest
  idiom so that it becomes one:
      "... is m/n where m and n are relatively prime positive integers. Find m+n."
      "... Find the remainder when N is divided by 1000."
  That clause is part of the MATHEMATICS. Never add it when the answer is
  already a small integer, and never write a formatting instruction such as
  "state only the integer" -- real contest problems contain no such sentence.
"""


TARGET_BLOCK = """
TARGET NICHE (the archive cell this child must land in):
TARGET GROUP: {group}
TARGET SKILL: {skill}

The CHILD FAMILY you invent must belong to TARGET GROUP, and the decisive
reasoning move of its shortest clean solution must be TARGET SKILL. Write the
GROUP and SKILL lines as exactly these two values. If the parent family cannot
be pushed into that niche, invent the child from scratch -- the niche matters,
the parent does not.
"""


def build_stage1(parent, arm, target, rng):
    sys_text = _load_template(FAMILY_SYSTEM_PROMPT_FILE)
    if arm in ("target_rotate", "target_rotate_full", "target_grammar", "target_any_cell", "target_blind"):
        head, blocks, tail = split_family_system(sys_text)
        keep = [b for b in blocks if b[0] == target[1]]
        others = [b for b in blocks if b[0] != target[1]]
        rng.shuffle(others)
        keep += others[: max(0, 3 - len(keep))]
        rng.shuffle(keep)
        renum = []
        for i, (sk, body) in enumerate(keep, 1):
            body = _EX_HEAD.sub(lambda m: f"EXAMPLE {i} — {(sk or '').upper()}", body, count=1)
            renum.append(body)
        sys_text = head + "".join(renum) + tail
    user = _render_template(
        _load_template(FAMILY_USER_PROMPT_FILE),
        {
            "parent_template": extract_problem_template(parent.source_code)
                               or _parent_problem_text(parent),
            "parent_problem": _parent_problem_text(parent),
            "allowed_groups": _load_definitions("group_definitions.txt"),
            "allowed_skills": _load_definitions("skill_definitions.txt"),
        },
    )
    if target is not None:
        user = user + "\n" + TARGET_BLOCK.format(group=target[0], skill=target[1])
    if arm == "target_grammar":
        user = user + "\n" + GRAMMAR_BLOCK
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]


def build_stage2(parent, plan, arm, rng, target=None):
    sk_defs = _load_definitions("skill_definitions.txt")
    if arm in ("target_rotate", "target_rotate_full", "target_grammar", "target_any_cell", "target_blind"):
        lines = [l for l in sk_defs.strip().split("\n") if l.strip()]
        rng.shuffle(lines)
        sk_defs = "\n".join(lines)
    sys_text = _render_template(_load_template(GENERATOR_SYSTEM_PROMPT_FILE),
                               {"skill_definitions": sk_defs})
    if arm in ("target_rotate", "target_rotate_full", "target_grammar", "target_any_cell", "target_blind"):
        head, sketches, mid, worked, tail = split_generator_system(sys_text)
        rng.shuffle(sketches)
        if arm in ("target_rotate_full", "target_grammar", "target_any_cell", "target_blind"):
            # 8-example pool: the 4 shipped + 4 authored. Always show the one
            # demonstrating the target SKILL when the pool has it, then fill to
            # 4 at random -- same slot count as baseline, different coverage.
            pool = [(_worked_skill(w), w) for w in worked] + _extra_worked_blocks()
            want = None if arm == "target_blind" else (target[1] if target else None)
            keep = [b for b in pool if b[0] == want]
            others = [b for b in pool if b[0] != want]
            rng.shuffle(others)
            keep = keep[:1] + others[: 4 - len(keep[:1])]
            rng.shuffle(keep)
            worked = [b for _, b in keep]
        else:
            rng.shuffle(worked)
        renum = [_WORKED.sub(f"WORKED EXAMPLE {i}", w, count=1)
                 for i, w in enumerate(worked, 1)]
        sys_text = head + "".join(sketches) + mid + "".join(renum) + tail
    user = _render_template(
        _load_template(GENERATOR_USER_PROMPT_FILE),
        {
            "parent_template": extract_problem_template(parent.source_code)
                               or _parent_problem_text(parent),
            "parent_source": strip_label_declarations(strip_module_docstring(parent.source_code)),
            "new_problem": plan["CHILD FAMILY"],
        },
    )
    return [{"role": "system", "content": sys_text}, {"role": "user", "content": user}]


# --------------------------------------------------------------------------
_session = threading.local()


def chat(base_url, model, messages, temperature, top_p, max_tokens, retries=3):
    s = getattr(_session, "s", None)
    if s is None:
        s = requests.Session(); _session.s = s
    payload = {"model": model, "messages": messages, "temperature": temperature,
               "top_p": top_p, "max_tokens": max_tokens}
    for k in range(retries):
        try:
            r = s.post(f"{base_url}/chat/completions", json=payload, timeout=1200)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:
            if k == retries - 1:
                return None
            time.sleep(2 * (k + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive_iter243.json")
    ap.add_argument("--base-url", default="http://127.0.0.1:8311/v1")
    ap.add_argument("--model", default="qwen3-4b-base")
    ap.add_argument("--n-per-arm", type=int, default=240)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rq_output/gate_experiment")
    ap.add_argument("--arms", default="baseline,target_cell,target_rotate")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    arch = json.load(open(args.archive))
    parents, occupied = [], set()
    for c in arch["champions"]:
        p = ProblemProgram(source_code=c["source_code"], program_id=c["program_id"])
        p.niche_group, p.niche_skill = c["niche_group"], c["niche_skill"]
        parents.append(p)
        occupied.add((GROUPS[c["niche_group"]], SKILLS[c["niche_skill"]]))
    empty = [(g, s) for g in GROUPS for s in SKILLS if (g, s) not in occupied]
    by_skill = defaultdict(list)
    for g, s in empty:
        by_skill[s].append(g)
    print(f"[exp] {len(parents)} parents, {len(occupied)} occupied cells, {len(empty)} empty cells")

    os.makedirs(args.out, exist_ok=True)
    for arm in args.arms.split(","):
        rng = random.Random(args.seed)
        jobs = []
        for i in range(args.n_per_arm):
            parent = rng.choice(parents)
            target = None
            if arm == "target_any_cell":
                # Pure MAP-Elites illumination: every cell is a legal target, so a
                # broken or solved-out champion can be challenged in place.
                target = (rng.choice(GROUPS), rng.choice(SKILLS))
            elif arm != "baseline":
                sk = SKILLS[i % len(SKILLS)]                 # balanced rows
                gs = by_skill.get(sk) or GROUPS
                target = (rng.choice(gs), sk)
            jobs.append({"i": i, "arm": arm, "parent": parent, "target": target,
                         "rng": random.Random(args.seed * 100003 + i)})

        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            def s1(j):
                m = build_stage1(j["parent"], arm, j["target"], j["rng"])
                j["stage1_reply"] = chat(args.base_url, args.model, m, 0.4, 0.95, 5000)
                j["plan"] = parse_family_plan(j["stage1_reply"] or "")
                return j
            jobs = list(ex.map(s1, jobs))
        n1 = sum(1 for j in jobs if j["plan"])
        print(f"[{arm}] stage1 {n1}/{len(jobs)} parsed  ({time.time()-t0:.0f}s)")

        live = [j for j in jobs if j["plan"]]
        t1 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            def s2(j):
                m = build_stage2(j["parent"], j["plan"], arm, j["rng"], j["target"])
                j["stage2_reply"] = chat(args.base_url, args.model, m, 0.0, 1.0, 5000)
                j["code"] = extract_generator_code(j["stage2_reply"] or "")
                ig, isk = parse_inferred_labels(j["stage2_reply"] or "")
                j["inferred_group"], j["inferred_skill"] = ig, isk
                return j
            live = list(ex.map(s2, live))
        print(f"[{arm}] stage2 done ({time.time()-t1:.0f}s)")

        rows = []
        for j in jobs:
            plan = j.get("plan")
            rows.append({
                "arm": arm, "i": j["i"],
                "parent_id": j["parent"].program_id,
                "parent_cell": [GROUPS[j["parent"].niche_group], SKILLS[j["parent"].niche_skill]],
                "target_cell": list(j["target"]) if j["target"] else None,
                "stage1_parsed": bool(plan),
                "plan_group": plan["GROUP"] if plan else None,
                "plan_skill": plan["SKILL"] if plan else None,
                "child_family": plan["CHILD FAMILY"][:600] if plan else None,
                "code_ok": bool(j.get("code")),
                "code": j.get("code"),
                "inferred_group": j.get("inferred_group"),
                "inferred_skill": j.get("inferred_skill"),
                "stage1_reply": (j.get("stage1_reply") or "")[:4000],
                "stage2_reply": (j.get("stage2_reply") or "")[:12000],
            })
        path = os.path.join(args.out, f"raw_{arm}{args.tag}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{arm}] wrote {path}")


if __name__ == "__main__":
    main()
