#!/usr/bin/env python
"""Stage-2 operator probe. Stage 1 runs ONCE per parent; every arm gets the SAME child family.

What this fixes about scripts/probe_operators.py:
  D1  no operator instruction contradicts the system prompt: stage 1 is the
      shipped prompt, unmodified, and is not an experimental variable here.
  D2  shot rotation is ON in every arm (as it ships) and uses the SAME seed for
      a given parent across arms, so the four worked examples shown are
      identical between arms and cancel in the paired comparison.
  D3  arms are PAIRED. One stage-1 call per parent produces one frozen plan;
      all N arms implement that identical plan for that identical parent. Every
      between-arm difference is the stage-2 user turn and nothing else. This
      matters: measured on the old probe, children whose FAMILY stayed close to
      the parent's family copied its CODE far more (code text sim 0.803 in the
      top half of family-similarity vs 0.457 in the bottom half), so unpaired
      arms confound the operator with how far stage 1 happened to wander.
  D4  every headline is reported twice, over parseable children and over
      EXECUTABLE children, because those disagree (full_rewrite 0.766 -> 0.933).
  D5  stage 2 is the variable. That is the point.

    python scripts/probe_stage2_operators.py --n 30 --port 8701 --model step160
"""
from __future__ import annotations
import argparse, ast, difflib, importlib.util, json, random, re, statistics, sys, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPS_DIR = ROOT / "prompt_templates" / "stage2_ops"
TPL = ROOT / "prompt_templates"


def _load_by_path(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Loaded by path so the probe does not import the rq_evolve package (which pulls torch).
SF = _load_by_path("sf", ROOT / "src" / "rq_evolve" / "structural_fingerprint.py")

sys.path.insert(0, str(ROOT / "src"))
from rq_evolve.code_utils import (extract_problem_template, strip_label_declarations,   # noqa: E402
                                  strip_module_docstring)
from rq_evolve.prompts import build_family_task, build_generator_task, parse_family_plan  # noqa: E402
from rq_evolve.program import ProblemProgram                                             # noqa: E402

# ---------------------------------------------------------------------------
# Arms. Each is (user-template file, system transform).
#   "shots"  keeps the shipped system prompt (worked examples rotated as shipped)
#   "noshots" strips every WORKED EXAMPLE block and splices in a note
# ---------------------------------------------------------------------------
ARMS = {
    "parent_source":        ("01_parent_source.user.txt",        "shots",  None),
    "parent_statement_only":("02_parent_statement_only.user.txt", "shots",  None),
    "spec_only":            ("03_spec_only.user.txt",             "shots",  "03_spec_only.system_note.txt"),
    "parent_no_shots":      ("04b_parent_no_shots.user.txt",      "noshots","04_shots_stripped.system_note.txt"),
    "bare_contract":        ("04_bare_contract.user.txt",         "noshots","04_shots_stripped.system_note.txt"),
    "skeleton_forbidden":   ("05_skeleton_forbidden.user.txt",    "shots",  None),
    "donor_source":         ("06_donor_source.user.txt",          "shots",  None),
    "derivation_required":  ("07_derivation_required.user.txt",   "shots",  None),
    "derivation_spec_only": ("08_derivation_spec_only.user.txt",  "shots",  "03_spec_only.system_note.txt"),
}

_WORKED = re.compile(r"^WORKED EXAMPLE \d+$", re.M)


def strip_worked_examples(system_prompt: str, note: str) -> str:
    """Drop every WORKED EXAMPLE block, keep head + skill sketches + tail."""
    hits = list(_WORKED.finditer(system_prompt))
    if not hits:
        return system_prompt
    head = system_prompt[: hits[0].start()]
    last = system_prompt[hits[-1].start():]
    cut = last.find("After the closing ```")
    tail = last[cut:] if cut >= 0 else ""
    return head + note.strip() + "\n\n" + tail


def build_arm_task(arm, parent, plan, donor, rng_seed):
    """Stage-2 messages for one arm. Shot rotation seeded identically per parent."""
    tpl_file, shots, note_file = ARMS[arm]
    # Shipped builder gives us the system prompt with rotation applied; the
    # SAME rng_seed for every arm of one parent makes the shots identical.
    base = build_generator_task(parent, plan, rotate_shots=True, rng=random.Random(rng_seed))
    system_prompt = base.messages[0]["content"]
    if shots == "noshots":
        system_prompt = strip_worked_examples(system_prompt, (OPS_DIR / note_file).read_text())
    elif note_file:
        marker = "Four worked transformations follow"
        note = (OPS_DIR / note_file).read_text().strip()
        idx = system_prompt.find(marker)
        system_prompt = (system_prompt[:idx] + note + "\n\n" + system_prompt[idx:]
                         if idx >= 0 else system_prompt + "\n\n" + note)

    fields = {
        "parent_template": extract_problem_template(parent.source_code) or "",
        "parent_source": strip_label_declarations(strip_module_docstring(parent.source_code)),
        "new_problem": plan["CHILD FAMILY"],
        "donor_template": extract_problem_template(donor.source_code) or "" if donor else "",
        "donor_source": (strip_label_declarations(strip_module_docstring(donor.source_code))
                         if donor else ""),
        "parent_fingerprint": "",
    }
    fp = SF.structural_fingerprint(parent.source_code)
    fields["parent_fingerprint"] = SF.render_fingerprint(fp) if fp else "(parent did not parse)"
    from string import Template
    user = Template((OPS_DIR / tpl_file).read_text()).safe_substitute(fields)
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


# ---------------------------------------------------------------------------
def chat(port, model, messages, max_tokens, temp=0.7):
    import urllib.request
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


def code_of(reply):
    m = re.findall(r"```python\n(.*?)```", reply, re.S)
    return max(m, key=len) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30, help="parents (each runs every arm)")
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--out", default="rq_output/probe_stage2_operators.jsonl")
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    snaps = sorted((ROOT / args.archive).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    ch = json.loads(snaps[-1].read_text())["champions"]
    ch = list(ch.values() if isinstance(ch, dict) else ch)
    pool = [dict(prog=ProblemProgram(source_code=c["source_code"]), src=c["source_code"],
                 cell=(c.get("niche_group"), c.get("niche_skill")),
                 fp=SF.structural_fingerprint(c["source_code"])) for c in ch]
    rng = random.Random(args.seed)
    parents = [pool[rng.randrange(len(pool))] for _ in range(args.n)]

    def farthest_donor(p):
        """Donor = champion whose coarse shape differs and whose cell differs."""
        cands = [q for q in pool if q["cell"] != p["cell"] and q["fp"] and p["fp"]
                 and SF.shape_key(q["fp"]) != SF.shape_key(p["fp"])]
        return random.Random(args.seed).choice(cands)["prog"] if cands else pool[0]["prog"]

    # ---- STAGE 1: once per parent, shipped prompt, frozen -------------------
    def s1(idx):
        p = parents[idx]
        task = build_family_task(p["prog"], rotate_shots=True,
                                 rng=random.Random(args.seed * 1000 + idx))
        return parse_family_plan(chat(args.port, args.model, task.messages, 1200))

    with ThreadPoolExecutor(16) as ex:
        plans = list(ex.map(s1, range(args.n)))
    ok = [i for i, pl in enumerate(plans) if pl]
    print(f"stage 1: {len(ok)}/{args.n} plans parsed; each is reused by all {len(ARMS)} arms", flush=True)

    # ---- STAGE 2: every arm, same plan, same shots -------------------------
    jobs = [(arm, i) for arm in ARMS for i in ok]

    def s2(job):
        arm, i = job
        p = parents[i]
        msgs = build_arm_task(arm, p["prog"], plans[i], farthest_donor(p), args.seed * 1000 + i)
        reply = chat(args.port, args.model, msgs, 2400)
        return {"arm": arm, "parent_i": i, "parent_src": p["src"],
                "child_family": plans[i]["CHILD FAMILY"], "plan_skill": plans[i]["SKILL"],
                "code": code_of(reply)}

    with ThreadPoolExecutor(24) as ex:
        rows = list(ex.map(s2, jobs))
    (ROOT / args.out).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {ROOT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
