#!/usr/bin/env python
"""Paired, factorial operator probe. The corrected replacement for probe_operators.py.

WHAT WAS WRONG WITH THE FIRST PROBE, AND WHERE EACH FIX LIVES
-------------------------------------------------------------
D1  CONTRADICTORY ORDERS. stage1() loaded diff_problem_system_prompt.txt raw.
    Its first two lines order "Create a structurally different child family" and
    "Changing only parameters, coefficients, ranges, names, wording ... does not
    count", while six of the ten operator paragraphs ordered the opposite. Those
    arms measured which contradictory order wins, not what the operator does.
    The tell: the `parameter` arm, which sits at similarity 1.000 by
    construction, measured 0.648 -- BELOW `scale` at 0.933.
    FIX: `arm_system_prompt()`. The shipped head is split off with
    prompts._split_family_system and REPLACED by exactly one strength order --
    the arm's own. The well-posedness sentence is the only line of the shipped
    head that survives into every arm, because it is a validity requirement and
    not a statement about how far to move. The tail's
    "STRUCTURAL MUTATION: <...substantive change in TARGET or DECISIVE REASONING
    MOVE>" descriptor is retuned too (the KEY is kept so parse_family_plan is
    unchanged), since it is a third copy of the same order.
    Two control arms keep the shipped head verbatim, so for the first time
    "what ships today" is one of the columns.

D2  MISSING ROTATION. The probe showed all 8 stage-1 EXAMPLE blocks; the shipped
    path is rotate_few_shots=True (config.py:178) -> a random 3 of 8, and a
    random 4 of 8 for stage 2 including gen_program_extra_examples.txt. With the
    fixed set, 20.4-25.6% of children copy a canned example and EXAMPLE 8 alone
    takes 12-16% (prompts.py:468-478).
    FIX: every prompt is built by prompts.build_family_task /
    build_generator_task with rotate_shots=True. Nothing about rotation is
    reimplemented here. `shot_copy_max` is reported so the copying is measured
    rather than assumed.

D3  UNPAIRED PARENTS. Parents were redrawn per (op, i), so arms did not share
    parents. Measured on the probe's own 300 rows: parent identity carries
    ICC = 0.53 of the variance in skeleton similarity (between-parent SD 0.155,
    within-parent SD 0.144), i.e. MORE than half of every between-arm difference
    the probe reported was the parent draw.
    FIX: `build_blocks()` + the job loop in `main()`. One block list, built once,
    consumed by every arm in the same order. The per-block seed is derived from
    the BLOCK ONLY -- so few-shot rotation, the donor draw, the adaptive
    strength, and the vLLM sampler seed are common random numbers across arms.
    Pairing on the parent alone removes 53% of the nuisance variance; blocking
    the rotation too removes more.

D4  METRIC OVER THE WRONG POPULATION. probe_operators_report.py accumulated
    similarity BEFORE the execution check, so its headline was over parseable
    children while the 0.996 archive figure is over ACCEPTED champions.
    Restricted to executables, full_rewrite went 0.766 -> 0.933.
    FIX: generation and scoring are ONE script. `score()` filters to executable
    children before any statistic is computed, and it is not possible to run the
    report over a different population than the one the table's `exec` column
    names. Every arm's attempted / stage1 / stage2 / exec / gated counts are
    printed on the same row as its similarity, so an arm cannot win by producing
    nothing.

D5  STAGE 2 NEVER VARIED. Stage 2 is handed the parent SOURCE as its worked
    example and is the likely copy vector; it was identical in every arm, so
    every stage-1 effect was measured through whatever ceiling stage 2 imposes.
    FIX: S2 is a real factor (`--s2`). ONE stage-1 plan per (arm, block) is
    implemented under every S2 level, so stage 1 is literally identical down the
    S2 columns and the interaction S1 x S2 is estimable. If stage 2 is the copy
    vector, this design shows it as a main effect of S2 plus a compressed range
    of S1 at the `prod` level -- which a sequential sweep would have reported as
    "no operator matters".

WHAT IS KEPT FROM probe_operators.py
------------------------------------
    chat()        the urllib client (extended with seed/top_p/finish_reason,
                  as probe_stage2_ablation.py already does)
    code_of()     the ```python fence extractor
    _prob()       the parent's seed-0 statement
    load_pool()   the archive-snapshot -> parent pool block out of its main()
    T() / D()     the template readers
    the OPS dict's SHAPE (its text comes from probe_operators_ops_replacement.py)
DROPPED, and why:
    stage1()      it is the D1 bug (raw system prompt) and the D2 bug (no
                  rotation); replaced by build_family_task + arm_system_prompt
    stage2()      D2/D5; replaced by build_generator_task + the S2 factor
    parse_plan()  weaker than prompts.parse_family_plan, which handles the base
                  policy re-emitting the template before answering it
    skel()/sim()  the NodeVisitor walk is not archive.program_skeleton's walk,
                  so its numbers were not commensurable with the 0.996 figure
    the jobs loop it is the D3 bug

SAMPLE SIZE, from the first probe's own 300 rows
------------------------------------------------
sigma_within = 0.144, sigma_between-parent = 0.155, sigma_unpaired = 0.212,
executability = 0.25 (measured on this harness's smoke runs, not the old 0.40 --
rotation and the production stage-2 framing both cost yield).

  UNPAIRED, one pairwise comparison, delta = 0.10 median skeleton similarity,
  alpha 0.05 two-sided, power 0.80:
      n = 7.85 * (2 * 0.212^2) / 0.10^2 = 71 EXECUTABLE children per arm.
      The first probe had 9-15. It was 5-8x short, which is why nothing on its
      table separated.

  PAIRED AT THE PARENT with R replicates, Dunnett-style against one control
  over k = 15 comparisons (Bonferroni z = 2.94; Dunnett is slightly looser):
      n_parents = (2.94 + 0.842)^2 * (2 * 0.144^2 / (R * 0.25)) / 0.10^2
      R = 8  ->  29.6 parents needed; 48 * (1 - 0.75^8)^2 = 39 available.  OK
      R = 6  ->  39.5 needed; 32 available.                           FAILS
      R = 4  ->  59.2 needed; 22 available.                           FAILS
  Wilcoxon instead of t costs a further /0.86 -> 34 parents at R = 8. Still OK.

  => --blocks 384  (48 parents x 8 replicates).

WALL CLOCK, measured on step160 at :8701
----------------------------------------
  stage 1  146 ms/call amortized at --concurrency 96 (240 sustained calls)
  stage 2  238-838 ms/call depending on batch depth (longer prompt, ~330 out tok)
  15 LLM arms x 384 blocks = 5760 stage-1 calls           ~14 min
  x 0.85 parse rate x 2 S2 levels = 9792 stage-2 calls    ~65-135 min
  => the |S2| = 2 screen is 1.5-2.5 h on one GPU. |S2| = 4 is 3-5 h.

    # 0. servers (once)
    bash scripts/serve_calibration.sh                     # 8401 4B, 8801 8B

    # 1. parent evidence, cached -- makes `feedback` a real operator instead of
    #    a constant (34 of 48 parents sit at s_hat > 0.85 and 30 at R_Q = 0.0)
    python scripts/probe_operators_paired.py evidence --port 8401 --model qwen3-4b-base

    # 2. the factorial screen
    python scripts/probe_operators_paired.py run --blocks 384 --s2 prod,none \
        --port 8701 --model step160 --concurrency 96

    # 3. score (also runs automatically at the end of `run`)
    python scripts/probe_operators_paired.py score

    # 4. difficulty: dump survivors in batches, then run the printed commands
    python scripts/probe_operators_paired.py dump --limit-per-arm 20 --dump-s2 prod
"""
from __future__ import annotations

import argparse
import ast
import asyncio
import collections
import difflib
import hashlib
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from rq_evolve.archive import MAPElitesArchive                       # noqa: E402
from rq_evolve.ast_contract import check_generator_contract          # noqa: E402
from rq_evolve.code_utils import (                                   # noqa: E402
    answer_is_bare_draw,
    answer_leaks_in_every_instance,
    extract_problem_template,
    lint_generator_source,
    lint_mutation_generator_source,
    set_label_declarations,
    strip_label_declarations,
    strip_module_docstring,
)
from rq_evolve.concepts import GROUPS, SKILLS, validate_label_decl   # noqa: E402
from rq_evolve.program import ProblemProgram                         # noqa: E402
from rq_evolve.prompts import (                                      # noqa: E402
    _EXAMPLE_HEAD,
    _renumber,
    _split_family_system,
    build_family_task,
    build_generator_task,
    parse_family_plan,
)
from route_signature import route_sim, route_signature               # noqa: E402

TPL = ROOT / "prompt_templates"
def T(n: str) -> str: return (TPL / n).read_text(encoding="utf-8")     # kept from probe_operators.py
def D(n: str) -> str: return T(n).strip()                              # kept from probe_operators.py


# ===========================================================================
# D1. The arms. Each supplies the ONE strength order that replaces the shipped
# head's three copies of "make it structurally different".
# ===========================================================================
#
# Text for local_diff / directed / adaptive_* / feedback / donor / simplify comes
# from scripts/probe_operators_ops_replacement.py, which rewrote the arms that
# did not implement the operator they were named after. The rationale for each
# rewrite lives in that file and is not repeated here.

SHIPPED_HEAD = None          # filled by arm_system_prompt() on first use

WELLPOSED = (
    "The CHILD FAMILY must be self-contained and well-posed as written: its "
    "requested answer must exist, be finite, unique, and integer-valued.\n"
)
# Replaces the shipped head's "The examples below illustrate the intended
# STRENGTH of Structural Mutation ... the child may move far away from the
# parent's objects". That sentence is a fourth strength order and it contradicts
# every conservative arm.
SHOTS_NOTE = (
    "The examples below show the required OUTPUT FORMAT and the level of detail "
    "expected in each field. They are not templates to copy, and they are NOT "
    "the amount of change you are being asked for. The OPERATOR paragraph is the "
    "only instruction about how far the child may move from the parent.\n"
)
NEUTRAL_OPEN = (
    "You are given a parameterized competition-math problem family. Write a "
    "CHILD family from it, following the OPERATOR below exactly.\n"
)

# The shipped tail's field descriptor is a third copy of the structural order.
# The KEY stays "STRUCTURAL MUTATION" so prompts.parse_family_plan is untouched.
_TAIL_DESC = re.compile(
    r"^STRUCTURAL MUTATION: <[^>]*>$", re.M)
_TAIL_DESC_NEUTRAL = ("STRUCTURAL MUTATION: <one sentence naming exactly what "
                      "you changed and what you left alone>")

# ... and so is the CHILD FAMILY descriptor one line below it. Measured: with
# the shipped descriptor in place ("<the NEW parameterized natural-language
# problem family>"), the local_diff arm emitted a full rewritten family instead
# of a patch on 8 of 8 blocks and the arm returned zero children. The SYSTEM
# tail's field description outranks the OPERATOR paragraph in the user turn, so
# an arm whose output SHAPE differs has to own its descriptor too.
_TAIL_FAMILY = re.compile(r"^CHILD FAMILY: <[^>]*>$", re.M)
OPS_TAIL: dict[str, str] = {
    "local_diff": ("CHILD FAMILY: <the SEARCH/REPLACE block described in the "
                   "OPERATOR, and nothing else -- do NOT write out the family>"),
    "parameter_llm": ("CHILD FAMILY: <the parent's family reproduced with only "
                      "its numeric ranges and constants changed, using "
                      "{descriptive_parameter_names} for the quantities that vary>"),
    "local_diff_prose": ("CHILD FAMILY: <the parent's family reproduced with "
                         "exactly one clause replaced and every other clause "
                         "word for word, using {descriptive_parameter_names} "
                         "for quantities that vary>"),
    "simplify": ("CHILD FAMILY: <the parent's family with the redundant material "
                 "deleted and nothing else altered, using "
                 "{descriptive_parameter_names} for quantities that vary; or the "
                 "single word NONE if nothing in the parent is redundant>"),
}

OPS: dict[str, str] = {

# ---- controls -------------------------------------------------------------
# ctl_shipped isolates the head swap: shipped head, no operator paragraph.
"ctl_shipped": "",
# ctl_production is the actual run: shipped head PLUS target-cell injection
# (config.target_cell_injection=True). Handled in arm_system_prompt/user turn.
"ctl_production": "",

# ---- conservative arms (these are the ones D1 broke) ----------------------
# ShinkaEvolve diff mode is a SEARCH/REPLACE patch APPLIED BY THE HARNESS.
# "change exactly one clause and nothing else" is a promise; a patch is a
# mechanism, so nothing outside it can drift.
"local_diff": """
Change EXACTLY ONE clause of the parent family. Express it as a PATCH, not as a
rewritten family: as the value of the CHILD FAMILY field, write the block

<<<<<<< SEARCH
[the clause you are replacing, copied character for character from PARENT PROBLEM FAMILY]
=======
[its replacement]
>>>>>>> REPLACE

and nothing else in that field. The SEARCH text must occur verbatim in the
parent family. Everything outside it is preserved by the harness, so do not
write the rest of the family out.
""",

# The prose twin of local_diff. It exists so the PATCH MECHANISM is a measured
# contrast rather than an assumption: local_diff enforces the one-clause edit in
# the harness, local_diff_prose only asks for it. Smoke-measured on step160,
# 0 of 18 local_diff replies contained a SEARCH/REPLACE block at all -- the arm
# returns nothing, which is a finding about the policy, not a harness fault, and
# it is only legible next to a twin that does produce children.
"local_diff_prose": """
Change EXACTLY ONE clause of the parent family and nothing else. Name the clause
you are replacing and give its replacement. The rest of the statement -- its
objects, its bounds, and the quantity asked for -- must survive word for word.
""",

# The prose parameter arm. It is the MANIPULATION CHECK on the D1 fix: with the
# contradiction removed it must move to ~1.00. If it does not, the head swap did
# not take and no other arm on this table can be read.
"parameter_llm": """
Keep the parent's statement structure exactly. Change ONLY the numeric ranges
and constants its parameters are drawn from, so instances differ in size while
the mathematics and the solution route are unchanged. Reproducing the parent
family with different bounds is the CORRECT output here.
""",

# EoH M3 must be allowed to decline; a forced deletion is a second operator.
"simplify": """
Find every condition, object, or bound in the parent that the answer does NOT
depend on, and delete it. The child asks the same underlying question with the
redundant material removed. If nothing is redundant, reply
CHILD FAMILY: NONE
and stop. A forced deletion is not a simplification.
""",

"scale": """
Keep the parent's structure and its decisive reasoning move exactly. Change only
the SIZE of the object the question is about -- more elements, a larger index, a
higher dimension -- so the same argument has to be carried further.
""",

# Promptbreeder hypermutation drew the strength uniformly at random, making one
# arm a 1/3-1/3-1/3 MIXTURE of three operators at n~10 each. Split.
"adaptive_small": """
Change exactly one clause of the parent family. The solution route stays the
parent's.
""",
"adaptive_medium": """
Keep the parent's objects. Change the QUESTION asked about them, so the answer
is a different quantity.
""",
"adaptive_radical": """
Change both the objects the question is about and the decisive move of its
shortest clean solution. The mathematical domain stays {group} -- that is the
one thing you may not change.
""",

# ---- expansive arms -------------------------------------------------------
"structural": """
Keep the parent's mathematical domain and its kind of object. Change the
DECISIVE SOLUTION STRUCTURE: the move that the shortest clean solution turns on
must become a different move. Changing parameters, bounds, wording, the
direction of a predicate, or the size of the object does NOT count -- a solver
who knows the parent's method must be unable to finish your child with it.
""",

"full_rewrite": """
Do not edit the parent. Write a NEW family from scratch that belongs to the same
mathematical domain and asks a question the parent does not. The parent is shown
only so you know which domain to stay in; do not reuse its objects, its bounds,
or the shape of its question.
""",

# EoH M1 is "motivated by the parent's idea", not "hit a success band". 34 of
# the 48 parents sit at s_hat > 0.85, so the band form collapsed to one sentence
# for 71% of draws and asked for an unreachable one-hop move.
"directed": """
Name the DECISIVE REASONING MOVE the parent's shortest clean solution turns on.
Keep the parent's domain and the kind of object it is about. Motivated by that
move but not bound to it, write a child whose shortest clean solution needs the
parent's move PLUS one further step that a direct route cannot skip. Say which
move you kept and which step is new. Never mention difficulty, solver success,
or a target band in the CHILD FAMILY itself.
""",

# EVOTOOL blame attribution names the COMPONENT that failed. The shipped version
# fed a 3-way if/else on s_hat plus n_cell (the constant 1 for all 48 cells) and
# R_Q (0.000 for 30 of 48): two of its three "measurements" carried no
# information. This one is fed real per-seed rollout evidence measured by
# `evidence` and cached.
"feedback": """
This parent was measured on the solver that will be trained on its children.
Per-seed solve rates: {per_seed}. Overall {n_solved} of {n_roll} rollouts
solved; of the {n_failed} that failed, {n_wrong} returned a wrong integer and
{n_timeout} ran out of budget. The most common wrong answer was {modal_wrong}.

Name the ONE clause of the parent family that this evidence blames, quote it,
and rewrite only that clause. Leave every other clause character for character
unchanged. If the evidence blames no clause, say so and stop.
""",

# Promptbreeder EDA conditions on the POPULATION's distribution, not on one
# donor. One donor is constrained crossover -- the thing probe_crossover.py
# already failed at (40-47% of children copied one parent wholesale).
"donor": """
Below are {k} families sampled from across the archive. They are a picture of
what this population already contains, NOT material to copy.

{population}

Write a child of the PARENT. It keeps the parent's domain and its objects, and
its decisive reasoning move must be one that none of the {k} families above
uses. Name the move you chose and name the family it is most unlike.
Reproducing any of the {k}, or ignoring them, both fail.
""",
}

# Arms that never call the model. parameter_ast is the metric's upper anchor:
# it rewrites only the parent's integer sampling bounds, so a metric that does
# not read it at ~1.000 is not measuring structure. It sits outside the S1 x S2
# factorial because it has no stage 1 and no stage 2.
HARNESS_ARMS = ("parameter_ast",)

# GROUP is held at the parent's own value in every arm. The archive's group axis
# is inherited, not declared, and an arm that flips GROUP buys a trivially novel
# skeleton. Appended to the USER turn so the SYSTEM turn stays shipped machinery
# plus exactly one operator paragraph.
GROUP_HOLD = """
Hold GROUP fixed at the parent's own value: {group}. Choose SKILL freely from
the allowed list, as a DESCRIPTION of what your child actually demands -- never
as a target to write toward.
"""

# ---- S2 levels (D5) -------------------------------------------------------
# prod  the shipped gen_program_user_prompt.txt: parent template + parent SOURCE
#       as the worked example. The suspected copy vector.
# none  the child family alone; no reference generator at all.
# fixed a FIXED unrelated seed generator as the reference pair -- separates
#       "a reference generator is shown" from "the PARENT's generator is shown".
# show  probe_operators.py's own framing. Kept only as the bridge back to
#       rq_output/probe_operators.jsonl; not a candidate design.
S2_LEVELS = ("prod", "none", "fixed", "show")


# ===========================================================================
# Prompt construction
# ===========================================================================
def arm_system_prompt(arm: str, rotated_system: str, shots: str = "rotate") -> str:
    """Swap the shipped strength order for the arm's own. THIS IS THE D1 FIX.

    ``rotated_system`` is what build_family_task already produced, few-shot
    rotation included, so the rotation machinery is never reimplemented here --
    only the head is replaced and the tail's field descriptor retuned.

    RESIDUAL D1 LEAK, measured not assumed. Each shipped EXAMPLE block carries
    its own line of the form "STRUCTURAL MUTATION: the target changes from X to
    Y and the decisive reasoning changes to Z". That is a FOURTH copy of the
    structural order, and it cannot be removed without authoring a second
    example set, because the examples are also the only demonstration of the
    output format. It biases the conservative arms upward in novelty -- i.e.
    toward the control -- so it makes those arms look WORSE, not better, than
    they are. ``--shots none`` drops the blocks entirely (head + tail only) so
    the size of the leak is measurable on whichever arms it matters for, and
    ``shot_copy_max`` reports per child how much of an example came back.
    """
    head, blocks, tail = _split_family_system(rotated_system)
    if shots == "none":
        blocks = []
    if arm.startswith("ctl_"):
        if shots == "none":
            return head + tail             # shipped head verbatim, no examples
        return rotated_system              # shipped head verbatim, tail included
    body = "".join(_renumber(blocks, _EXAMPLE_HEAD, "EXAMPLE"))
    tail = _TAIL_DESC.sub(_TAIL_DESC_NEUTRAL, tail)
    if arm in OPS_TAIL:
        tail = _TAIL_FAMILY.sub(OPS_TAIL[arm].replace("\\", "\\\\"), tail)
    note = SHOTS_NOTE if blocks else ""
    return NEUTRAL_OPEN + WELLPOSED + note + "\n" + body + tail


def _prob(p: ProblemProgram) -> str:                    # kept from probe_operators.py
    try:
        i = p.execute(seed=0)
    except Exception:
        i = None
    return i.problem.strip() if i else "(parent did not run)"


def op_paragraph(arm: str, blk: dict, pool: list, rng: random.Random,
                 evidence: dict) -> str:
    """The arm's operator text with its per-block substitutions resolved."""
    text = OPS[arm]
    if not text.strip():
        return ""
    parent = pool[blk["parent"]]
    if arm == "adaptive_radical":
        text = text.format(group=parent["group"])
    elif arm == "donor":
        # k families sampled from ACROSS the archive, drawn from the block seed
        # so the donor set is identical for every arm that reads it.
        k = 4
        idx = [j for j in rng.sample(range(len(pool)), min(k + 1, len(pool)))
               if j != blk["parent"]][:k]
        pop = "\n\n".join(
            f"({n+1}) " + (extract_problem_template(pool[j]["src"]) or _prob(pool[j]["prog"]))
            for n, j in enumerate(idx))
        text = text.format(k=len(idx), population=pop)
    elif arm == "feedback":
        ev = evidence.get(parent["pid"])
        if ev is None:
            return ""                       # no evidence -> arm cannot run here
        text = text.format(
            per_seed=", ".join(f"{x:.2f}" for x in ev["per_seed"]),
            n_solved=ev["n_solved"], n_roll=ev["n_roll"],
            n_failed=ev["n_roll"] - ev["n_solved"],
            n_wrong=ev["n_wrong"], n_timeout=ev["n_timeout"],
            modal_wrong=ev["modal_wrong"])
    return "\nOPERATOR\n" + text.strip() + "\n"


def family_messages(arm: str, blk: dict, pool: list, evidence: dict,
                    inject_cell: bool, shots: str = "rotate"):
    """Stage-1 messages. Rotation and the shipped user turn come from prompts.py."""
    parent = pool[blk["parent"]]
    rng = random.Random(blk["seed"])            # D3: block-derived, arm-independent
    target = None
    if arm == "ctl_production" or inject_cell:
        # config.target_cell_injection: the run names a cell for the child to
        # land in. Drawn from the block seed so both controls and every arm see
        # the SAME cell when the flag is on.
        target = (parent["group"], SKILLS[random.Random(blk["seed"] + 7).randrange(len(SKILLS))])
    task = build_family_task(parent["prog"], rotate_shots=True,
                             rng=rng, target_cell=target)
    system = arm_system_prompt(arm, task.messages[0]["content"], shots)
    op = op_paragraph(arm, blk, pool, random.Random(blk["seed"] + 11), evidence)
    if arm == "feedback" and not op:
        return None
    user = task.messages[1]["content"] + "\n" + op + GROUP_HOLD.format(group=parent["group"])
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_FIXED_REF: ProblemProgram | None = None


def generator_messages(s2: str, parent: ProblemProgram, plan: dict, blk: dict):
    """Stage-2 messages at one S2 level. `prod` is the shipped build_generator_task."""
    global _FIXED_REF
    rng = random.Random(blk["seed"] + 23)       # same rotation for all four levels
    if s2 == "prod":
        return build_generator_task(parent, plan, rotate_shots=True, rng=rng).messages
    if s2 == "fixed":
        if _FIXED_REF is None:
            _FIXED_REF = ProblemProgram.from_file(
                sorted((ROOT / "seed_programs").glob("*.py"))[0])
        return build_generator_task(_FIXED_REF, plan, rotate_shots=True, rng=rng).messages
    # `none` and `show` need the shipped SYSTEM turn (rotation included) with a
    # different USER turn, so borrow the system turn build_generator_task made.
    sysmsg = build_generator_task(parent, plan, rotate_shots=True,
                                  rng=random.Random(blk["seed"] + 23)).messages[0]
    if s2 == "none":
        user = ("NOW WRITE THE GENERATOR FOR THIS FAMILY\n\n"
                f"{plan['CHILD FAMILY']}\n\nCORRECT OUTPUT:")
    else:  # show -- probe_operators.py's framing, kept as the bridge only
        user = (f"PARENT SOURCE\n\n```python\n"
                f"{strip_label_declarations(strip_module_docstring(parent.source_code))}\n```\n\n"
                f"NEW PROBLEM FAMILY TO IMPLEMENT\n\n{plan['CHILD FAMILY']}\n\n"
                f"GROUP: {plan['GROUP']}\nSKILL: {plan['SKILL']}\n\n"
                "Write the complete generator for the NEW PROBLEM FAMILY.")
    return [sysmsg, {"role": "user", "content": user}]


# ===========================================================================
# Model client -- kept from probe_operators.py, plus seed / top_p / finish_reason
# ===========================================================================
def chat(port, model, messages, max_tokens, temp=0.7, top_p=0.95, seed=None):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": temp, "top_p": top_p}
    if seed is not None:
        payload["seed"] = int(seed) % (2 ** 31)      # D3: common random numbers
    body = json.dumps(payload).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    last = ""
    for attempt in range(3):
        try:
            r = json.loads(urllib.request.urlopen(req, timeout=900).read())
            ch = r["choices"][0]
            return (ch["message"]["content"] or ""), (ch.get("finish_reason") or "")
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"
            time.sleep(1.0 + attempt)
    return "", last


def code_of(reply):                                    # kept from probe_operators.py
    m = re.findall(r"```python\n(.*?)```", reply, re.S)
    return max(m, key=len) if m else None


# ===========================================================================
# Arm-specific stage-1 post-processing
# ===========================================================================
_PATCH = re.compile(
    r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE", re.S)


def apply_patch(field: str, parent_family: str) -> tuple[str | None, str]:
    """local_diff: apply the SEARCH/REPLACE that arrived in the CHILD FAMILY field.

    Whitespace-tolerant. The parent family is an f-string lifted out of source,
    so it carries line breaks and indentation the model reflows when it quotes a
    clause back. Rejecting those would have measured the model's transcription,
    not its edit: in the smoke run every single local_diff reply failed on
    exactly that, and the arm returned zero children.
    """
    m = _PATCH.search(field or "")
    if not m:
        return None, "no patch block"
    search, repl = m.group(1).strip("\n").strip(), m.group(2).strip("\n").strip()
    if not search:
        return None, "empty SEARCH"
    if search in parent_family:
        return parent_family.replace(search, repl, 1), ""
    # Re-find the clause ignoring how whitespace was reflowed, then cut the
    # ORIGINAL span so the surrounding text is preserved byte for byte.
    flex = r"\s+".join(re.escape(tok) for tok in search.split())
    hit = re.search(flex, parent_family)
    if hit:
        return parent_family[:hit.start()] + repl + parent_family[hit.end():], ""
    return None, "SEARCH not found in parent"


_SAMPLERS = {"randint", "randrange"}


def parameter_ast(parent_source: str, rng: random.Random) -> str:
    """The harness parameter operator: widen integer sampling bounds, nothing else.

    From probe_operators_ops_replacement.py. No model call -- the metric anchor.
    """
    class _Widen(ast.NodeTransformer):
        def visit_Call(self, node):
            self.generic_visit(node)
            name = getattr(node.func, "attr", getattr(node.func, "id", ""))
            if name in _SAMPLERS:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
                        arg.value = max(1, int(arg.value * rng.uniform(1.2, 2.0)))
            return node
    tree = _Widen().visit(ast.parse(parent_source))
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


# ===========================================================================
# Metrics
# ===========================================================================
def skeleton(src: str) -> tuple[str, ...] | None:
    """archive.MAPElitesArchive.program_skeleton's walk, verbatim.

    Not probe_operators.py's NodeVisitor walk: generic_visit's traversal order
    differs, so its ratios were not commensurable with the 0.996 archive figure
    the whole probe exists to explain.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None
    seq: list[str] = []

    def walk(node):
        seq.append(type(node).__name__)
        for child in ast.iter_child_nodes(node):
            walk(child)
    walk(tree)
    return tuple(seq)


def sim(a, b) -> float:
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def _shot_families() -> list[str]:
    """The CHILD FAMILY line of each of the 8 shipped EXAMPLE blocks."""
    _, blocks, _ = _split_family_system(T("diff_problem_system_prompt.txt"))
    out = []
    for b in blocks:
        m = re.search(r"^CHILD FAMILY:\s*(.+?)\s*$", b, re.M)
        if m:
            out.append(m.group(1))
    return out


# ===========================================================================
# Parent pool -- kept from probe_operators.py's main()
# ===========================================================================
def load_pool(archive_dir: str):
    snaps = sorted((ROOT / archive_dir).glob("archive_iter*.json"),
                   key=lambda p: int("".join(c for c in p.name if c.isdigit())))
    if not snaps:
        raise SystemExit(f"no archive snapshots under {ROOT/archive_dir}")
    champs = json.loads(snaps[-1].read_text())["champions"]
    champs = list(champs.values() if isinstance(champs, dict) else champs)
    pool = []
    for c in champs:
        gi = c.get("niche_group")
        pool.append(dict(
            prog=ProblemProgram(source_code=c["source_code"]),
            group=GROUPS[gi] if isinstance(gi, int) and gi < len(GROUPS) else GROUPS[0],
            cell=(c.get("niche_group"), c.get("niche_skill")),
            s_hat=float(c.get("s_hat") or 0.0), rq=float(c.get("rq_score") or 0.0),
            src=c["source_code"],
            pid=hashlib.md5(c["source_code"].encode()).hexdigest()[:10]))
    # Deterministic order so a re-run with the same --seed reproduces the blocks.
    pool.sort(key=lambda p: p["pid"])
    return pool, snaps[-1]


def build_blocks(pool, n_blocks: int, seed: int):
    """D3. ONE block list, consumed by every arm in the same order.

    A block is (parent, replicate). Parents cycle so every parent is used
    ceil(n_blocks/len(pool)) times and the design stays balanced -- with 48
    parents, --blocks 384 is exactly 8 replicates of each, which is the R the
    power note in the module docstring lands on.

    The block's seed is derived from the BLOCK ONLY. That makes the few-shot
    rotation, the donor sample, the target cell and the vLLM sampler seed common
    random numbers across arms, so those nuisance factors are BLOCKED rather
    than merely randomised.
    """
    rng = random.Random(seed)
    order = list(range(len(pool)))
    blocks = []
    for b in range(n_blocks):
        if b % len(pool) == 0:
            rng.shuffle(order)              # a fresh permutation per replicate
        parent = order[b % len(pool)]
        # Derived from the BLOCK INDEX ONLY -- never from the arm. This is what
        # makes rotation / donor / target cell / sampler common random numbers.
        blocks.append({"b": b, "parent": parent, "rep": b // len(pool),
                       "pid": pool[parent]["pid"],
                       "seed": (seed * 1_000_003 + b * 7919) % (2 ** 31)})
    return blocks


# ===========================================================================
# `evidence` -- pre-measure the parents so `feedback` is fed real numbers
# ===========================================================================
async def _measure_parents(pool, args):
    from openai import AsyncOpenAI
    from rq_evolve.prompts import build_solver_messages
    from rq_evolve.reward import answers_match, extract_boxed
    from rq_evolve.solver_trace import SOLVER_CHAT_BOUNDARY_STOPS, sanitize_solver_trace

    client = AsyncOpenAI(base_url=f"http://127.0.0.1:{args.port}/v1",
                         api_key="none", timeout=1200.0, max_retries=0)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(problem):
        async with sem:
            try:
                r = await client.chat.completions.create(
                    model=args.model, messages=build_solver_messages(problem),
                    temperature=1.0, top_p=0.95, max_tokens=args.tokens,
                    stop=list(SOLVER_CHAT_BOUNDARY_STOPS))
                c = r.choices[0]
                return (c.message.content or ""), (c.finish_reason or "")
            except Exception:
                return "", "error"

    work = []
    for p in pool:
        for z in range(args.eval_seeds):
            inst = p["prog"].execute(seed=z)
            if inst is not None:
                work.append((p["pid"], inst))
    print(f"{len(work)} parent instances x {args.rollouts} rollouts "
          f"= {len(work)*args.rollouts} on :{args.port}", flush=True)
    t0 = time.time()
    outs = await asyncio.gather(*[one(i.problem) for _, i in work
                                  for _ in range(args.rollouts)])
    print(f"  done in {(time.time()-t0)/60:.1f} min", flush=True)

    ev: dict = collections.defaultdict(lambda: dict(
        per_seed=[], n_solved=0, n_roll=0, n_wrong=0, n_timeout=0,
        wrong=collections.Counter()))
    for i, (pid, inst) in enumerate(work):
        chunk = outs[i * args.rollouts:(i + 1) * args.rollouts]
        hits = 0
        e = ev[pid]
        for text, finish in chunk:
            e["n_roll"] += 1
            pred = extract_boxed(sanitize_solver_trace(text))
            if pred is not None and answers_match(pred, inst.answer):
                hits += 1
                e["n_solved"] += 1
            elif finish == "length":
                e["n_timeout"] += 1
            else:
                e["n_wrong"] += 1
                if pred is not None:
                    e["wrong"][str(pred)] += 1
        e["per_seed"].append(hits / max(len(chunk), 1))
    out = {}
    for pid, e in ev.items():
        out[pid] = {k: v for k, v in e.items() if k != "wrong"}
        out[pid]["modal_wrong"] = (e["wrong"].most_common(1)[0][0]
                                   if e["wrong"] else "(none recorded)")
    return out


def cmd_evidence(args):
    pool, snap = load_pool(args.archive)
    ev = asyncio.run(_measure_parents(pool, args))
    path = ROOT / args.evidence
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"snapshot": snap.name, "args": vars(args),
                                "evidence": ev}, indent=2))
    live = sum(1 for e in ev.values() if 0 < e["n_solved"] < e["n_roll"])
    print(f"{len(ev)} parents measured, {live} with 0 < s_hat < 1 -> {path}")
    return 0


# ===========================================================================
# `run` -- the factorial
# ===========================================================================
def cmd_run(args):
    pool, snap = load_pool(args.archive)
    blocks = build_blocks(pool, args.blocks, args.seed)
    arms = [a for a in (args.arms.split(",") if args.arms else list(OPS)) if a in OPS]
    s2s = [s for s in args.s2.split(",") if s in S2_LEVELS]
    evidence = {}
    epath = ROOT / args.evidence
    if epath.exists():
        evidence = json.loads(epath.read_text())["evidence"]
    elif "feedback" in arms:
        print(f"[warn] {epath} missing; dropping the `feedback` arm "
              f"(run `evidence` first)", flush=True)
        arms = [a for a in arms if a != "feedback"]

    print(f"archive {snap.name} | {len(pool)} parents | {len(blocks)} blocks "
          f"({args.blocks//len(pool)} replicates) | {len(arms)} arms x {len(s2s)} S2 "
          f"= {len(arms)*len(blocks)} stage-1 and up to "
          f"{len(arms)*len(blocks)*len(s2s)} stage-2 calls", flush=True)

    # ---- stage 1: ONE plan per (arm, block); shared down the S2 columns (D5)
    def do1(job):
        arm, blk = job
        rec = {"arm": arm, "b": blk["b"], "rep": blk["rep"], "pid": blk["pid"],
               "parent_idx": blk["parent"]}
        msgs = family_messages(arm, blk, pool, evidence,
                               args.inject_target_cell, args.shots)
        if msgs is None:
            rec["stage1"] = "no_evidence"; return rec, None
        reply, finish = chat(args.port, args.model, msgs, args.tokens1,
                             args.temp1, args.top_p, blk["seed"])
        rec["s1_finish"] = finish
        parent_family = (extract_problem_template(pool[blk["parent"]]["src"])
                         or _prob(pool[blk["parent"]]["prog"]))
        if re.search(r"^CHILD FAMILY:\s*NONE\s*$", reply, re.M):
            # simplify is allowed to decline. Counted, not silently dropped: an
            # operator that declines is not producing children, and that is a
            # property of the operator.
            rec["stage1"] = "declined"; return rec, None
        plan = parse_family_plan(reply)
        if plan is None:
            rec["stage1"] = "unparsed"; return rec, None
        if arm == "local_diff":
            # The patch arrives INSIDE the CHILD FAMILY field (parse_family_plan
            # reads a field to the next header, so the multi-line block survives)
            # and the harness -- not the model -- applies it. That is what makes
            # "change exactly one clause" a mechanism instead of a promise.
            child, why = apply_patch(plan["CHILD FAMILY"], parent_family)
            if child is None:
                rec["stage1"] = f"patch_failed: {why}"; return rec, None
            rec["patch"] = plan["CHILD FAMILY"]
            plan["CHILD FAMILY"] = child
        rec["stage1"] = "ok"
        rec["child_family"] = plan["CHILD FAMILY"]
        rec["plan_group"], rec["plan_skill"] = plan["GROUP"], plan["SKILL"]
        rec["mutation_note"] = plan.get("STRUCTURAL MUTATION", "")
        return rec, plan

    jobs1 = [(arm, blk) for arm in arms for blk in blocks]
    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        out1 = list(ex.map(do1, jobs1))
    n_ok = sum(1 for r, p in out1 if p is not None)
    print(f"stage 1: {n_ok}/{len(jobs1)} plans in {(time.time()-t0)/60:.1f} min",
          flush=True)

    # ---- stage 2: every surviving plan under every S2 level
    plans = {(r["arm"], r["b"]): (r, p) for r, p in out1 if p is not None}

    def do2(job):
        (arm, b), s2 = job
        rec, plan = plans[(arm, b)]
        blk = blocks[b]
        parent = pool[blk["parent"]]
        reply, finish = chat(args.port, args.model,
                             generator_messages(s2, parent["prog"], plan, blk),
                             args.tokens2, args.temp2, args.top_p, blk["seed"])
        row = dict(rec)
        row["s2"] = s2
        row["s2_finish"] = finish
        code = code_of(reply)
        row["stage2"] = "ok" if code else "no_code"
        if code:
            # evolution.py:878 appends the stage-1 labels; auditing the raw code
            # fails the label gate on 100% of children for a reason the run does
            # not have (probe_operator_gates.py documents this).
            row["code"] = set_label_declarations(code, row["plan_group"],
                                                 row["plan_skill"])
        return row

    jobs2 = [(k, s2) for k in plans for s2 in s2s]
    t0 = time.time()
    with ThreadPoolExecutor(args.concurrency) as ex:
        rows = list(ex.map(do2, jobs2))
    print(f"stage 2: {sum(1 for r in rows if r.get('code'))}/{len(jobs2)} with code "
          f"in {(time.time()-t0)/60:.1f} min", flush=True)

    # rows for plans that never reached stage 2, so the denominator is honest
    for (r, p) in out1:
        if p is None:
            for s2 in s2s:
                rows.append(dict(r, s2=s2, stage2="no_plan"))

    # ---- the harness arms: no model call, one row per block per S2 level
    for arm in HARNESS_ARMS:
        if args.arms and arm not in args.arms.split(","):
            continue
        for blk in blocks:
            parent = pool[blk["parent"]]
            src = parent["src"]
            try:
                code = parameter_ast(strip_label_declarations(strip_module_docstring(src)),
                                     random.Random(blk["seed"]))
            except Exception:
                code = None
            skill = ProblemProgram(source_code=src).declared_skill() or SKILLS[0]
            base = {"arm": arm, "b": blk["b"], "rep": blk["rep"], "pid": blk["pid"],
                    "parent_idx": blk["parent"], "stage1": "ok",
                    "child_family": extract_problem_template(src) or "",
                    "plan_group": parent["group"], "plan_skill": skill}
            for s2 in s2s:
                row = dict(base, s2=s2, stage2="ok" if code else "no_code")
                if code:
                    row["code"] = set_label_declarations(code, base["plan_group"], skill)
                rows.append(row)

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    print(f"{len(rows)} rows -> {out}", flush=True)
    return cmd_score(args)


# ===========================================================================
# `score` -- D4: executability first, then every statistic
# ===========================================================================
def cmd_score(args):
    rows = [json.loads(l) for l in (ROOT / args.out).read_text().splitlines() if l.strip()]
    pool, snap = load_pool(args.archive)
    by_pid = {p["pid"]: p for p in pool}

    arch = MAPElitesArchive()
    arch.load(snap)
    champs = [n.champion for n in arch.grid.values() if n.champion is not None]
    arch_skel = [s for s in (skeleton(c.source_code) for c in champs) if s]
    arch_route = [r for r in (route_signature(c.source_code) for c in champs) if r]
    shots = _shot_families()

    for r in rows:
        r["exec_ok"] = False
        code = r.get("code")
        if not code:
            continue
        parent = by_pid[r["pid"]]
        prog = ProblemProgram(source_code=code, metadata={"op": "mutation"})

        # --- D4. EXECUTABILITY FIRST. Seeds 0-2, >= 2 distinct texts. Nothing
        # below this line is computed for a child that does not clear it.
        try:
            insts = [prog.execute(seed=z) for z in range(3)]
        except Exception:
            insts = [None]
        if any(i is None for i in insts):
            r["died"] = "executes"; continue
        if len({" ".join(i.problem.split()) for i in insts}) < 2:
            r["died"] = "seed_variation"; continue
        r["exec_ok"] = True

        # --- similarity, on executables only.
        # The child always carries a GROUP/SKILL pair (the run appends it), so
        # the parent it is measured against must carry one too. Without this the
        # two extra Assign nodes cost every child ~0.017 of skeleton similarity
        # and parameter_ast -- the metric's own upper anchor, which must read
        # 1.000 -- came back at 0.983.
        pstrip = set_label_declarations(
            strip_label_declarations(strip_module_docstring(parent["src"])),
            parent["prog"].declared_group() or "", parent["prog"].declared_skill() or "")
        cs, ps = skeleton(code), skeleton(pstrip)
        r["skel_parent"] = sim(cs, ps) if cs and ps else None
        r["skel_archive"] = (max((sim(cs, a) for a in arch_skel), default=0.0)
                             if cs else None)
        cr, pr = route_signature(code), route_signature(pstrip)
        r["route_parent"] = route_sim(cr, pr)
        r["route_archive"] = max((route_sim(cr, a) for a in arch_route), default=0.0)
        r["src_parent"] = sim(code, pstrip)
        ptpl = extract_problem_template(parent["src"]) or ""
        r["family_parent"] = sim(r.get("child_family", ""), ptpl) if ptpl else None
        r["shot_copy_max"] = max((sim(r.get("child_family", ""), s) for s in shots),
                                 default=0.0)
        r["skel_hash"] = hashlib.md5(("|".join(cs)).encode()).hexdigest()[:12] if cs else None
        r["route_hash"] = (hashlib.md5("|".join(sorted(cr)).encode()).hexdigest()[:12]
                           if cr else None)

        # --- the three new gates, named individually
        r["gate_bare_draw"] = answer_is_bare_draw(code) is None
        r["gate_no_leak"] = answer_leaks_in_every_instance(
            [prog.execute(seed=z) for z in range(5)]) is None
        sdup = arch._find_structural_duplicate(prog)
        r["gate_not_structural_dup"] = sdup is None
        r["structural_dup_ratio"] = round(sdup[1], 3) if sdup else None

        # --- the rest of the production admission stack, in run order
        errs = list(lint_generator_source(code)) + list(lint_mutation_generator_source(
            code, require_assert=False, reject_trivial_assert=False,
            reject_unbounded_sampling=False, require_answer_routes=False,
            require_canonical_instance_data=False, require_mechanical_shape=False))
        if errs:
            r["died"] = "lint"; continue
        if check_generator_contract(code):
            r["died"] = "ast_contract"; continue
        if validate_label_decl(prog.declared_group(), prog.declared_skill()):
            r["died"] = "labels"; continue
        if not r["gate_bare_draw"]:
            r["died"] = "answer_is_bare_draw"; continue
        if not r["gate_no_leak"]:
            r["died"] = "answer_leaks_every_instance"; continue
        if (arch._find_duplicate_behavior(prog) or arch._find_duplicate_template(prog)
                or arch._find_near_duplicate_template(prog)):
            r["died"] = "template_duplicate"; continue
        if not r["gate_not_structural_dup"]:
            r["died"] = "structural_duplicate"; continue
        r["died"] = None                       # clears the whole production stack

    scored = ROOT / (args.out.replace(".jsonl", "_scored.jsonl"))
    scored.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # ---------------- tables ----------------
    s2s = sorted({r["s2"] for r in rows}, key=lambda s: S2_LEVELS.index(s))
    arms = sorted({r["arm"] for r in rows},
                  key=lambda a: (list(OPS) + list(HARNESS_ARMS)).index(a)
                  if a in list(OPS) + list(HARNESS_ARMS) else 99)
    med = lambda v: statistics.median(v) if v else float("nan")

    for s2 in s2s:
        print(f"\n=== S2 = {s2} " + "=" * 74)
        hdr = (f"{'arm':<17}{'n':>4}{'s1':>4}{'s2':>4}{'exec':>5}{'gate':>5}"
               f"{'skelP':>7}{'routeP':>8}{'srcP':>7}{'famP':>7}"
               f"{'skelA':>7}{'routeA':>8}{'shot':>6}"
               f"{'bare':>6}{'leak':>6}{'sdup':>6}{'uSkel':>7}{'uRoute':>7}")
        print(hdr); print("-" * len(hdr))
        for arm in arms:
            g = [r for r in rows if r["arm"] == arm and r["s2"] == s2]
            if not g:
                continue
            ok1 = [r for r in g if r.get("stage1") == "ok"]
            ok2 = [r for r in g if r.get("code")]
            ex = [r for r in g if r.get("exec_ok")]
            gate = [r for r in ex if r.get("died") is None]
            pct = lambda v: (100 * sum(v) / len(v)) if v else float("nan")
            print(f"{arm:<17}{len(g):>4}{len(ok1):>4}{len(ok2):>4}{len(ex):>5}{len(gate):>5}"
                  f"{med([r['skel_parent'] for r in ex if r.get('skel_parent') is not None]):>7.3f}"
                  f"{med([r['route_parent'] for r in ex]):>8.3f}"
                  f"{med([r['src_parent'] for r in ex]):>7.3f}"
                  f"{med([r['family_parent'] for r in ex if r.get('family_parent') is not None]):>7.3f}"
                  f"{med([r['skel_archive'] for r in ex if r.get('skel_archive') is not None]):>7.3f}"
                  f"{med([r['route_archive'] for r in ex]):>8.3f}"
                  f"{med([r['shot_copy_max'] for r in ex]):>6.2f}"
                  f"{pct([not r['gate_bare_draw'] for r in ex]):>5.0f}%"
                  f"{pct([not r['gate_no_leak'] for r in ex]):>5.0f}%"
                  f"{pct([not r['gate_not_structural_dup'] for r in ex]):>5.0f}%"
                  f"{len({r['skel_hash'] for r in ex}):>4}/{len(ex):<2}"
                  f"{len({r['route_hash'] for r in ex}):>4}/{len(ex):<2}")

    print("\n  skelP/routeP/srcP/famP = median similarity to the PARENT (lower = more novel)")
    print("  skelA/routeA           = to the NEAREST archive champion")
    print("  shot                   = max similarity of CHILD FAMILY to one of the 8 shipped EXAMPLEs")
    print("  bare/leak/sdup         = % of EXECUTABLE children failing each of the three new gates")
    print("  uSkel/uRoute           = distinct skeletons / route signatures among executables")
    print("  gate                   = children clearing the FULL production admission stack")

    print("\ndeaths (executable children only):")
    for c, k in collections.Counter(
            r.get("died") for r in rows if r.get("exec_ok") and r.get("died")).most_common():
        print(f"  {c:<28}{k:>5}")
    print("pre-execution losses:")
    for c, k in collections.Counter(
            (r.get("stage1") if r.get("stage1") != "ok" else r.get("stage2"))
            for r in rows if not r.get("exec_ok")).most_common():
        print(f"  {c:<28}{k:>5}")

    _contrasts(rows, arms, s2s, args)
    print(f"\nwritten to {scored}")
    return 0


def _contrasts(rows, arms, s2s, args):
    """D3's payoff: the PAIRED contrast of each arm against the control.

    PAIRED AT THE PARENT, not at the block. Pairing at the block -- (parent,
    replicate) -- is the tighter design but it needs BOTH arms to produce an
    executable child on the SAME block, and executability runs about 0.25, so
    complete blocks arrive at 0.25^2 = 6%. Measured on the smoke run: 12 blocks
    yielded 1-3 usable pairs per arm. Pairing on the parent instead compares the
    arm's MEAN over its executable children at that parent against the control's,
    so a parent contributes whenever each arm cleared execution at least ONCE in
    its R replicates: at R = 8 that is 1 - 0.75^8 = 90% per arm and 81% jointly.
    Same removal of the parent effect (which is the 53% of variance D3 is about),
    an order of magnitude more of the data kept.

    Block-level pairs are still printed underneath, because when they are
    plentiful they are the stronger test and the two should agree.
    """
    base = args.control if args.control in arms else arms[0]
    for s2 in s2s:
        cell = collections.defaultdict(list)
        blk = {}
        for r in rows:
            if r["s2"] != s2 or not r.get("exec_ok"):
                continue
            v = r.get(args.endpoint)
            if v is None:
                continue
            cell[(r["arm"], r["pid"])].append(v)
            blk[(r["arm"], r["b"])] = v

        print(f"\n=== paired contrasts vs `{base}`, endpoint={args.endpoint}, S2={s2} ===")
        print(f"{'arm':<18}{'parents':>8}{'cover':>7}{'medDiff':>9}{'95% CI':>18}"
              f"{'p':>9}{'blkPr':>7}{'blkMed':>8}")
        n_parents = len({p for (_, p) in cell})
        for arm in arms:
            if arm == base:
                continue
            d = [statistics.fmean(cell[(arm, p)]) - statistics.fmean(cell[(base, p)])
                 for (a, p) in list(cell) if a == arm and (base, p) in cell]
            db = [blk[(arm, b)] - blk[(base, b)]
                  for (a, b) in list(blk) if a == arm and (base, b) in blk]
            if len(d) < 5:
                print(f"{arm:<18}{len(d):>8}   too few paired parents")
                continue
            lo, hi = _boot_ci(d, args.seed)
            bmed = statistics.median(db) if db else float("nan")
            print(f"{arm:<18}{len(d):>8}{len(d)/max(n_parents,1):>7.0%}"
                  f"{statistics.median(d):>9.3f}{f'[{lo:+.3f},{hi:+.3f}]':>18}"
                  f"{_wilcoxon_p(d):>9.4f}{len(db):>7}{bmed:>8.3f}")
    print("\n  medDiff < 0 means the arm produces children LESS similar to the parent")
    print("  than the control does. p is a two-sided Wilcoxon signed-rank on the")
    print("  PARENT differences; compare against alpha/(#arms-1) for Dunnett-style")
    print("  control of the family-wise error rate. blkPr/blkMed are the stricter")
    print("  block-level pairs; they should agree in sign once blkPr is large.")


def _boot_ci(d, seed, iters=4000):
    rng = random.Random(seed)
    n = len(d)
    meds = sorted(statistics.median([d[rng.randrange(n)] for _ in range(n)])
                  for _ in range(iters))
    return meds[int(0.025 * iters)], meds[int(0.975 * iters)]


def _wilcoxon_p(d):
    try:
        from scipy.stats import wilcoxon
        return float(wilcoxon(d, zero_method="wilcox",
                              alternative="two-sided").pvalue)
    except Exception:
        pass
    nz = [x for x in d if x != 0]
    if not nz:
        return float("nan")
    ranks = {v: i + 1 for i, v in enumerate(sorted(nz, key=abs))}
    w = sum(ranks[v] for v in nz if v > 0)
    n = len(nz)
    mu, sd = n * (n + 1) / 4, math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu) / sd if sd else 0.0
    return 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))


# ===========================================================================
# `dump` -- hand the survivors to calibrate_seeds.py
# ===========================================================================
def cmd_dump(args):
    """Write survivors as .py files for calibrate_seeds.py, IN BATCHES.

    calibrate_seeds.py builds one asyncio.gather over every rollout with no
    semaphore and a 600 s client timeout, against httpx's default 100-connection
    pool. Handing it 280 programs at 4x4 is 4480 coroutines; all but ~100 sit in
    the pool, and at the measured 0.45-0.9 rollouts/s most of them pass 600 s
    before they are ever sent and raise httpx.ConnectTimeout -- which is exactly
    how the throughput measurement for this design died. So the dump is split
    into batches sized to finish inside that timeout, and each batch is a
    separate calibrate_seeds.py invocation.
    """
    path = ROOT / args.out.replace(".jsonl", "_scored.jsonl")
    if not path.exists():
        raise SystemExit(f"{path} missing -- run `score` first")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    pick = [r for r in rows
            if r.get("exec_ok") and (args.all_exec or r.get("died") is None)
            and (not args.dump_s2 or r["s2"] == args.dump_s2)]
    by_arm = collections.defaultdict(list)
    for r in pick:
        by_arm[(r["arm"], r["s2"])].append(r)

    chosen = []
    for (arm, s2), g in sorted(by_arm.items()):
        rng = random.Random(args.seed)
        rng.shuffle(g)                       # a random, not a lucky, subsample
        chosen.extend((arm, s2, r) for r in g[:args.limit_per_arm])

    per = max(1, args.batch_size)
    n_batch = max(1, math.ceil(len(chosen) / per))
    base = ROOT / args.dump_dir
    print(f"{len(chosen)} of {len(pick)} eligible "
          f"({'executable' if args.all_exec else 'full-stack survivors'}"
          f"{', S2=' + args.dump_s2 if args.dump_s2 else ''}) "
          f"-> {n_batch} batch dir(s) under {base}")
    for b in range(n_batch):
        out = base / f"batch{b:02d}"
        out.mkdir(parents=True, exist_ok=True)
        for f in out.glob("*.py"):
            f.unlink()
        for arm, s2, r in chosen[b * per:(b + 1) * per]:
            (out / f"{arm}__{s2}__b{r['b']:04d}.py").write_text(
                r["code"], encoding="utf-8")

    roll = args.eval_seeds * args.rollouts
    print(f"\n  # {per} programs x {args.eval_seeds} seeds x {args.rollouts} "
          f"rollouts = {per*roll} rollouts per server per batch")
    print(f"  # measured: 4B :8401 ~0.9 rollouts/s, 8B :8801 ~0.45 -> "
          f"~{per*roll/0.45/60:.0f} min per batch, 8B-bound")
    for b in range(n_batch):
        print(f"  python scripts/calibrate_seeds.py "
              f"--seed-dir {base/f'batch{b:02d}'} \\\n"
              f"      --eval-seeds {args.eval_seeds} --rollouts {args.rollouts} "
              f"--tokens 5000 \\\n"
              f"      --out rq_output/probe_paired_difficulty_b{b:02d}.json")
    print(f"\n  total: {len(chosen)*roll} rollouts on EACH of :8401 and :8801 "
          f"(~{len(chosen)*roll/0.45/3600:.1f} h wall, both servers in one loop)")
    return 0


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["run", "score", "evidence", "dump"])
    ap.add_argument("--blocks", type=int, default=192,
                    help="(parent, replicate) pairs per arm. 192 = 4 replicates "
                         "of all 48 parents. See the power note in the docstring.")
    ap.add_argument("--arms", default=None, help="comma list; default all")
    ap.add_argument("--s2", default="prod,none",
                    help=f"comma list from {S2_LEVELS}")
    ap.add_argument("--control", default="ctl_shipped")
    ap.add_argument("--endpoint", default="skel_parent",
                    choices=["skel_parent", "route_parent", "src_parent",
                             "skel_archive", "route_archive", "family_parent"])
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--model", default="step160")
    ap.add_argument("--archive", default="rq_output/rq_evolve_4b_4gpu/rq_archive")
    ap.add_argument("--out", default="rq_output/probe_operators_paired.jsonl")
    ap.add_argument("--evidence", default="rq_output/probe_paired_evidence.json")
    ap.add_argument("--dump-dir", default="rq_output/probe_paired_survivors")
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--concurrency", type=int, default=48)
    ap.add_argument("--tokens1", type=int, default=2048)
    ap.add_argument("--tokens2", type=int, default=3000)
    ap.add_argument("--temp1", type=float, default=0.7)
    ap.add_argument("--temp2", type=float, default=0.7)
    ap.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    ap.add_argument("--shots", default="rotate", choices=["rotate", "none"],
                    help="stage-1 EXAMPLE blocks. `rotate` is the shipped "
                         "setting (a random 3 of 8). `none` drops them, which "
                         "is how the residual D1 leak in the examples' own "
                         "STRUCTURAL MUTATION lines is measured.")
    ap.add_argument("--inject-target-cell", action="store_true",
                    help="turn on config.target_cell_injection for EVERY arm, "
                         "not just ctl_production")
    # evidence / dump
    ap.add_argument("--eval-seeds", type=int, default=5)
    ap.add_argument("--rollouts", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=5000)
    ap.add_argument("--limit-per-arm", type=int, default=20,
                    help="survivors sampled per (arm, S2) for the difficulty pass")
    ap.add_argument("--dump-s2", default="prod",
                    help="restrict the difficulty pass to one S2 level; \"\" for all")
    ap.add_argument("--all-exec", action="store_true",
                    help="dump every EXECUTABLE child, not only full-stack "
                         "survivors. Use when the gate yield (~3-11%%) leaves an "
                         "arm with too few children to estimate s_hat from.")
    ap.add_argument("--batch-size", type=int, default=70,
                    help="programs per calibrate_seeds.py invocation. 70 x 4 x 4 "
                         "= 1120 rollouts ~ 21 min on the 8B, inside its 600 s "
                         "per-request client timeout.")
    args = ap.parse_args()
    return {"run": cmd_run, "score": cmd_score,
            "evidence": cmd_evidence, "dump": cmd_dump}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
