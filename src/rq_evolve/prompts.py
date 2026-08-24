import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

from .code_utils import (
    extract_problem_template,
    strip_label_declarations,
    strip_module_docstring,
)
from .concepts import GROUPS, SKILLS
from .program import ProblemProgram

SOLVER_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


SKILL_DEFINITIONS_FILE = "skill_definitions.txt"
GROUP_DEFINITIONS_FILE = "group_definitions.txt"


def _load_definitions(filename: str) -> str:
    path = PROMPT_TEMPLATE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"missing axis definitions: {path}. Without them the model picks "
            "labels from a bare vocabulary list with nothing saying what they "
            "mean, and a mislabel becomes unfalsifiable."
        )
    return path.read_text(encoding="utf-8").strip()


JUDGE_SYSTEM_PROMPT_FILE = "mutation_judge_system_prompt.txt"
JUDGE_USER_PROMPT_FILE = "mutation_judge_user_prompt.txt"

# One mutation operator. The pair that forced a label change on one axis --
# in_depth held GROUP and moved SKILL, in_breadth the mirror -- is retired.
# Ordering the model to land on a SKILL it had not yet solved for made the
# label a target instead of a description, and the child was written to satisfy
# the order: the archived GROUP/SKILL then disagreed with what the visible
# problem actually demanded, which is the one error the MAP cannot survive.
# The child now picks both labels from its own finished mathematics, and the
# judge re-derives them from the visible problem alone.
MUTATION_OP = "mutate"


@lru_cache(maxsize=4)
def judge_system_prompt(rubric_file: str = JUDGE_SYSTEM_PROMPT_FILE) -> str:
    """The judge's system turn: the validity gate and taxonomy rubric.

    ``rubric_file`` is a parameter rather than a constant so a rubric can be
    swapped without touching code -- ``evolution.judge_rubric`` selects it and
    ``scripts/compare_judges.py`` uses it to put two rubrics on one corpus.

    The shipped rubric keeps two validity gates and then judges both axes
    against the same one-line definitions the Evolver reads. An earlier draft
    additionally required a SKILL to survive a routineness test, a mandatory
    named witness and a closest-alternative challenge; measured over 41 items
    that took both-axis agreement from 41% to 2% and returned ``SKILL: none``
    for 6 of 8 hand-labelled seeds, so a "declared == judged" gate built on it
    rejected ground truth. See analysis/judge_pipeline_v2/.
    """
    return _load_template(rubric_file).strip()


@dataclass(slots=True)
class MutationTask:
    op: str
    prompt: str
    parent: ProblemProgram
    # When set, the backend renders this full chat conversation as the prompt
    # (multi-turn self-fix) instead of wrapping ``prompt`` as a single user msg.
    messages: list[dict] | None = None
    stage: str = "code"
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompt_templates"
PROMPT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_PROMPT_DIR", DEFAULT_PROMPT_TEMPLATE_DIR)
)
SHOT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_SHOT_DIR", PROMPT_TEMPLATE_DIR / "shots")
)


JUDGE_FIELDS = (
    "GROUP",
    "GROUP_EVIDENCE",
    "SKILL",
    "SKILL_WITNESS",
    "CLOSEST_ALTERNATIVE",
    "WHY_NOT_ALTERNATIVE",
    "FAILURE_REASON",
)


@dataclass(slots=True)
class JudgeVerdict:
    """One parsed judge reply.

    ``group`` and ``skill`` are None whenever the judge failed closed (it emits
    the literal ``none``) or whenever the field was missing, unparseable, or
    outside the vocabulary. None is the answer, not an error: every one of those
    shapes means the judge did not certify a label, and they must all reject.
    """

    group: str | None = None
    skill: str | None = None
    group_evidence: str = ""
    skill_witness: str = ""
    closest_alternative: str = ""
    why_not_alternative: str = ""
    failure_reason: str = ""
    raw: str = ""

    def to_dict(self) -> dict[str, str | None]:
        return {
            "group": self.group,
            "skill": self.skill,
            "group_evidence": self.group_evidence,
            "skill_witness": self.skill_witness,
            "closest_alternative": self.closest_alternative,
            "why_not_alternative": self.why_not_alternative,
            "failure_reason": self.failure_reason,
        }


def build_judge_messages(
    problem_text: str,
    answer_text: str,
    *,
    rubric_file: str = JUDGE_SYSTEM_PROMPT_FILE,
) -> list[dict]:
    """The judge conversation for one (problem, answer) pair.

    The generator source, the declared labels, and the parent never travel with
    it. The judge has to reach GROUP and SKILL from the visible problem alone,
    which is the only way its answer can disagree with the declared one.
    """
    user = _render_template(
        _load_template(JUDGE_USER_PROMPT_FILE),
        {
            "problem_text": str(problem_text).strip(),
            "answer": str(answer_text).strip(),
        },
    )
    return [
        {"role": "system", "content": judge_system_prompt(rubric_file)},
        {"role": "user", "content": user},
    ]


def build_judge_task(
    program: ProblemProgram,
    problem_text: str,
    answer_text: str,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
    rubric_file: str = JUDGE_SYSTEM_PROMPT_FILE,
) -> MutationTask:
    """Wrap a judge query as a MutationTask so ``backend.mutate`` can run it.

    Reuses the batched generate path; no new backend method is needed.
    ``parent`` carries the program under review purely for reporting.
    """
    messages = build_judge_messages(
        problem_text, answer_text, rubric_file=rubric_file
    )
    return MutationTask(
        op="judge",
        prompt=f"{messages[0]['content']}\n\n{messages[1]['content']}",
        parent=program,
        messages=messages,
        stage="judge",
        temperature=temperature,
        top_p=top_p,
    )


def _judge_field(text: str, name: str) -> str:
    """Read one ``NAME: value`` line, tolerating decoration around the label.

    The output contract asks for seven bare lines, but a base model reliably
    wraps a field name in bullets, bold markers, or numbering while getting the
    value right. The label match is lenient for that reason; the VALUES are
    then held to the closed vocabularies below, so leniency here can only
    recover a well-formed verdict, never invent one.
    """
    match = re.search(
        r"^[\s>*_#-]*" + name + r"[\s*_]*:[ \t]*(.*)$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def _judge_label(value: str, vocabulary: tuple[str, ...]) -> str | None:
    """A vocabulary member, or None for ``none``/missing/anything unrecognised."""
    # Strip whitespace and decoration together: a base model writes
    # "**GROUP:** number_theory", which leaves "** number_theory" as the value,
    # and stripping punctuation alone would hand back a leading space.
    token = value.strip(" \t`'\"*_.:#").lower()
    if not token or token == "none":
        return None
    return token if token in vocabulary else None


def parse_judge_verdict(output: str) -> JudgeVerdict:
    """Parse a judge reply into its seven fields.

    Never raises. An empty, truncated, or off-contract reply parses to a verdict
    with ``group``/``skill`` None, which ``judge_accepts`` rejects -- the judge
    is a gate that has to fail closed, so an unreadable answer must not be
    distinguishable from a refusal.
    """
    text = output or ""
    return JudgeVerdict(
        group=_judge_label(_judge_field(text, "GROUP"), GROUPS),
        skill=_judge_label(_judge_field(text, "SKILL"), SKILLS),
        group_evidence=_judge_field(text, "GROUP_EVIDENCE"),
        skill_witness=_judge_field(text, "SKILL_WITNESS"),
        closest_alternative=_judge_field(text, "CLOSEST_ALTERNATIVE"),
        why_not_alternative=_judge_field(text, "WHY_NOT_ALTERNATIVE"),
        failure_reason=_judge_field(text, "FAILURE_REASON"),
        raw=text,
    )


def judge_accepts(
    verdict: JudgeVerdict,
    declared_group: str | None,
    declared_skill: str | None,
) -> tuple[bool, str]:
    """The gate: both labels must survive the judge AND match what was declared.

    Agreement on both axes is the whole condition. A child the judge labels
    validly but differently is still rejected: the archive would file it under
    the declared cell while the problem belongs in another, and a MAP whose
    coordinates lie is worse than a MAP with an empty cell.
    """
    if verdict.group is None or verdict.skill is None:
        reason = verdict.failure_reason.strip()
        missing = " and ".join(
            axis
            for axis, value in (("GROUP", verdict.group), ("SKILL", verdict.skill))
            if value is None
        )
        if not reason or reason.lower() == "none":
            reason = f"judge returned no {missing}"
        return False, f"judge failed closed ({missing}): {reason}"

    mismatches = []
    if verdict.group != declared_group:
        mismatches.append(f"GROUP declared={declared_group!r} judged={verdict.group!r}")
    if verdict.skill != declared_skill:
        mismatches.append(f"SKILL declared={declared_skill!r} judged={verdict.skill!r}")
    if mismatches:
        return False, "label mismatch: " + "; ".join(mismatches)
    return True, ""


def _load_template(filename: str) -> str:
    path = PROMPT_TEMPLATE_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _template_identifiers(template: str) -> set[str]:
    """Names referenced by a ``$``-placeholder template.

    ``Template.get_identifiers`` only exists on Python 3.11+, and the training
    env runs 3.10, so walk the class's own pattern instead.
    """
    identifiers: set[str] = set()
    for match in Template.pattern.finditer(template):
        name = match.group("named") or match.group("braced")
        if name is not None:
            identifiers.add(name)
    return identifiers


def _render_template(template: str, context: dict[str, str]) -> str:
    """Substitute every placeholder, refusing to leave one unresolved.

    ``safe_substitute`` would ship a literal ``$parent_skill`` inside the
    prompt, which reads as a plausible instruction and fails silently. The
    check runs on the template's own identifiers, before substitution, so a
    ``$`` appearing inside the substituted parent source is not mistaken for
    an unresolved placeholder.
    """
    missing = sorted(_template_identifiers(template) - set(context))
    if missing:
        raise KeyError(
            "prompt template references placeholders the context does not "
            f"supply: {', '.join(missing)}"
        )
    return Template(template).safe_substitute(context)


def _template_context(parent: ProblemProgram) -> dict[str, str]:
    """Placeholders for ``mutation_user_prompt.txt``.

    Both vocabularies go in whole. The retired operators each withheld the axis
    they held fixed, on the reasoning that the parent was already a worked
    instance of it; with no axis held there is nothing to withhold, and a label
    the child may choose but cannot read the meaning of is a label it will
    misapply.
    """
    return {
        # The parent's cell is withheld: from the prose, and from the tail of
        # its own source. With the real labels shown, 97% of 118 distinct
        # children declared the cell their parent already occupied, across only
        # 12 distinct cells. What replaces the ending the tail used to teach is
        # PART 1 committing to GROUP and SKILL before any code is written.
        "parent_source": strip_label_declarations(
            strip_module_docstring(parent.source_code)
        ),
        "parent_problem": _parent_problem_text(parent),
        "allowed_groups": _load_definitions(GROUP_DEFINITIONS_FILE),
        "allowed_skills": _load_definitions(SKILL_DEFINITIONS_FILE),
    }


def _parent_problem_text(parent: ProblemProgram) -> str:
    """The parent's seed-0 statement, or a note saying it could not be run.

    The prompt used to show the source alone, which framed the task as editing
    a program. It is not: the object being mutated is a MATHEMATICAL PROBLEM,
    and the program is only the machine that emits it. A model shown code
    rewrites code -- measured, the largest single failure of distinct children
    was the child's own cross-check firing, i.e. mathematics committed to before
    it was worked out. Showing the statement lets the child be designed as a
    problem first.

    Never raises: the parent is an archived champion that ran at insertion time,
    but a resume can load a snapshot whose source no longer executes here, and a
    prompt is not the place to discover that.
    """
    try:
        instance = parent.execute(seed=0)
    except Exception:  # pragma: no cover - execute already swallows most of these
        instance = None
    if instance is None:
        return "(the parent program did not run here; read its code below instead)"
    return instance.problem.strip()


def build_solver_messages(problem: str) -> list[dict]:
    """The solver conversation: rules in the system turn, problem in the user turn.

    Every other measurement path in the codebase -- ``dataset.py``,
    ``math_eval.py``, the expansion experiments, the standalone vLLM eval --
    sends ``SOLVER_SYSTEM_PROMPT`` as a system turn. The rollout path used to
    concatenate it into the user turn instead, so the policy was trained on one
    prompt shape and scored on another.

    ``shot_internal.build_solver_messages`` is the same conversation with an
    optional worked example spliced in, and its no-shot branch returns exactly
    what this returns. The two stay in step because both read the one
    ``SOLVER_SYSTEM_PROMPT`` above; this is the production rollout/eval builder,
    that one belongs to the shot diagnostics.
    """
    return [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]


_PART1_LABEL = r"^[ \t]*{key}[ \t]*:[ \t]*[\"']?([A-Za-z_]+)"


def parse_declared_labels(reply: str) -> tuple[str | None, str | None]:
    """Read GROUP / SKILL off a reply's PART 1 prose.

    The mutation contract puts the two labels in PART 1 and ends the python
    block at ``return problem, str(answer)``. That is deliberate: a label line
    sitting after ``return`` is past the strongest completion boundary in the
    file and was simply dropped -- 15 of 24 children on one probe -- while a
    label decided in PART 1, with the solution still in view, is written as
    part of the reasoning. The cost is that the extracted program has no labels
    of its own, so they have to be put back from here.

    Returns whatever is recognisable; validation of the pair belongs to
    :func:`rq_evolve.concepts.validate_label_decl`, which the caller already
    runs.
    """
    text = reply or ""
    out: list[str | None] = []
    for key, vocabulary in (("GROUP", GROUPS), ("SKILL", SKILLS)):
        found = None
        for match in re.finditer(_PART1_LABEL.format(key=key), text, re.M):
            value = match.group(1)
            if value in vocabulary:
                found = value  # last wins: a reply that restates them means it
        out.append(found)      # settled late, and PART 2 must not carry any
    return out[0], out[1]


_INFERRED_LABEL = r"^[ \t>*_#-]*INFERRED_{key}[\s*_]*:[ \t]*(.*)$"


def parse_inferred_labels(reply: str) -> tuple[str | None, str | None]:
    """Read INFERRED_GROUP / INFERRED_SKILL off a stage-2 generator reply.

    Stage 2 is asked to write the labels AFTER the code block, using the
    ``INFERRED_`` prefix so they are distinct from any ``GROUP = "..."`` /
    ``SKILL = "..."`` assignment lines in the code itself. The labels are a
    blind re-derivation: stage 2 never sees stage 1's plan labels, so
    agreement between the two is genuine cross-verification.

    Returns ``(group, skill)`` where each is a vocabulary member or None.
    """
    text = reply or ""
    out: list[str | None] = []
    for key, vocabulary in (("GROUP", GROUPS), ("SKILL", SKILLS)):
        found = None
        for match in re.finditer(_INFERRED_LABEL.format(key=key), text, re.IGNORECASE | re.MULTILINE):
            raw_val = match.group(1).strip()
            # Clean decoration: backticks, quotes, asterisks, whitespace
            token = raw_val.strip(" \t`'\"*_.:#").lower()
            token = token.split()[0] if token.split() else ""
            token = token.strip(" \t`'\"*_.:#,")
            if token in vocabulary:
                found = token  # last wins, same convention as parse_declared
        out.append(found)
    return out[0], out[1]



# --- two-stage mutation -----------------------------------------------------
#
# Stage 1 writes the child PROBLEM in prose, with no program in front of it;
# stage 2 writes that problem's generator, with the parent pair shown only as a
# worked example of the statement-to-program mapping. Measured against asking
# one call to mutate the parent program: child/parent source similarity fell
# from 0.99 to 0.14, and the share of children declaring their parent's own
# cell fell from 96% to 0%. Both follow from the same thing -- a base policy
# shown a complete program to "mutate" reproduces it, and a base policy shown a
# program it must not reproduce (because the target statement is already fixed
# and different) does not.

FAMILY_SYSTEM_PROMPT_FILE = "diff_problem_system_prompt.txt"
FAMILY_USER_PROMPT_FILE = "diff_problem_user_prompt.txt"
GENERATOR_SYSTEM_PROMPT_FILE = "gen_program_system_prompt.txt"
GENERATOR_USER_PROMPT_FILE = "gen_program_user_prompt.txt"

# What a stage-1 reply must carry for the child to be usable.
FAMILY_KEYS = ("STRUCTURAL MUTATION", "CHILD FAMILY", "GROUP", "SKILL")
# Asked for, parsed when present, but NOT required. WHY FINITE makes the model
# name the clause that bounds its own answer before it labels the problem --
# 31% of archived champions were judged ill-posed, nearly all of them "find the
# maximum X" with the clause that made X finite dropped somewhere in the
# mutation. Requiring it would be the wrong trade: stage-1 parse failures are
# already the largest single loss (230 mutation_failed of 2144 candidates), and
# a fifth mandatory field buys more of them. The value is in writing it, not in
# gating on it.
OPTIONAL_FAMILY_KEYS = ("WHY FINITE",)
# EVERY header the reply may contain, required or not. The lookahead that ends
# one field has to know all of them: a header missing from this list is not a
# boundary, so its whole line is swallowed into the previous field's value --
# which would have quietly appended the finiteness prose to CHILD FAMILY.
_FAMILY_KEY_ALT = "|".join(FAMILY_KEYS + OPTIONAL_FAMILY_KEYS)


def parse_family_plan(reply: str) -> dict[str, str] | None:
    """The stage-1 fields, or None if a REQUIRED one is missing.

    Takes the LAST value for each key that is not a leftover ``<...>``. A base
    policy re-emits the template before answering it -- 13 of 24 replies opened
    with the placeholder lines and only then wrote the real four -- so reading
    the first match reports a complete reply as incomplete.
    """
    text = reply or ""
    out: dict[str, str] = {}
    for key in FAMILY_KEYS:
        best = ""
        for match in re.finditer(
            rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_FAMILY_KEY_ALT})[ \t]*:|\Z)",
            text, re.M | re.S,
        ):
            value = match.group(1).strip()
            if value and not value.startswith("<"):
                best = value
        if not best:
            return None
        out[key] = best
    group = _first_token(out["GROUP"], GROUPS)
    skill = _first_token(out["SKILL"], SKILLS)
    if group is None or skill is None:
        return None
    out["GROUP"], out["SKILL"] = group, skill
    for key in OPTIONAL_FAMILY_KEYS:
        for match in re.finditer(
            rf"^[ \t]*{key}[ \t]*:[ \t]*(.+?)(?=^[ \t]*(?:{_FAMILY_KEY_ALT})[ \t]*:|\Z)",
            text, re.M | re.S,
        ):
            value = match.group(1).strip()
            if value and not value.startswith("<"):
                out[key] = value
    return out


def _first_token(value: str, vocabulary) -> str | None:
    token = value.strip().strip("\"'").split()
    token = token[0].strip(".,") if token else ""
    return token if token in vocabulary else None


def build_family_task(
    parent: ProblemProgram,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    """Stage 1: mutate the parent's problem FAMILY, in prose, with no program.

    The parent goes in as its ``problem = ...`` f-string rather than as one
    rendered instance. Shown concrete numbers a model changes the numbers --
    85-89% of children were near-copies of the statement they were given, one
    reporting its own mutation as "a different prime p and a different range".
    Against the braces, substituting values is visibly not a change.
    """
    instance = _parent_problem_text(parent)
    template = extract_problem_template(parent.source_code) or instance
    system_prompt = _load_template(FAMILY_SYSTEM_PROMPT_FILE)
    user_prompt = _render_template(
        _load_template(FAMILY_USER_PROMPT_FILE),
        {
            "parent_template": template,
            "parent_problem": instance,
            "allowed_groups": _load_definitions(GROUP_DEFINITIONS_FILE),
            "allowed_skills": _load_definitions(SKILL_DEFINITIONS_FILE),
        },
    )
    return MutationTask(
        op=MUTATION_OP,
        prompt=f"{system_prompt}\n\n{user_prompt}",
        parent=parent,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stage="family",
        temperature=temperature,
        top_p=top_p,
    )


def build_generator_task(
    parent: ProblemProgram,
    plan: dict[str, str],
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    """Stage 2: write the generator for the fixed child family.

    The parent's label lines are stripped from what the model sees. They would
    only be copied -- every redaction tried (real labels, ``"..."``, the
    skeleton placeholder, deletion) produced children whose tail was whatever
    the parent's tail contained. The labels come from stage 1 and the caller
    appends them, so they cannot be dropped and cannot disagree with the
    problem they describe.

    After the code block, the model is asked to output ``INFERRED_SKILL`` --
    a blind re-derivation of the skill label from the code it just wrote.
    The caller compares this against the stage-1 plan and rejects the child
    when they disagree, catching implementation drift where the code is
    easier to write than the problem is to solve.
    """
    system_prompt = _render_template(
        _load_template(GENERATOR_SYSTEM_PROMPT_FILE),
        {
            "skill_definitions": _load_definitions(SKILL_DEFINITIONS_FILE),
        },
    )
    user_prompt = _render_template(
        _load_template(GENERATOR_USER_PROMPT_FILE),
        {
            "parent_template": extract_problem_template(parent.source_code)
            or _parent_problem_text(parent),
            "parent_source": strip_label_declarations(
                strip_module_docstring(parent.source_code)
            ),
            "new_problem": plan["CHILD FAMILY"],
        },
    )
    return MutationTask(
        op=MUTATION_OP,
        prompt=f"{system_prompt}\n\n{user_prompt}",
        parent=parent,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        stage="generator",
        temperature=temperature,
        top_p=top_p,
    )

