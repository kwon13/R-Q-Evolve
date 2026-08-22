import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

from .code_utils import strip_label_declarations, strip_module_docstring
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


MUTATION_SYSTEM_PROMPT_FILE = "mutation_system_prompt.txt"
MUTATION_USER_PROMPT_FILE = "mutation_user_prompt.txt"
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


@lru_cache(maxsize=1)
def mutation_system_prompt() -> str:
    """The Evolver's system turn, read verbatim from the template directory."""
    return _load_template(MUTATION_SYSTEM_PROMPT_FILE).strip()


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


def build_mutation_task(
    parent: ProblemProgram,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    """One mutation request: the parent program plus both label vocabularies.

    There is no operator argument. The Evolver decides for itself what to change
    and labels the finished child from its own shortest solution; nothing here
    demands a particular GROUP or SKILL.
    """
    system_prompt = mutation_system_prompt()
    user_prompt = _render_template(
        _load_template(MUTATION_USER_PROMPT_FILE),
        _template_context(parent),
    )
    return MutationTask(
        op=MUTATION_OP,
        prompt=f"{system_prompt}\n\n{user_prompt}",  # flat fallback only
        parent=parent,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        top_p=top_p,
    )


def build_fix_task(
    task: MutationTask,
    failed_output: str,
    reason: str,
) -> MutationTask:
    system_prompt = mutation_system_prompt()
    if task.messages:
        original_user = str(task.messages[-1].get("content", ""))
    else:
        original_user = task.prompt
        if original_user.startswith(system_prompt):
            original_user = original_user[len(system_prompt):].lstrip("\n")

    fix_request = (
        "Your program above was REJECTED by the validator.\n"
        f"Rejection reason(s): {reason or 'unspecified'}\n"
        "Fix ONLY the issue(s) above while keeping the mathematical idea intact. "
        "Output the corrected full program in one ```python ``` block."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": original_user},
        {"role": "assistant", "content": failed_output},
        {"role": "user", "content": fix_request},
    ]
    return MutationTask(
        op=task.op,
        prompt=f"{failed_output}\n\n{fix_request}",  # flat fallback only
        parent=task.parent,
        messages=messages,
        stage=task.stage,
        max_output_tokens=task.max_output_tokens,
        temperature=task.temperature,
        top_p=task.top_p,
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
