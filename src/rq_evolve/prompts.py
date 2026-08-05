import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from string import Template

from .code_utils import strip_module_docstring
from .concepts import GROUPS, SKILLS
from .program import ProblemProgram

SOLVER_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

groups = ", ".join(GROUPS)
skills = ", ".join(SKILLS)

MUTATION_CODE_SKELETON = """
Required structural skeleton (replace every <...> placeholder; this is not a
mathematical example and must not be returned with placeholders):
```python
import random

GROUP = "<one of the GROUP vocabulary>"
SKILL = "<one of the SKILL vocabulary>"

def generate(seed):
    rng = random.Random(seed)
    <draw parameters from ranges that admit no degenerate case>
    answer = <the intended route to the integer answer>
    check = <the same quantity, recomputed from what the problem text says>
    assert answer == check, f"answer={answer} check={check}"
    problem = <one question about that one integer>
    return problem, str(answer)
```
"""

# The one contract item nothing else in the pipeline can replace. verify_program
# checks the answer is an integer and that the statement varies; the evaluator
# checks the statement reads coherently. Neither checks that problem_text and
# answer describe the SAME mathematics -- a sample run produced "arrange 4
# distinct objects in 9 positions" answered with 4**9 instead of P(9,4), and it
# passed every gate. An independent recomputation is the only mechanical way to
# catch that, so the prompt shows how to build one rather than merely demanding it.
MUTATION_ASSERT_RULE = """
The assert is the only check that problem_text and answer describe the same
mathematics. Build it like this:

```python
    answer = <the intended route>
    check = <the same quantity, recomputed from the words of problem_text>
    assert answer == check, f"answer={answer} check={check}"
```

Give the assert that message. When it fires you are shown the failure and get
one chance to repair the program, and "AssertionError" with no values does not
say which of the two routes was wrong.

`check` must come from a genuinely different procedure -- counting the stated
objects one by one, a complement, a closed form against a loop, a small-case
enumeration. Repeating the first route's expression, or asserting a property of
`answer` alone, checks nothing. Keep `check` cheap: seeds 0-4 each run under a
few seconds.
"""


# One file per axis, each the single source for that axis's meanings. The
# operator that HOLDS an axis has the parent as a concrete instance of it and
# needs no definition; the operator that MOVES an axis is crossing into a value
# it has never seen an instance of, so it gets that axis's file:
#   in_depth   holds GROUP, moves SKILL -> skill_definitions
#   in_breadth holds SKILL, moves GROUP -> group_definitions
# The evaluator reads skill_definitions too, to judge a declared label against
# the same text the mutation was written from.
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


@lru_cache(maxsize=1)
def mutation_system_prompt() -> str:
    """Role, output discipline, code skeleton, then the SKILL_CRITERIA block.

    The per-operator templates own the numbered instructions and the axis
    definitions each operator needs. This carries only what neither says: who
    the model is, that exactly one code block comes back, the shape to fill in,
    and how to build the self-check.

    Built on demand rather than at import: the criteria block is read from
    ``PROMPT_TEMPLATE_DIR``, which honours ``RQ_EVOLVE_PROMPT_DIR`` and is
    defined further down this module.
    """
    return (
        "You design one executable Python program for competition-math "
        "problems. Each file defines `generate(seed)`, which returns one "
        "(problem_text, answer) pair, and then labels what it produced on two "
        "independent axes.\n"
        "\n"
        "Think silently. Return exactly one complete program in one ```python``` "
        "block and no other text. Never output analysis, JSON, a chat-message "
        "object, role/content wrappers, example labels, the parent program, or "
        "any few-shot example. Generate a new program for the LIVE parent only.\n"
        "\n"
        f"GROUP must be exactly one of: {groups}\n"
        f"SKILL must be exactly one of: {skills}\n"
        "The two axes are independent: any GROUP may pair with any SKILL. "
        "GROUP names the mathematical domain; SKILL names the reasoning the "
        "visible problem forces a solver to perform.\n"
        "\n"
        f"{MUTATION_CODE_SKELETON}\n"
        f"{MUTATION_ASSERT_RULE}\n"
        "Before emitting code, silently verify: exactly one Python block; seeds "
        "0 through 4 terminate; `check` recomputes the answer from the problem "
        "text rather than repeating the first route; the problem is not the "
        "parent's problem reworded; and the declared SKILL is the reasoning the "
        "visible problem actually forces. If any check fails, redesign before "
        "answering."
    )

EVALUATOR_SYSTEM_PROMPT = (
    "You are an evaluator for generated math word problems and their generator "
    "programs.\n"
    "Determine whether the problem is internally coherent, whether the supplied "
    "answer solves the visible problem, and—when source is supplied—whether the "
    "code computes exactly the mathematics stated in problem_text.\n\n"
    "Recompute from the literal values printed in the visible problem; do not "
    "trust source comments, intended formulas, or the supplied answer. If source "
    "overwrites a list, matrix, graph, sequence, bound, or coefficient object, "
    "mark INVALID when problem_text is formatted from stale pre-overwrite aliases "
    "while the answer is computed from the updated object.\n\n"
    "Mark the problem as INVALID if any stated condition, theorem, system, recurrence, optimization, or variable definition is not logically connected to the final question (even if the answer can still be computed by ignoring it), if the statement combines two or more independent problems or poses multiple unrelated final questions, if the same variable name is reused for unrelated objects in an ambiguous way, or if the final requested answer does not follow from the stated problem; otherwise, check for contradictory conditions, irrelevant conditions, inapplicable claims about solution methods, and extraneous assumptions.\n"
    "Also mark INVALID if the returned answer is wrong, the code uses hidden "
    "quantities or a different mathematical object than the problem states, "
    "the code's declared answer checks are internally inconsistent, bounded "
    "sampling cannot terminate, or the program copies an example instead of mutating the live "
    "parent.\n"
    "A declared SKILL is a claim about what the problem forces a solver to do, "
    "and it is the claim most often false: a label is free to write, the "
    "reasoning is not. Judge it against the SKILL definition supplied with the "
    "candidate, reading only the visible problem -- never the source comments, "
    "the constant, or what the generator seems to have intended. Answer "
    "skill_required: NO whenever the problem can be solved without that "
    "reasoning, and mark the verdict INVALID with it.\n"
    "Return:\n"
    "- reason: concise explanation\n"
    "- skill_required: YES or NO\n"
    "- verdict: VALID or INVALID"
)

EVALUATOR_SHOT_FILE = "evaluator.txt"

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

# The two operators each hold one axis fixed and move the other.
#
#   in_depth   -- operator A: GROUP fixed, SKILL must change.
#                 Same mathematical domain, different decisive reasoning.
#   in_breadth -- operator B: SKILL fixed, GROUP must change.
#                 Same decisive reasoning, transported to another domain.
#
# The op names are the pre-migration ones, kept so `evolution.in_depth_ratio`,
# the sampler and every archived report string still line up. "depth" now means
# "stays in the domain", "breadth" means "leaves the domain".
PROMPT_TEMPLATE_FILES = {
    "in_depth": "mutate_A_skill_within_group.txt",
    "in_breadth": "mutate_B_group_within_skill.txt",
}
# Live mutation substitutes $few_shot_examples with the empty string, so no shot
# file is read today. The mapping stays name-parallel for the day it is re-enabled;
# the old shots/in_*.txt fixtures are written against the retired CONCEPT_* labels
# and must be regenerated before any of them is injected again.
SHOT_TEMPLATE_FILES = PROMPT_TEMPLATE_FILES


def build_mutation_task(
    op: str,
    parent: ProblemProgram,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    if op not in PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")

    template = _load_prompt_template(op)
    # Mutation runs WITHOUT few-shot examples, deliberately. Two reasons, both
    # measured: the verified example pairs cost +8,047 tokens, which leaves 872
    # of the 12,000-token rollout window for the response (a child generator is
    # 2-4k); and code_temperature is 0.0, the greedy decoding that made the
    # model copy an example instead of mutating the live parent. The structural
    # lesson those pairs carry is in MUTATION_CODE_SKELETON and
    # MUTATION_ASSERT_RULE instead, at ~150 tokens. Restoring injection means
    # raising rollout.max_model_len first -- see tests/fixtures/mutation_pairs/.
    context = _template_context(op=op, parent=parent)
    live_user = _render_template(template, {**context, "few_shot_examples": ""})
    system_prompt = mutation_system_prompt()

    return MutationTask(
        op=op,
        prompt=f"{system_prompt}\n\n{live_user}",
        parent=parent,
        # Both representations omit code-rich few-shots; greedy decoding was
        # copying them instead of mutating the live parent.
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": live_user},
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


def _load_evaluator_shots() -> str:
    path = SHOT_TEMPLATE_DIR / EVALUATOR_SHOT_FILE
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def skill_definition(skill: str | None) -> str:
    """The one SKILL_CRITERIA line for ``skill``, or "" when it is unknown.

    The evaluator cannot judge a label it has no definition for, so the line
    travels with the candidate instead of the evaluator being expected to
    remember all eight.
    """
    if not skill:
        return ""
    for line in _load_definitions(SKILL_DEFINITIONS_FILE).splitlines():
        if line.startswith(f"- {skill}:"):
            return line.strip()
    return ""


def build_evaluator_messages(
    problem_text: str,
    *,
    answer_text: str | None = None,
    program_source: str | None = None,
    skill: str | None = None,
) -> list[dict]:
    """Render the semantic generator review conversation for one problem."""
    blocks: list[str] = []
    shots = _load_evaluator_shots()
    if shots:
        blocks.append(shots)
    review = (
        "Now evaluate the following problem.\n\n"
        f"Problem:\n{problem_text.strip()}\n"
    )
    if answer_text is not None:
        review += f"\nGenerated answer:\n{str(answer_text).strip()}\n"
    if program_source is not None:
        review += (
            "\nGenerator source:\n```python\n"
            + str(program_source).strip()
            + "\n```\n"
        )
    definition = skill_definition(skill)
    if definition:
        review += (
            f"\nThe generator declares SKILL = {skill!r}, defined as:\n"
            f"{definition}\n"
            "Decide from the visible problem alone whether that reasoning is "
            "genuinely required. Output skill_required: YES only if it is.\n"
        )
    blocks.append(f"{review}\nAnswer:")
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def build_evaluator_task(
    program: ProblemProgram,
    problem_text: str,
    *,
    answer_text: str | None = None,
    program_source: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    """Wrap an evaluator query as a MutationTask so ``backend.mutate`` can run it.

    Reuses the existing batched generate path (mutate reads ``messages``); no new
    backend method is needed. ``parent`` carries the program under review purely
    for reporting -- mutate only consumes ``messages``/``prompt``.
    """
    messages = build_evaluator_messages(
        problem_text,
        answer_text=answer_text,
        program_source=program_source,
        skill=program.get_skill(),
    )
    flat = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return MutationTask(
        op="evaluate",
        prompt=flat,
        parent=program,
        messages=messages,
        stage="evaluate",
        temperature=temperature,
        top_p=top_p,
    )


def parse_evaluator_verdict(
    output: str,
    *,
    require_skill: bool = False,
) -> tuple[bool, str]:
    """Parse an evaluator response into ``(is_valid, reason)``.

    A candidate passes ONLY on an explicit VALID verdict. Field labels are read
    leniently (case, bullets, emphasis markers) because a base model varies the
    spelling of the label far more than the judgement; field VALUES stay strict.
    Falls back to scanning the whole text for a verdict. Anything else -- INVALID,
    no verdict at all, empty, or unreadable -- is NOT valid and is discarded, so
    only problems the evaluator clearly endorses reach the archive. ``INVALID``
    is checked before ``VALID`` because it contains ``VALID`` as a substring.
    """
    text = output or ""
    # Tolerate the shapes a base model actually emits around the field name --
    # "Reason:", "- reason:", "**verdict**:", a leading bullet or number. Of 160
    # captured off-format replies, the judgement was usually right and only the
    # spelling of the label was wrong, so an exact-prefix match discarded good
    # verdicts. The field VALUES stay strict.
    def field(name: str) -> str:
        m = re.search(
            r"^[\s>*_#-]*" + name + r"[\s*_]*:\s*(.+)$",
            text, re.IGNORECASE | re.MULTILINE,
        )
        return m.group(1).strip() if m else ""

    reason = field("reason")
    verdict = field("verdict").upper()
    skill_required = field("skill_required").upper()

    if not verdict:
        upper = text.upper()
        if "INVALID" in upper:
            verdict = "INVALID"
        elif "VALID" in upper:
            verdict = "VALID"
    is_valid = verdict.startswith("VALID")  # INVALID / missing / off-format -> discard
    if require_skill and not skill_required.startswith("YES"):
        # A silent or negative answer both mean the same thing: nothing
        # established that the declared SKILL is the reasoning this problem
        # forces, which is exactly the claim that goes unchecked otherwise.
        #
        # Silence carries the raw output. In an earlier run 11 of 17 evaluator
        # rejections were silent, and this branch overwrote `reason` before the
        # text fallback below could run -- so 19% of a batch was discarded on a
        # signal there was no way to audit. Verdict unchanged; the evidence now
        # survives.
        is_valid = False
        if skill_required:
            reason = "declared SKILL is not required by the visible problem" + (
                "; " + reason if reason else ""
            )
        else:
            reason = (
                "evaluator gave no skill_required line; raw output: "
                + (text.strip()[:220] or "(empty)")
            )
    if not reason:
        reason = text.strip()[:300] or "no explicit VALID verdict"
    return is_valid, reason


def _load_prompt_template(op: str) -> str:
    path = PROMPT_TEMPLATE_DIR / PROMPT_TEMPLATE_FILES[op]
    if not path.exists():
        raise FileNotFoundError(f"missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def _load_shot_examples(op: str) -> str:
    path = SHOT_TEMPLATE_DIR / SHOT_TEMPLATE_FILES[op]
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"Few-shot examples:\n\n{text}"


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


def _template_context(
    op: str,
    parent: ProblemProgram,
) -> dict[str, str]:
    parent_group = str(parent.get_group() or "")
    parent_skill = str(parent.get_skill() or "")
    return {
        "few_shot_examples": _load_shot_examples(op),
        "parent_id": parent.program_id,
        "parent_generation": str(parent.generation),
        "parent_source": strip_module_docstring(parent.source_code),
        "parent_group": parent_group,
        "parent_skill": parent_skill,
        # Operator A holds GROUP and must land on a different SKILL; operator B
        # holds SKILL and must land on a different GROUP. Each list is its axis
        # minus the parent's own value, so the forbidden option is never offered.
        "allowed_skills": ", ".join(
            skill for skill in SKILLS if skill != parent_skill
        ),
        "allowed_groups": ", ".join(
            group for group in GROUPS if group != parent_group
        ),
        # Each template references only its own; the renderer ignores the
        # unused key, and both stay on one source file.
        "skill_definitions": _load_definitions(SKILL_DEFINITIONS_FILE),
        "group_definitions": _load_definitions(GROUP_DEFINITIONS_FILE),
        "parent_p_hat": f"{float(getattr(parent, 'p_hat', 0.0) or 0.0):.3f}",
        "parent_h_score": f"{float(getattr(parent, 'h_score', 0.0) or 0.0):.3f}",
        "parent_rq_score": f"{float(getattr(parent, 'rq_score', 0.0) or 0.0):.6f}",
    }


def build_solver_prompt(problem: str) -> str:
    return f"{SOLVER_SYSTEM_PROMPT}\n\nProblem: {problem}\n\n"
