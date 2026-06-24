import os
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .code_utils import strip_module_docstring
from .program import ProblemProgram

SOLVER_SYSTEM_PROMPT = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

CONCEPT_GROUPS: tuple[str, ...] = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
)

groups = ", ".join(CONCEPT_GROUPS)

MUTATION_SYSTEM_PROMPT = (
    "You design a Python program for competition-math problems. Each file defines `generate(seed)`, which returns one (problem_text, answer) pair, and then labels what it produced.\n"
    "\n"
    "Output structure, in this order:\n"
    "  1. an optional module docstring — the mutation idea and how the resulting problem is solved;\n"
    "  2. imports (only collections, fractions, functools, itertools, math, random, sympy);\n"
    "  3. `def generate(seed)`;\n"
    "  4. the constants CONCEPT_REASON, CONCEPT_GROUP, CONCEPT_TYPE, in that order.\n"
    "\n"
    f"CONCEPT_GROUP must be exactly one of: {groups}\n"
    "CONCEPT_TYPE is a free-form '<group>.<snake_case_name>' string.\n"
    "Fill the three constants:\n"
    "  - CONCEPT_REASON: It must describe the core mathematical reasoning the solver performs based strictly on the problem text. If the problem reduces to a simple operation, state it plainly rather than overcomplicating the label."
    "\n"
    "  - CONCEPT_GROUP and CONCEPT_TYPE: name the reasoning that CONCEPT_REASON describes."
    "\n"
    "  - The problem text must never reveal the answer's value — no \"... and the result is 17\", no \"simplify A/B = 2002\". The solver computes it.\n"
    "\n"
    "Please reason step by step, and put your final program within ```python ```"
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are an evaluator for math word problems.\n"
    "Your task is to determine whether the problem statement itself is internally coherent.\n\n"
    "Mark the problem as INVALID if any stated condition, theorem, system, recurrence, optimization, or variable definition is not logically connected to the final question (even if the answer can still be computed by ignoring it), if the statement combines two or more independent problems or poses multiple unrelated final questions, if the same variable name is reused for unrelated objects in an ambiguous way, or if the final requested answer does not follow from the stated problem; otherwise, check for contradictory conditions, irrelevant conditions, inapplicable claims about solution methods, and extraneous assumptions.\n"
    "Return:\n"
    "- reason: concise explanation\n"
    "- verdict: VALID or INVALID"
)

EVALUATOR_SHOT_FILE = "evaluator.txt"

@dataclass(slots=True)
class MutationTask:
    op: str
    prompt: str
    parent: ProblemProgram
    parent_b: ProblemProgram | None = None
    # When set, the backend renders this full chat conversation as the prompt
    # (multi-turn self-fix) instead of wrapping ``prompt`` as a single user msg.
    messages: list[dict] | None = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_TEMPLATE_DIR = PROJECT_ROOT / "prompt_templates"
PROMPT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_PROMPT_DIR", DEFAULT_PROMPT_TEMPLATE_DIR)
)
SHOT_TEMPLATE_DIR = Path(
    os.environ.get("RQ_EVOLVE_SHOT_DIR", PROMPT_TEMPLATE_DIR / "shots")
)

PROMPT_TEMPLATE_FILES = {
    "in_depth": "in_depth.txt",
    "in_breadth": "in_breadth.txt",
    "crossover": "crossover.txt",
}
SHOT_TEMPLATE_FILES = PROMPT_TEMPLATE_FILES


def build_mutation_task(
    op: str,
    parent: ProblemProgram,
    parent_b: ProblemProgram | None = None,
) -> MutationTask:
    if op not in PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")
    if op == "crossover" and parent_b is None:
        raise ValueError("crossover requires parent_b")

    template = _load_prompt_template(op)
    user = Template(template).safe_substitute(
        _template_context(op=op, parent=parent, parent_b=parent_b)
    )

    return MutationTask(
        op=op,
        prompt=f"{MUTATION_SYSTEM_PROMPT}\n\n{user}",
        parent=parent,
        parent_b=parent_b,
    )


def build_fix_task(
    task: MutationTask,
    failed_output: str,
    reason: str,
) -> MutationTask:
    original_user = task.prompt
    if original_user.startswith(MUTATION_SYSTEM_PROMPT):
        original_user = original_user[len(MUTATION_SYSTEM_PROMPT):].lstrip("\n")

    fix_request = (
        "Your program above was REJECTED by the validator.\n"
        f"Rejection reason(s): {reason or 'unspecified'}\n"
        "Fix ONLY the issue(s) above while keeping the mathematical idea intact. "
        "Output the corrected full program in one ```python ``` block."
    )
    messages = [
        {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": original_user},
        {"role": "assistant", "content": failed_output},
        {"role": "user", "content": fix_request},
    ]
    return MutationTask(
        op=task.op,
        prompt=f"{failed_output}\n\n{fix_request}",  # flat fallback only
        parent=task.parent,
        parent_b=task.parent_b,
        messages=messages,
    )


def _load_evaluator_shots() -> str:
    path = SHOT_TEMPLATE_DIR / EVALUATOR_SHOT_FILE
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_evaluator_messages(problem_text: str) -> list[dict]:
    """Render the coherence-check conversation for one problem.

    The shot file demonstrates the ``Problem: ... Answer: reason: ... verdict:``
    format, so the user turn presents only the problem text and stops at
    ``Answer:`` for the model to continue. The math answer is deliberately
    omitted: the evaluator judges the *statement's* internal coherence, not
    whether a number is correct.
    """
    shots = _load_evaluator_shots()
    blocks: list[str] = []
    if shots:
        blocks.append(shots)
    blocks.append(
        "Now evaluate the following problem.\n\n"
        f"Problem:\n{problem_text.strip()}\n\n"
        "Answer:"
    )
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def build_evaluator_task(program: ProblemProgram, problem_text: str) -> MutationTask:
    """Wrap an evaluator query as a MutationTask so ``backend.mutate`` can run it.

    Reuses the existing batched generate path (mutate reads ``messages``); no new
    backend method is needed. ``parent`` carries the program under review purely
    for reporting -- mutate only consumes ``messages``/``prompt``.
    """
    messages = build_evaluator_messages(problem_text)
    flat = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return MutationTask(op="evaluate", prompt=flat, parent=program, messages=messages)


def parse_evaluator_verdict(output: str) -> tuple[bool, str]:
    """Parse an evaluator response into ``(is_valid, reason)``.

    A candidate passes ONLY on an explicit VALID verdict. Reads the ``verdict:`` /
    ``reason:`` lines first, then falls back to scanning the whole text. Anything
    else -- INVALID, no verdict at all, empty, or off-format output -- is treated
    as NOT valid and discarded, so only problems the evaluator clearly endorses
    reach the archive. ``INVALID`` is checked before ``VALID`` because it contains
    ``VALID`` as a substring.
    """
    text = output or ""
    reason = ""
    verdict = ""
    for line in text.splitlines():
        low = line.strip().lower()
        if low.startswith("reason:"):
            reason = line.split(":", 1)[1].strip()
        elif low.startswith("verdict:"):
            verdict = line.split(":", 1)[1].strip().upper()
    if not verdict:
        upper = text.upper()
        if "INVALID" in upper:
            verdict = "INVALID"
        elif "VALID" in upper:
            verdict = "VALID"
    is_valid = verdict.startswith("VALID")  # INVALID / missing / off-format -> discard
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


def _template_context(
    op: str,
    parent: ProblemProgram,
    parent_b: ProblemProgram | None = None,
) -> dict[str, str]:
    context = {
        "few_shot_examples": _load_shot_examples(op),
        "parent_id": parent.program_id,
        "parent_generation": str(parent.generation),
        "parent_source": strip_module_docstring(parent.source_code),
        "parent_concept_group": str(parent.get_concept_group() or ""),
        "parent_concept_type": str(parent.get_concept_type() or ""),
        "parent_p_hat": f"{float(getattr(parent, 'p_hat', 0.0) or 0.0):.3f}",
        "parent_h_score": f"{float(getattr(parent, 'h_score', 0.0) or 0.0):.3f}",
        "parent_rq_score": f"{float(getattr(parent, 'rq_score', 0.0) or 0.0):.6f}",
    }
    if parent_b is not None:
        context.update(
            {
                "parent_b_id": parent_b.program_id,
                "parent_b_generation": str(parent_b.generation),
                "parent_b_source": strip_module_docstring(parent_b.source_code),
                "parent_b_concept_group": str(parent_b.get_concept_group() or ""),
                "parent_b_concept_type": str(parent_b.get_concept_type() or ""),
                "parent_b_p_hat": f"{float(getattr(parent_b, 'p_hat', 0.0) or 0.0):.3f}",
                "parent_b_h_score": f"{float(getattr(parent_b, 'h_score', 0.0) or 0.0):.3f}",
                "parent_b_rq_score": f"{float(getattr(parent_b, 'rq_score', 0.0) or 0.0):.6f}",
            }
        )
    return context


def build_solver_prompt(problem: str) -> str:
    return f"{SOLVER_SYSTEM_PROMPT}\n\nProblem: {problem}\n\n"
