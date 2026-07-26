import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template

from .code_utils import strip_module_docstring
from .metacognition import mutation_plan_id, validate_mutation_plan
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

METACOGNITIVE_PLAN_SYSTEM_PROMPT = (
    "You are the metacognitive planner for an evolving competition-math "
    "generator. Analyze the current Solver's observed reasoning evidence and "
    "write an executable mutation specification, not Python code.\n"
    "Return exactly one JSON object with schema_version 2. The plan must identify "
    "a concrete confident-wrong route, preserve or transfer one explicit "
    "target_reasoning_move, specify bounded deterministic sampling, define "
    "independent insight and brute routes, and define a live integer-valued decoy "
    "route that differs from the answer.\n"
    "Do not fabricate observed facts that are absent from the evidence. When the "
    "evidence is sparse, state the uncertainty in failure_summary but still make "
    "the proposed generator mechanically testable."
)

PLANNED_MUTATION_SYSTEM_PROMPT = (
    "You implement a validated metacognitive mutation plan as a deterministic "
    "Python competition-math problem generator.\n\n"
    "Output exactly one full program in a ```python``` block. The program must:\n"
    "  1. import only collections, fractions, functools, itertools, math, random, sympy;\n"
    "  2. define MAX_ATTEMPTS = 200 and `generate(seed)`;\n"
    "  3. use only `rng = random.Random(seed)` for randomness;\n"
    "  4. sample with a bounded `for _ in range(MAX_ATTEMPTS)` loop and raise "
    "RuntimeError in the loop's `else` clause;\n"
    "  5. compute genuinely independent `answer_insight` and `answer_brute` "
    "routes and assert their equality;\n"
    "  6. compute the planned decoy route, resample if it accidentally equals "
    "the answer, then assert that the accepted decoy differs;\n"
    "  7. return exactly one problem and a base-10 integer string serialized as "
    "`str(sympy.Integer(answer))`;\n"
    "  8. finish with CONCEPT_REASON, CONCEPT_GROUP, CONCEPT_TYPE in that order.\n\n"
    f"CONCEPT_GROUP must be exactly one of: {groups}\n"
    "Never reveal the answer, intended insight, decoy, theorem choice, or "
    "intermediate computed values in the problem text."
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
    # When set, the backend renders this full chat conversation as the prompt
    # (multi-turn self-fix) instead of wrapping ``prompt`` as a single user msg.
    messages: list[dict] | None = None
    stage: str = "code"
    mutation_plan: dict | None = None
    plan_id: str | None = None
    plan_status: str = "legacy"
    max_output_tokens: int | None = None


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
}
SHOT_TEMPLATE_FILES = PROMPT_TEMPLATE_FILES

METACOGNITIVE_SHOT_TEMPLATE_FILES = {
    "in_depth": "metacognitive_in_depth.txt",
    "in_breadth": "metacognitive_in_breadth.txt",
}

PLANNED_PROMPT_TEMPLATE_FILES = {
    "in_depth": "planned_in_depth.txt",
    "in_breadth": "planned_in_breadth.txt",
}


def build_mutation_task(
    op: str,
    parent: ProblemProgram,
) -> MutationTask:
    if op not in PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")

    template = _load_prompt_template(op)
    user = Template(template).safe_substitute(
        _template_context(op=op, parent=parent)
    )

    return MutationTask(
        op=op,
        prompt=f"{MUTATION_SYSTEM_PROMPT}\n\n{user}",
        parent=parent,
    )


def build_metacognitive_plan_task(
    op: str,
    parent: ProblemProgram,
    *,
    evidence: list[dict],
    meta_progress: dict,
    max_output_tokens: int | None = None,
) -> MutationTask:
    if op not in PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")
    template = (PROMPT_TEMPLATE_DIR / "metacognitive_plan.txt").read_text(
        encoding="utf-8"
    )
    constraint = (
        "Preserve both CONCEPT_GROUP and CONCEPT_TYPE."
        if op == "in_depth"
        else (
            "Change CONCEPT_GROUP, mathematical object, domain-specific operation, "
            "and surface vocabulary while preserving the abstract "
            "target_reasoning_move."
        )
    )
    inherited_reasoning_move = str(
        (
            (parent.metadata or {}).get("mutation_plan") or {}
        ).get("target_reasoning_move", "")
    )
    shots = _load_named_shots(METACOGNITIVE_SHOT_TEMPLATE_FILES[op])
    context = {
        **_template_context(op=op, parent=parent),
        "operator": op,
        "operator_contract": constraint,
        "inherited_reasoning_move": inherited_reasoning_move,
        "behavioral_evidence": json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
        ),
        "meta_progress": json.dumps(
            meta_progress,
            ensure_ascii=False,
            indent=2,
        ),
    }
    user = Template(template).safe_substitute(
        {**context, "few_shot_examples": shots}
    )
    live_user = Template(template).safe_substitute(
        {**context, "few_shot_examples": ""}
    )
    messages = [{"role": "system", "content": METACOGNITIVE_PLAN_SYSTEM_PROMPT}]
    if shots:
        messages.extend(
            [
                {"role": "user", "content": shots},
                {
                    "role": "assistant",
                    "content": (
                        "I will follow the demonstrated schema and grounding "
                        "rules for the live request."
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": live_user})
    return MutationTask(
        op=op,
        prompt=f"{METACOGNITIVE_PLAN_SYSTEM_PROMPT}\n\n{user}",
        parent=parent,
        messages=messages,
        stage="plan",
        plan_status="requested",
        max_output_tokens=max_output_tokens,
    )


def parse_mutation_plan(
    output: str | None,
    op: str,
    *,
    required_target_reasoning_move: str = "",
) -> tuple[dict | None, str]:
    text = str(output or "").strip()
    if not text:
        return None, "empty plan output"

    candidates = [
        match.group(1).strip()
        for match in re.finditer(
            r"```json[ \t]*\n(.*?)```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
    ]
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    candidates.append(text)

    last_error = "no JSON object"
    decoded: list[object] = []
    for candidate in candidates:
        try:
            decoded.append(json.loads(candidate))
        except json.JSONDecodeError as exc:
            last_error = f"invalid plan JSON: {exc}"

    # Qwen-style reasoning may put prose or unrelated braces before an otherwise
    # valid bare JSON plan. Recover complete objects without accepting partial
    # JSON fragments or changing the strict schema check below.
    decoder = json.JSONDecoder()
    for match in list(re.finditer(r"\{", text))[:32]:
        try:
            payload, _end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        decoded.append(payload)

    for payload in decoded:
        if not isinstance(payload, dict):
            last_error = "plan JSON must be an object"
            continue
        errors = validate_mutation_plan(payload, op)
        if errors:
            last_error = "; ".join(errors[:5])
            continue
        if (
            required_target_reasoning_move
            and str(payload.get("target_reasoning_move", ""))
            != required_target_reasoning_move
        ):
            reason = (
                "target_reasoning_move must exactly match the inherited "
                "breadth transfer join key"
            )
            return None, reason
        return payload, ""
    return None, last_error


def build_planned_mutation_task(
    op: str,
    parent: ProblemProgram,
    plan: dict,
) -> MutationTask:
    if op not in PLANNED_PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")
    path = PROMPT_TEMPLATE_DIR / PLANNED_PROMPT_TEMPLATE_FILES[op]
    template = path.read_text(encoding="utf-8")
    plan_id = mutation_plan_id(plan)
    shots = _load_named_shots(PLANNED_PROMPT_TEMPLATE_FILES[op])
    context = {
        **_template_context(op=op, parent=parent),
        "mutation_plan": json.dumps(plan, ensure_ascii=False, indent=2),
        "plan_id": plan_id,
    }
    user = Template(template).safe_substitute(
        {**context, "few_shot_examples": shots}
    )
    live_user = Template(template).safe_substitute(
        {**context, "few_shot_examples": ""}
    )
    messages = [{"role": "system", "content": PLANNED_MUTATION_SYSTEM_PROMPT}]
    if shots:
        messages.extend(
            [
                {"role": "user", "content": shots},
                {
                    "role": "assistant",
                    "content": (
                        "I will preserve this program structure while "
                        "implementing only the live plan."
                    ),
                },
            ]
        )
    messages.append({"role": "user", "content": live_user})
    return MutationTask(
        op=op,
        prompt=f"{PLANNED_MUTATION_SYSTEM_PROMPT}\n\n{user}",
        parent=parent,
        messages=messages,
        stage="code",
        mutation_plan=plan,
        plan_id=plan_id,
        plan_status="planned",
    )


def build_fix_task(
    task: MutationTask,
    failed_output: str,
    reason: str,
) -> MutationTask:
    system_prompt = (
        PLANNED_MUTATION_SYSTEM_PROMPT
        if task.mutation_plan is not None
        else MUTATION_SYSTEM_PROMPT
    )
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
        mutation_plan=task.mutation_plan,
        plan_id=task.plan_id,
        plan_status=task.plan_status,
        max_output_tokens=task.max_output_tokens,
    )


def _load_evaluator_shots() -> str:
    path = SHOT_TEMPLATE_DIR / EVALUATOR_SHOT_FILE
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_evaluator_messages(
    problem_text: str,
    mutation_plan: dict | None = None,
) -> list[dict]:
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
    review = (
        "Now evaluate the following problem.\n\n"
        f"Problem:\n{problem_text.strip()}\n"
    )
    if mutation_plan:
        review += (
            "\nThe generator was produced from this mutation plan. In addition "
            "to coherence, mark INVALID if the problem leaks the intended move, "
            "does not plausibly require it, or violates the stated problem/answer "
            "contract.\n\nMutation plan:\n"
            + json.dumps(mutation_plan, ensure_ascii=False, indent=2)
            + "\n"
        )
    blocks.append(f"{review}\nAnswer:")
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(blocks)},
    ]


def build_evaluator_task(
    program: ProblemProgram,
    problem_text: str,
    mutation_plan: dict | None = None,
) -> MutationTask:
    """Wrap an evaluator query as a MutationTask so ``backend.mutate`` can run it.

    Reuses the existing batched generate path (mutate reads ``messages``); no new
    backend method is needed. ``parent`` carries the program under review purely
    for reporting -- mutate only consumes ``messages``/``prompt``.
    """
    messages = build_evaluator_messages(problem_text, mutation_plan)
    flat = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return MutationTask(
        op="evaluate",
        prompt=flat,
        parent=program,
        messages=messages,
        stage="evaluate",
        mutation_plan=mutation_plan,
    )


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


def _load_named_shots(filename: str) -> str:
    path = SHOT_TEMPLATE_DIR / filename
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return ""
    return f"Few-shot examples:\n\n{text}"


def _template_context(
    op: str,
    parent: ProblemProgram,
) -> dict[str, str]:
    return {
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


def build_solver_prompt(problem: str) -> str:
    return f"{SOLVER_SYSTEM_PROMPT}\n\nProblem: {problem}\n\n"
