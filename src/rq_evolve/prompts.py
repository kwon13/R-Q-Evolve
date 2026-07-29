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

MUTATION_CODE_SKELETON = """
Required structural skeleton (replace every <...> placeholder; this is not a
mathematical example and must not be returned with placeholders):
```python
import random
import sympy

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        <sample primitive parameters>
        if <degenerate condition>:
            continue
        answer_insight = <insight computation>
        answer_brute = <independent brute computation>
        assert answer_insight == answer_brute
        answer = answer_insight
        problem = <one visible problem statement>
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed to sample a valid instance")

CONCEPT_REASON = "<visible reasoning required by the problem>"
CONCEPT_GROUP = "<allowed group>"
CONCEPT_TYPE = "<group.snake_case_name>"
```
"""

PLANNED_CODE_SKELETON = """
Required structural skeleton (replace every <...> placeholder; this is not a
mathematical example and must not be returned with placeholders):
```python
import random
import sympy

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        <sample the plan's primitive parameters>
        if <degenerate condition>:
            continue
        answer = <the plan's executable answer_route>
        problem = <one visible problem statement>
        return problem, str(sympy.Integer(answer))
    else:
        raise RuntimeError("failed to sample a valid instance")

CONCEPT_REASON = "<visible reasoning required by the problem>"
CONCEPT_GROUP = "<allowed group>"
CONCEPT_TYPE = "<group.snake_case_name>"
```
"""

MUTATION_SYSTEM_PROMPT = (
    "You design one executable Python program for competition-math problems. "
    "Each file defines `generate(seed)`, which returns one (problem_text, answer) "
    "pair, and then labels what it produced.\n"
    "\n"
    "Think silently. Return exactly one complete program in one ```python``` "
    "block and no other text. Never output analysis, JSON, a chat-message "
    "object, role/content wrappers, example labels, the parent program, or any "
    "few-shot example. Generate a new program for the LIVE parent only.\n"
    "\n"
    "Output structure, in this order:\n"
    "  1. an optional concise module docstring;\n"
    "  2. imports (only collections, fractions, functools, itertools, math, random, sympy);\n"
    "  3. `MAX_ATTEMPTS = 200`;\n"
    "  4. `def generate(seed)` using `rng = random.Random(seed)` and a bounded "
    "`for _ in range(MAX_ATTEMPTS)` sampler with a RuntimeError exhaustion else;\n"
    "  5. the constants CONCEPT_REASON, CONCEPT_GROUP, CONCEPT_TYPE, in that order.\n"
    "Return the answer as one base-10 integer string using "
    "`str(sympy.Integer(answer))`.\n"
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
    "Semantic consistency contract:\n"
    "  - Every mathematical object described in problem_text must be exactly "
    "the object used to compute the returned answer.\n"
    "  - The answer may depend only on quantities stated or unambiguously "
    "defined in problem_text. Hidden coefficients, unstated transformations, "
    "unused sampled variables, and mislabeled sequences are forbidden.\n"
    "  - `answer_insight` and `answer_brute` may share sampled primitive "
    "parameters, but they may not share a derived target quantity. Direct "
    "assignment between the routes, identical right-hand sides, and the same "
    "formula under renamed variables are forbidden.\n"
    "  - The brute route must reconstruct the answer by literal enumeration, "
    "recurrence, direct substitution, or another algorithm that does not use "
    "the insight route's identity or intermediates.\n"
    "  - Every guard must be jointly satisfiable. No derived-variable definition "
    "may make a later guard always true or always false.\n"
    "\n"
    f"{MUTATION_CODE_SKELETON}\n"
    "Before emitting code, silently verify: exactly one Python block; seeds 0 "
    "through 4 terminate; problem_text and answer describe the same mathematics; "
    "the two routes are genuinely independent; assertions pass; the result is "
    "not copied from a few-shot example; and CONCEPT_REASON describes the visible "
    "problem. If any check fails, redesign before answering."
)

METACOGNITIVE_PLAN_SYSTEM_PROMPT = (
    "You are the metacognitive planner for an evolving competition-math "
    "generator. Analyze the current Solver's observed reasoning evidence and "
    "write an executable mutation specification, not Python code.\n"
    "Return exactly one JSON object with schema_version 3, using only the fields "
    "demonstrated by the reference schema. The plan must ground one concrete "
    "observed failure step, preserve or transfer one explicit "
    "target_reasoning_move, specify bounded deterministic sampling, define "
    "one executable integer-valued `answer_route`, and keep the visible problem "
    "consistent with that route.\n"
    "First validate evidence integrity. Ignore any suffix beginning with a new "
    "User:/Assistant: role, a repeated solver instruction, a second problem, or "
    "an unrelated question/answer. A claimed failure must be supported by a "
    "specific step in the cleaned wrong trace and contrasted with a specific "
    "step in the correct trace. Do not infer behavior from predicted_answer "
    "alone. Do not fabricate observed facts, and never reuse a few-shot domain, "
    "formula, parameter set, or target move merely because evidence is sparse."
)

PLANNED_MUTATION_SYSTEM_PROMPT = (
    "You implement a validated metacognitive mutation plan as a deterministic "
    "Python competition-math problem generator.\n\n"
    "Think silently. Output exactly one full program in one ```python``` block "
    "and no other text. Do not repeat the plan, parent, examples, analysis, JSON, "
    "or chat wrappers. The program must:\n"
    "  1. import only collections, fractions, functools, itertools, math, random, sympy;\n"
    "  2. define MAX_ATTEMPTS = 200 and `generate(seed)`;\n"
    "  3. use only `rng = random.Random(seed)` for randomness;\n"
    "  4. sample with a bounded `for _ in range(MAX_ATTEMPTS)` loop and raise "
    "RuntimeError in the loop's `else` clause;\n"
    "  5. compute `answer` by the plan's single executable `answer_route`;\n"
    "  6. return exactly one problem and a base-10 integer string serialized as "
    "`str(sympy.Integer(answer))`;\n"
    "  7. finish with CONCEPT_REASON, CONCEPT_GROUP, CONCEPT_TYPE in that order.\n\n"
    f"CONCEPT_GROUP must be exactly one of: {groups}\n"
    "Never reveal the answer, intended reasoning move, theorem choice, or "
    "intermediate computed values in the problem text.\n\n"
    "Semantic consistency contract:\n"
    "  - problem_text must describe exactly the mathematical object used to "
    "compute `answer`; no hidden parameters or mislabeled structures;\n"
    "  - implement every parameter and guard required by the plan, and do not "
    "introduce unstated quantities into the answer computation;\n"
    "  - all guards must be jointly satisfiable and seeds 0 through 4 must "
    "terminate without assertion failure;\n"
    "  - do not copy the mathematical object, formulas, parameter names, or "
    "CONCEPT_TYPE of a few-shot example.\n\n"
    f"{PLANNED_CODE_SKELETON}\n"
    "Before emitting code, silently audit output shape, seed termination, "
    "problem/answer equivalence, plan coverage, novelty, and concept-label "
    "accuracy. Redesign instead of emitting a failing program."
)

EVALUATOR_SYSTEM_PROMPT = (
    "You are an evaluator for generated math word problems and their generator "
    "programs.\n"
    "Determine whether the problem is internally coherent, whether the supplied "
    "answer solves the visible problem, and—when source is supplied—whether the "
    "code computes exactly the mathematics stated in problem_text.\n\n"
    "Mark the problem as INVALID if any stated condition, theorem, system, recurrence, optimization, or variable definition is not logically connected to the final question (even if the answer can still be computed by ignoring it), if the statement combines two or more independent problems or poses multiple unrelated final questions, if the same variable name is reused for unrelated objects in an ambiguous way, or if the final requested answer does not follow from the stated problem; otherwise, check for contradictory conditions, irrelevant conditions, inapplicable claims about solution methods, and extraneous assumptions.\n"
    "Also mark INVALID if the returned answer is wrong, the code uses hidden "
    "quantities or a different mathematical object than the problem states, "
    "the code's declared answer checks are internally inconsistent, bounded "
    "sampling cannot terminate, or the program copies an example instead of mutating the live "
    "parent.\n"
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
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    if op not in PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")

    template = _load_prompt_template(op)
    context = _template_context(op=op, parent=parent)
    live_user = Template(template).safe_substitute(
        {**context, "few_shot_examples": ""}
    )

    return MutationTask(
        op=op,
        prompt=f"{MUTATION_SYSTEM_PROMPT}\n\n{live_user}",
        parent=parent,
        # Both representations omit code-rich few-shots; greedy decoding was
        # copying them instead of mutating the live parent.
        messages=[
            {"role": "system", "content": MUTATION_SYSTEM_PROMPT},
            {"role": "user", "content": live_user},
        ],
        temperature=temperature,
        top_p=top_p,
    )


def build_metacognitive_plan_task(
    op: str,
    parent: ProblemProgram,
    *,
    evidence: list[dict],
    meta_progress: dict,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
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
        temperature=temperature,
        top_p=top_p,
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
    best_schema_errors: list[str] | None = None
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
            if best_schema_errors is None or len(errors) < len(best_schema_errors):
                best_schema_errors = errors
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
    if best_schema_errors is not None:
        return None, "; ".join(best_schema_errors[:5])
    return None, last_error


def build_planned_mutation_task(
    op: str,
    parent: ProblemProgram,
    plan: dict,
    *,
    temperature: float | None = None,
    top_p: float | None = None,
) -> MutationTask:
    if op not in PLANNED_PROMPT_TEMPLATE_FILES:
        raise ValueError(f"unknown mutation op: {op}")
    path = PROMPT_TEMPLATE_DIR / PLANNED_PROMPT_TEMPLATE_FILES[op]
    template = path.read_text(encoding="utf-8")
    plan_id = mutation_plan_id(plan)
    context = {
        **_template_context(op=op, parent=parent),
        "mutation_plan": json.dumps(plan, ensure_ascii=False, indent=2),
        "plan_id": plan_id,
    }
    live_user = Template(template).safe_substitute(
        {**context, "few_shot_examples": ""}
    )
    # The validated live plan supplies the needed semantics. Omitting the
    # content-rich code shot here prevents greedy decoding from copying it.
    messages = [
        {"role": "system", "content": PLANNED_MUTATION_SYSTEM_PROMPT},
        {"role": "user", "content": live_user},
    ]
    return MutationTask(
        op=op,
        prompt=f"{PLANNED_MUTATION_SYSTEM_PROMPT}\n\n{live_user}",
        parent=parent,
        messages=messages,
        stage="code",
        mutation_plan=plan,
        plan_id=plan_id,
        plan_status="planned",
        temperature=temperature,
        top_p=top_p,
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
        temperature=task.temperature,
        top_p=task.top_p,
    )


def _load_evaluator_shots() -> str:
    path = SHOT_TEMPLATE_DIR / EVALUATOR_SHOT_FILE
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def build_evaluator_messages(
    problem_text: str,
    mutation_plan: dict | None = None,
    *,
    answer_text: str | None = None,
    program_source: str | None = None,
) -> list[dict]:
    """Render the semantic generator review conversation for one problem."""
    blocks: list[str] = []
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
        mutation_plan,
        answer_text=answer_text,
        program_source=program_source,
    )
    flat = f"{messages[0]['content']}\n\n{messages[1]['content']}"
    return MutationTask(
        op="evaluate",
        prompt=flat,
        parent=program,
        messages=messages,
        stage="evaluate",
        mutation_plan=mutation_plan,
        temperature=temperature,
        top_p=top_p,
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
