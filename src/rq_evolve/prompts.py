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
    "You design a Python program for competition-math problems. "
    "Each file defines `generate(seed)`, which returns one "
    "(problem_text, answer) pair, and then labels what it produced.\n"
    "Each parent carries three solver signals:\n"
    "  - p_hat: the solver's success rate (0-1).\n"
    "  - uncertainty/H: how unsure the solver is; higher is better.\n"
    "  - R_Q: the product p_hat * H, the problem's overall quality.\n"
    "Design the new problem to maximize R_Q: since it is p_hat * H, aim for the edge of the solver's ability — solvable, yet still uncertain.\n"
    "\n"
    "Output structure, in this order:\n"
    "  1. an optional module docstring — the mutation idea and how "
    "the resulting problem is solved;\n"
    "  2. imports (only collections, fractions, functools, itertools, "
    "math, random, sympy);\n"
    "  3. `def generate(seed)`;\n"
    "  4. the constants CONCEPT_REASON, CONCEPT_GROUP, CONCEPT_TYPE, "
    "in that order.\n"
    "\n"
    f"CONCEPT_GROUP must be exactly one of: {groups}\n"
    "CONCEPT_TYPE is a free-form '<group>.<snake_case_name>' string.\n"
    "Fill the three constants:\n"
    "  - CONCEPT_REASON: It must describe the core mathematical reasoning the solver performs based strictly on the problem text. If the problem reduces to a simple operation, state it plainly rather than overcomplicating the label."
    "\n"
    "  - CONCEPT_GROUP and CONCEPT_TYPE: name the reasoning that CONCEPT_REASON describes."
    "\n"
    "  - The problem text must never reveal the answer's value — no "
    "\"... and the result is 17\", no \"simplify A/B = 2002\". The "
    "solver computes it.\n"
    "\n"
    "Please reason step by step, and put your final program within ```python ```"
)


@dataclass(slots=True)
class MutationTask:
    op: str
    prompt: str
    parent: ProblemProgram
    parent_b: ProblemProgram | None = None


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
    """Build one mutation prompt from ``prompt_templates/<op>.txt``.

    Edit these files to customize each mutation operator:
      - prompt_templates/in_depth.txt
      - prompt_templates/in_breadth.txt
      - prompt_templates/crossover.txt

    Templates use ``string.Template`` placeholders such as
    ``$few_shot_examples``, ``$parent_source``, and ``$parent_b_source`` so
    Python code blocks can contain normal ``{...}`` braces without escaping.
    """
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
