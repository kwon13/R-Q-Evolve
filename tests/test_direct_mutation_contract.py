"""The direct mutation contract inside the training loop.

``mutation_contract="direct"`` puts the correct/wrong solver traces straight
into the code-writing prompt and drops the plan step. It exists because the
summarising step was where the contrast was lost: a 22-field plan reported "the
solver incorrectly adds the equations" on a trace whose addition was
arithmetically correct, so nothing downstream could target the real error.

These tests pin what the path must keep once the notation lint is waived: an
integer answer taken from execution rather than from the source text, and one
generation call per mutation with the traces present only when a clean contrast
exists.
"""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import EvolutionConfig, MetacognitionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import MutationTask, render_reasoning_contrast

PARENT_SOURCE = '''
import random

MAX_ATTEMPTS = 200

def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 9)
    b = rng.randint(1, 9)
    return f"Compute {a} + {b}.", str(a + b)

CONCEPT_REASON = "addition"
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
'''


def _parent() -> ProblemProgram:
    return ProblemProgram(source_code=PARENT_SOURCE, metadata={})


def _evidence(problem: str = "Compute 2 + 3.") -> list[dict]:
    return [
        {
            "role": "success",
            "correct": True,
            "problem": problem,
            "predicted_answer": "5",
            "response": "Two plus three is five.",
        },
        {
            "role": "failure",
            "correct": False,
            "problem": problem,
            "predicted_answer": "6",
            "response": "I add one to the total to be safe, giving six.",
        },
    ]


class _RecordingBackend:
    """Captures the prompts the evolver would send."""

    tokenizer = None

    def __init__(self) -> None:
        self.batches: list[list[MutationTask]] = []

    def mutate(self, tasks):
        self.batches.append(list(tasks))
        return ["" for _ in tasks]


def _evolver(backend, contract: str = "direct") -> RQEvolver:
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(
            mutation_contract=contract,
            verify_seeds=3,
        ),
        metacognition_config=MetacognitionConfig(enabled=True),
    )


# ---------------------------------------------------------------------------
# The traces reach the code-writing call, without a plan step.
# ---------------------------------------------------------------------------


def test_direct_contract_makes_one_call_and_skips_planning(monkeypatch):
    parent = _parent()
    backend = _RecordingBackend()
    evolver = _evolver(backend)
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: _evidence(),
    )

    prepared = evolver._prepare_mutation_tasks([("in_depth", parent)])

    # No plan call was made: the planning path batches a "plan" stage first.
    assert backend.batches == []
    assert len(prepared) == 1
    task = prepared[0]
    assert isinstance(task, MutationTask)
    assert task.stage == "code"
    assert task.mutation_plan is None
    assert task.generation_path == "direct_freeform"
    assert task.plan_status == "direct_with_contrast"
    # Both traces are visible verbatim in the prompt the model receives.
    user = task.messages[1]["content"]
    assert "Two plus three is five." in user
    assert "I add one to the total to be safe, giving six." in user


@pytest.mark.parametrize("evidence", [[], [_evidence()[0]]], ids=["none", "half"])
def test_direct_contract_refuses_to_mutate_without_a_contrast(
    monkeypatch,
    evidence,
):
    """Silently dropping the traces would mix two conditions in one run.

    A direct mutation without the pair is an ordinary free-form mutation, not
    the treatment under test, so it is refused and reported rather than run.
    """
    from rq_evolve.evolution import CandidateReport

    parent = _parent()
    evolver = _evolver(_RecordingBackend())
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: list(evidence),
    )

    prepared = evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert len(prepared) == 1
    report = prepared[0]
    assert isinstance(report, CandidateReport)
    assert report.status == "direct_contrast_missing"
    assert "correct/wrong trace pair" in report.reason
    assert any(
        event["event"] == "direct_contrast_missing" for event in evolver.events
    )


def test_planned_contract_is_unchanged(monkeypatch):
    """The default path must still plan; direct is opt-in."""
    parent = _parent()
    backend = _RecordingBackend()
    evolver = _evolver(backend, contract="planned")
    monkeypatch.setattr(
        "rq_evolve.evolution.collect_planning_evidence",
        lambda *args, **kwargs: _evidence(),
    )

    evolver._prepare_mutation_tasks([("in_depth", parent)])

    assert backend.batches, "planned contract must issue a planning call"
    assert backend.batches[0][0].stage == "plan"


def test_contrast_renders_both_roles_and_withholds_them_from_the_program():
    rendered = render_reasoning_contrast(_evidence())
    assert "Two plus three is five." in rendered
    assert "I add one to the total to be safe, giving six." in rendered
    # The generated program must not talk about the traces.
    assert "Do not mention the traces" in rendered
    assert render_reasoning_contrast([]) == ""
    # A pair missing one side carries no contrast.
    assert render_reasoning_contrast([_evidence()[0]]) == ""


# ---------------------------------------------------------------------------
# What replaces the waived notation lint.
# ---------------------------------------------------------------------------


def _verify(source: str, *, direct: bool):
    metadata = {"op": "in_depth"}
    if direct:
        metadata["free_form_direct"] = True
    program = ProblemProgram(source_code=source, metadata=metadata)
    return _evolver(_RecordingBackend()).verify_program(program, n_seeds=3)


_NO_SYMPY_WRAPPER = '''
import random

def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 9)
    return f"Compute {a} + 1.", str(a + 1)

CONCEPT_REASON = "addition"
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
'''

_NON_INTEGER_ANSWER = '''
import random

def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 9)
    return f"Compute half of the integer {a}.", str(a / 2)

CONCEPT_REASON = "division"
CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
'''


def test_direct_path_accepts_a_program_without_the_notation_boilerplate():
    """Nine of fourteen rejections were notation, not mathematics."""
    instance, reason = _verify(_NO_SYMPY_WRAPPER, direct=True)
    assert instance is not None, reason


def test_the_same_program_is_still_rejected_on_the_planned_path():
    instance, reason = _verify(_NO_SYMPY_WRAPPER, direct=False)
    assert instance is None
    assert "MAX_ATTEMPTS" in reason or "sympy.Integer" in reason


def test_direct_path_still_requires_an_integer_answer_from_execution():
    """Waiving the static contract moves the guarantee, it does not drop it."""
    instance, reason = _verify(_NON_INTEGER_ANSWER, direct=True)
    assert instance is None
    assert "integer" in reason


@pytest.mark.parametrize(
    "source,expected",
    [
        (
            # Same problem on every seed: a constant generator teaches nothing.
            "import random\n"
            "def generate(seed):\n"
            "    rng = random.Random(seed)\n"
            '    return "Compute 1 + 1.", "2"\n'
            'CONCEPT_REASON = "t"\nCONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n',
            "vary",
        ),
        (
            # Forbidden import survives the relaxation.
            "import os\n"
            "import random\n"
            "def generate(seed):\n"
            "    rng = random.Random(seed)\n"
            '    return f"Compute {rng.randint(1,9)} + 1.", "2"\n'
            'CONCEPT_REASON = "t"\nCONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n',
            "import",
        ),
    ],
)
def test_relaxation_does_not_disable_the_substantive_checks(source, expected):
    instance, reason = _verify(source, direct=True)
    assert instance is None
    assert expected in reason
