"""The judge gate: a child is archived only under labels the judge re-derives.

Two failures this file pins down. First, one oversized candidate must not take a
run down with it -- a generated statement with no upper bound reached 12,554
tokens against a 12,000-token window, and batched generation fails as a unit.
Second, and the reason the gate exists at all: the Evolver labels its own output,
so a child can declare GROUP/SKILL that its visible problem does not support. The
judge sees the problem and answer ONLY, and both of its labels must match.
"""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.code_utils import MAX_PROBLEM_TEXT_CHARS, lint_problem_instance
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.openai_evaluator import EvaluatorRuntimeError
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.prompts import MUTATION_OP

DECLARED_GROUP = "algebra"
DECLARED_SKILL = "counting"

AGREES = (
    f"GROUP: {DECLARED_GROUP}\n"
    "GROUP_EVIDENCE: The objects are polynomials related by their coefficients.\n"
    f"SKILL: {DECLARED_SKILL}\n"
    "SKILL_WITNESS: The count comes from a complement over the stated set.\n"
    "CLOSEST_ALTERNATIVE: casework\n"
    "WHY_NOT_ALTERNATIVE: The branches are resolved by one uniform formula.\n"
    "FAILURE_REASON: none"
)


def _verdict(group: str, skill: str, failure: str = "none") -> str:
    return (
        f"GROUP: {group}\nGROUP_EVIDENCE: e\nSKILL: {skill}\n"
        f"SKILL_WITNESS: w\nCLOSEST_ALTERNATIVE: none\n"
        f"WHY_NOT_ALTERNATIVE: none\nFAILURE_REASON: {failure}"
    )


def _instance(problem: str) -> ProblemInstance:
    return ProblemInstance(problem=problem, answer="42", program_id="p", seed=0)


def _child(tag: str = "x") -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return "The {tag} question: what is 1 + {{seed}}?", str(1 + seed)


GROUP = "{DECLARED_GROUP}"
SKILL = "{DECLARED_SKILL}"
''',
        metadata={"op": MUTATION_OP},
    )


def _entry(tag: str = "x", problem: str = "What is 2 + 2?") -> dict:
    return {
        "task": type("T", (), {"op": MUTATION_OP})(),
        "child": _child(tag),
        "inst": _instance(problem),
    }


class _Backend:
    """Records what actually reached the model."""

    def __init__(self, window: int | None, reply: str = AGREES) -> None:
        self.max_model_len = window
        self.reply = reply
        self.seen: list = []

    def mutate(self, tasks):
        self.seen.append(list(tasks))
        # A real engine raises on an over-window prompt, failing the batch.
        for task in tasks:
            size = sum(len(m.get("content", "")) for m in task.messages or [])
            if self.max_model_len and size > self.max_model_len * 4:
                raise ValueError(
                    f"The decoder prompt (length {size // 4}) is longer than "
                    f"the maximum model length of {self.max_model_len}."
                )
        return [self.reply] * len(tasks)


def _evolver(backend) -> RQEvolver:
    return RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(use_evaluator=True),
    )


# --- the root cause: an unbounded statement ------------------------------


def test_a_runaway_problem_statement_is_rejected():
    ok = _instance("Count the divisors of 5040. State only the integer.")
    assert lint_problem_instance(ok) == []

    runaway = _instance("Consider " + "1, " * 4000 + "and count them.")
    reasons = lint_problem_instance(runaway)
    assert any("too long" in r for r in reasons), reasons


def test_the_bound_leaves_real_statements_far_below_it():
    """Seeds and verified fixtures top out near 400 chars."""
    assert MAX_PROBLEM_TEXT_CHARS >= 4000
    assert lint_problem_instance(_instance("x" * (MAX_PROBLEM_TEXT_CHARS - 1))) == []


# --- the amplifier: one bad child killing the batch ----------------------


def test_an_oversized_candidate_is_dropped_not_escalated():
    backend = _Backend(window=12000)
    evolver = _evolver(backend)
    entries = [_entry("big", "Q " * 40000)]
    evolver._apply_judge(entries)

    assert entries[0]["report"].status == "judge_input_too_large"
    assert backend.seen == [], "the over-budget prompt must never reach the model"
    assert any(e["event"] == "judge_input_too_large" for e in evolver.events)


def test_a_healthy_neighbour_still_gets_judged():
    """The point of dropping rather than raising: the rest of the batch lives."""
    backend = _Backend(window=12000)
    evolver = _evolver(backend)
    entries = [_entry("big", "Q " * 40000), _entry("small")]
    evolver._apply_judge(entries)

    assert entries[0]["report"].status == "judge_input_too_large"
    assert "child" in entries[1], "the healthy candidate survived the judge"
    assert len(backend.seen) == 1 and len(backend.seen[0]) == 1


def test_a_genuine_backend_failure_still_aborts_the_run():
    """Dropping oversized inputs must not soften real judge breakage."""

    class Broken(_Backend):
        def mutate(self, tasks):
            raise RuntimeError("engine died")

    evolver = _evolver(Broken(window=12000))
    with pytest.raises(EvaluatorRuntimeError, match="engine died"):
        evolver._apply_judge([_entry()])


def test_without_a_declared_window_nothing_is_dropped():
    """Refuse to guess a limit and reject valid children on an invented one."""
    backend = _Backend(window=None)
    evolver = _evolver(backend)
    evolver._apply_judge([_entry("x", "Q " * 40000)])
    assert len(backend.seen) == 1


# --- the label-truth gate ------------------------------------------------


def test_agreement_on_both_axes_lets_the_child_through():
    evolver = _evolver(_Backend(window=12000, reply=AGREES))
    entries = [_entry()]
    evolver._apply_judge(entries)

    assert "child" in entries[0]
    judged = entries[0]["child"].metadata["judge"]
    assert judged["accepted"] is True
    assert judged["group"] == DECLARED_GROUP and judged["skill"] == DECLARED_SKILL


@pytest.mark.parametrize(
    "reply, axis",
    [
        (_verdict("number_theory", DECLARED_SKILL), "GROUP"),
        (_verdict(DECLARED_GROUP, "invariant"), "SKILL"),
        (_verdict("geometry", "induction"), "GROUP"),
    ],
)
def test_a_label_the_judge_does_not_reach_is_rejected(reply, axis):
    """A valid problem under the WRONG label is the failure the MAP cannot take.

    It is not a scoring error: the child is filed in a cell whose coordinates it
    does not satisfy, so the parent sampler and the coverage metric both read a
    fiction. Rejecting is the only outcome that keeps the grid honest.
    """
    evolver = _evolver(_Backend(window=12000, reply=reply))
    entries = [_entry()]
    evolver._apply_judge(entries)

    report = entries[0]["report"]
    assert report.status == "judge_rejected"
    assert "label mismatch" in report.reason and axis in report.reason


@pytest.mark.parametrize(
    "reply",
    [
        _verdict("none", "none", "The omitted terms are not determined."),
        _verdict(DECLARED_GROUP, "none", "Valid but routine."),
        "I am not able to classify this problem.",
        "",
    ],
)
def test_a_judge_that_fails_closed_rejects(reply):
    """Missing, refused, and unreadable must be indistinguishable from a NO."""
    evolver = _evolver(_Backend(window=12000, reply=reply))
    entries = [_entry()]
    evolver._apply_judge(entries)

    assert entries[0]["report"].status == "judge_rejected"
    assert "failed closed" in entries[0]["report"].reason


def test_the_judge_never_sees_the_source_or_the_declared_labels():
    """Its answer can only disagree if it was derived independently."""
    backend = _Backend(window=12000)
    evolver = _evolver(backend)
    evolver._apply_judge([_entry()])

    sent = "\n".join(m["content"] for m in backend.seen[0][0].messages)
    assert "What is 2 + 2?" in sent and "42" in sent
    assert "def generate" not in sent
    assert f'GROUP = "{DECLARED_GROUP}"' not in sent


def test_the_disagreement_is_recorded_even_when_it_rejects():
    """The mismatch rate is the measurement that says whether labels are trusted."""
    evolver = _evolver(_Backend(window=12000, reply=_verdict("geometry", "invariant")))
    entries = [_entry()]
    child = entries[0]["child"]
    evolver._apply_judge(entries)

    judged = child.metadata["judge"]
    assert judged["accepted"] is False
    assert judged["group"] == "geometry" and judged["skill"] == "invariant"


def test_the_gate_can_be_switched_off():
    backend = _Backend(window=12000)
    evolver = RQEvolver(
        archive=MAPElitesArchive(),
        backend=backend,
        evolution_config=EvolutionConfig(use_evaluator=False, ast_contract="enforce"),
    )
    entries = [_entry()]
    evolver._apply_judge(entries)
    assert "child" in entries[0] and backend.seen == []


# --- configuration -------------------------------------------------------


def test_ast_contract_only_accepts_the_three_modes():
    with pytest.raises(ValueError, match="ast_contract"):
        EvolutionConfig(ast_contract="maybe")


def test_turning_off_both_gates_is_refused():
    """With judge and AST contract both off nothing checks problem vs answer."""
    with pytest.raises(ValueError, match="use_evaluator"):
        EvolutionConfig(use_evaluator=False, ast_contract="off")

    assert EvolutionConfig(use_evaluator=False, ast_contract="enforce")
    assert EvolutionConfig(use_evaluator=False, ast_contract="shadow")
    assert EvolutionConfig(use_evaluator=True, ast_contract="off")


def test_the_judge_is_read_greedily():
    """Sampling noise in the gate reads as label disagreement it did not cause."""
    cfg = EvolutionConfig()
    assert cfg.judge_temperature == 0.0 and cfg.judge_top_p == 1.0


# --- R_Q = 0 occupies the map, but earns nothing from it ------------------


def _scored(pid: str, group: str, skill: str, rq: float, s_hat: float):
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            f'    return f"{pid}: what is {{seed}} + 1?", str(seed + 1)\n\n\n'
            f'GROUP = "{group}"\nSKILL = "{skill}"\n'
        ),
        program_id=pid,
    )
    program.s_hat = s_hat
    return program


def test_a_zero_scoring_program_still_takes_an_empty_cell():
    """Classical MAP-Elites does not evict on fitness alone.

    The old gate rejected every p=0 and p=1 candidate, which cost the bootstrap
    3 of 8 seeds and with them the only representative of geometry, inequality,
    induction and extremal_principle.
    """
    archive = MAPElitesArchive()
    program = _scored("zero", "geometry", "extremal_principle", 0.0, 0.0)
    assert archive.try_insert(
        program=program, u_value=0.0, rq_score=0.0
    )
    assert archive.program_to_cell(program) in {
        archive.program_to_cell(c) for c in archive.champions()
    }


def test_a_scoring_program_displaces_the_zero_one_but_not_the_reverse():
    archive = MAPElitesArchive()
    zero = _scored("zero", "algebra", "counting", 0.0, 0.0)
    real = _scored("real", "algebra", "counting", 0.25, 0.5)
    other_zero = _scored("other", "algebra", "counting", 0.0, 1.0)

    assert archive.try_insert(program=zero, u_value=0.0, rq_score=0.0)
    # 0 > 0 is False: the incumbent keeps the cell against another zero.
    assert not archive.try_insert(
        program=other_zero, u_value=0.0, rq_score=0.0
    )
    assert archive.try_insert(
        program=real, u_value=1.0, rq_score=0.25
    )
    assert [c.program_id for c in archive.champions()] == ["real"]
    # ...and the zero cannot take it back.
    assert not archive.try_insert(
        program=zero, u_value=0.0, rq_score=0.0
    )


def test_a_zero_scoring_champion_contributes_no_training_examples():
    """Occupying a cell and earning a training slot are separate questions.

    The frontier band (low < s_hat < high) is what keeps p=0 and p=1 out of the
    dataset, so the archive gate was never the thing protecting training data.
    """
    from rq_evolve.scoring import is_frontier

    assert not is_frontier(0.0, 0.0, 1.0)
    assert not is_frontier(1.0, 0.0, 1.0)
    assert is_frontier(0.5, 0.0, 1.0)
