"""One oversized candidate must not take a run down with it.

A generated statement with no upper bound reached 12,554 tokens against a
12,000-token rollout window in a sample run. Batched generation fails as a
unit, and the evaluator escalates any backend failure to
``EvaluatorRuntimeError``, so that single child aborted the whole phase.
"""

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.code_utils import MAX_PROBLEM_TEXT_CHARS, lint_problem_instance
from rq_evolve.config import EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.openai_evaluator import EvaluatorRuntimeError
from rq_evolve.program import ProblemInstance, ProblemProgram


def _instance(problem: str) -> ProblemInstance:
    return ProblemInstance(problem=problem, answer="42", program_id="p", seed=0)


def _child(tag: str = "x") -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return "The {tag} question: what is 1 + {{seed}}?", str(1 + seed)


GROUP = "algebra"
SKILL = "counting"
''',
        metadata={"op": "in_depth"},
    )


class _Backend:
    """Records what actually reached the model."""

    def __init__(
        self,
        window: int | None,
        # A child now has to be judged on skill necessity as well as
        # coherence, so the stock reply answers both.
        reply: str = "reason: fine\nskill_required: YES\nverdict: VALID",
    ) -> None:
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
    entries = [
        {
            "task": type("T", (), {"op": "in_depth"})(),
            "child": _child("big"),
            "inst": _instance("Q " * 40000),
        }
    ]
    evolver._apply_evaluator(entries)

    assert "report" in entries[0]
    assert entries[0]["report"].status == "evaluator_input_too_large"
    assert backend.seen == [], "the over-budget prompt must never reach the model"
    assert any(
        e["event"] == "evaluator_input_too_large" for e in evolver.events
    )


def test_a_healthy_neighbour_still_gets_evaluated():
    """The point of dropping rather than raising: the rest of the batch lives."""
    backend = _Backend(window=12000)
    evolver = _evolver(backend)
    entries = [
        {
            "task": type("T", (), {"op": "in_depth"})(),
            "child": _child("big"),
            "inst": _instance("Q " * 40000),
        },
        {
            "task": type("T", (), {"op": "in_breadth"})(),
            "child": _child("small"),
            "inst": _instance("What is 2 + 2?"),
        },
    ]
    evolver._apply_evaluator(entries)

    assert entries[0]["report"].status == "evaluator_input_too_large"
    assert "child" in entries[1], "the healthy candidate survived evaluation"
    assert len(backend.seen) == 1 and len(backend.seen[0]) == 1


def test_a_genuine_backend_failure_still_aborts_the_run():
    """Dropping oversized inputs must not soften real evaluator breakage."""

    class Broken(_Backend):
        def mutate(self, tasks):
            raise RuntimeError("engine died")

    evolver = _evolver(Broken(window=12000))
    entries = [
        {
            "task": type("T", (), {"op": "in_depth"})(),
            "child": _child(),
            "inst": _instance("What is 2 + 2?"),
        }
    ]
    with pytest.raises(EvaluatorRuntimeError, match="engine died"):
        evolver._apply_evaluator(entries)


def test_without_a_declared_window_nothing_is_dropped():
    """Refuse to guess a limit and reject valid children on an invented one."""
    backend = _Backend(window=None)
    evolver = _evolver(backend)
    entries = [
        {
            "task": type("T", (), {"op": "in_depth"})(),
            "child": _child(),
            "inst": _instance("Q " * 40000),
        }
    ]
    evolver._apply_evaluator(entries)
    assert len(backend.seen) == 1


# --- the label-truth gate ------------------------------------------------


def test_a_child_whose_skill_is_unjudged_is_rejected():
    """The declared SKILL is a claim the model made about its own output.

    `_validate_mutation_contract` only checks the label CHANGED. A sample run
    produced sum-plus-product of six listed integers declaring SKILL="counting"
    and it passed every gate, so silence about necessity has to read as failure.
    """
    from rq_evolve.prompts import parse_evaluator_verdict

    coherent = "reason: the statement is consistent\nverdict: VALID"
    assert parse_evaluator_verdict(coherent)[0] is True
    valid, reason = parse_evaluator_verdict(coherent, require_skill=True)
    assert valid is False
    # The rejection must carry what the evaluator actually said. 11 of 17
    # rejections in one run were silent, and the reason string overwrote the
    # output, so there was no way to tell a false label from a format failure.
    assert "no skill_required line" in reason
    assert "the statement is consistent" in reason


def test_a_silent_rejection_preserves_off_format_output_too():
    """The unparseable case is exactly the one worth seeing."""
    from rq_evolve.prompts import parse_evaluator_verdict

    json_reply = '```json\n{"reason": "looks fine", "verdict": "VALID"}\n```'
    valid, reason = parse_evaluator_verdict(json_reply, require_skill=True)
    assert valid is False
    assert "json" in reason.lower()


def test_an_explicit_no_is_rejected_and_an_explicit_yes_passes():
    from rq_evolve.prompts import parse_evaluator_verdict

    said_no = (
        "reason: a single formula answers it\n"
        "skill_required: NO\nverdict: VALID"
    )
    valid, reason = parse_evaluator_verdict(said_no, require_skill=True)
    assert valid is False
    assert "not required by the visible problem" in reason

    said_yes = "reason: the invariant decides it\nskill_required: YES\nverdict: VALID"
    assert parse_evaluator_verdict(said_yes, require_skill=True)[0] is True


def test_the_candidates_own_skill_definition_travels_with_it():
    """The evaluator cannot judge a label it has no definition for."""
    from rq_evolve.prompts import build_evaluator_task, skill_definition

    child = _child()  # SKILL = "counting"
    task = build_evaluator_task(child, "How many?", answer_text="42")
    user = task.messages[1]["content"]
    assert skill_definition("counting") in user
    assert "skill_required: YES only if" in user
    assert "Example 1" in user, "format shots must be injected"


def test_seeds_are_exempt_from_the_skill_gate():
    """A seed's label is hand-written and trusted; a child's is a model claim."""
    backend = _Backend(window=12000, reply="reason: fine\nverdict: VALID")
    evolver = _evolver(backend)
    seed = _child()
    seed.metadata.pop("op", None)  # seeds carry no mutation op
    entries = [
        {
            "task": type("T", (), {"op": "seed"})(),
            "child": seed,
            "inst": _instance("What is 2 + 2?"),
        }
    ]
    evolver._apply_evaluator(entries)
    assert "child" in entries[0], "a seed passes on coherence alone"


# --- label spelling vs judgement -----------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        "reason: fine\nskill_required: YES\nverdict: VALID",
        "Reason: fine\nSkill_Required: YES\nVerdict: VALID",
        "- reason: fine\n- skill_required: YES\n- verdict: VALID",
        "**reason**: fine\n**skill_required**: YES\n**verdict**: VALID",
        "Let's evaluate step by step.\n\nreason: fine\nskill_required: YES\nverdict: VALID",
    ],
)
def test_label_spelling_does_not_decide_the_verdict(reply):
    """A base model varies the label far more than the judgement.

    An exact-prefix match on `skill_required:` discarded 160 replies in one run
    whose reasoning was sound -- one wrote "Reason:", another led with a bullet,
    another thought aloud first. Reading the label leniently keeps those; the
    field VALUES stay strict.
    """
    from rq_evolve.prompts import parse_evaluator_verdict

    assert parse_evaluator_verdict(reply, require_skill=True)[0] is True


def test_lenient_labels_do_not_loosen_the_values():
    from rq_evolve.prompts import parse_evaluator_verdict

    for reply in (
        "Reason: shortcut exists\nSkill_Required: NO\nVerdict: VALID",
        "- reason: bad\n- skill_required: YES\n- verdict: INVALID",
        "**reason**: fine\n**verdict**: VALID",  # no skill line at all
    ):
        assert parse_evaluator_verdict(reply, require_skill=True)[0] is False


def test_the_shots_demonstrate_every_field_the_evaluator_must_emit():
    """The shots are the format contract; a field only named in prose is not one.

    skill_required was added to the instructions but not to the examples, and
    the evaluator then omitted it on 160 of 320 candidates.
    """
    from rq_evolve.prompts import _load_evaluator_shots

    shots = _load_evaluator_shots()
    for field in ("reason:", "skill_required:", "verdict:"):
        assert field in shots, field
    assert shots.count("skill_required:") == shots.count("verdict:")


def test_ast_contract_only_accepts_the_three_modes():
    with pytest.raises(ValueError, match="ast_contract"):
        EvolutionConfig(ast_contract="bogus")


def test_turning_off_both_gates_is_refused():
    """With use_evaluator off and ast_contract off, nothing between a mutation
    and the archive asks whether the statement determines the answer -- only
    that the program runs and returns an integer. That configuration produced
    an archive where 20 of 26 champions had a visible defect.
    """
    with pytest.raises(ValueError, match="use_evaluator"):
        EvolutionConfig(use_evaluator=False, ast_contract="off")

    # The supported combinations stay constructible.
    assert EvolutionConfig(use_evaluator=False, ast_contract="enforce")
    assert EvolutionConfig(use_evaluator=False, ast_contract="shadow")
    assert EvolutionConfig(use_evaluator=True, ast_contract="off")
