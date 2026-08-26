"""Regression: all-rejected rollout groups must NEVER zero a program's scores.

A transient infra failure (chunk timeout, worker error) during
reevaluate_champions / bootstrap previously flowed into _score_from_rollouts,
which mutates s_hat/u_score/rq_score IN PLACE -- evicting a champion from its
true niche (the stale-s_hat archive-pollution failure mode). evaluate_instances
now yields None for those groups and callers keep prior scores.
"""

from types import SimpleNamespace
from dataclasses import asdict

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.backends import PendingRollouts, RolloutRecord
from rq_evolve.config import ArchiveConfig, EvolutionConfig
from rq_evolve.evolution import RQEvolver
from rq_evolve.openai_evaluator import (
    EvaluatorConfigurationError,
    EvaluatorRuntimeError,
    load_project_dotenv,
    validate_openai_evaluator_environment,
)
from rq_evolve.program import ProblemInstance, ProblemProgram
from rq_evolve.prompts import MUTATION_OP, build_judge_messages


class ScriptedBackend:
    """Returns pre-scripted grouped rollouts, one group per instance."""

    def __init__(self, grouped):
        self.grouped = grouped

    def begin_session(self):
        pass

    def end_session(self):
        pass

    def sync_weights(self):
        pass

    def mutate(self, tasks):
        return [None] * len(tasks)

    def generate_rollouts(self, instances, n_rollouts):
        return PendingRollouts(
            instances=list(instances), n_rollouts=n_rollouts, grouped=self.grouped
        )

    def finalize_rollouts(self, pending):
        return pending.grouped

    def rollout(self, instances, n_rollouts):
        return self.grouped


def _rejected(reason="timeout"):
    return RolloutRecord(
        response="", predicted_answer=None, correct=False, entropy=0.0,
        status="rejected", reject_reason=reason,
    )


def _accepted(correct, entropy=1.0):
    return RolloutRecord(
        response="x", predicted_answer="4", correct=correct, entropy=entropy,
    )


def _evolver(grouped, eval_seeds=1):
    """One seed per program by default, so scripted rollout groups line up 1:1.

    These tests are about how a group of rollouts is HANDLED (rejected, mixed,
    contaminated), not about the n x m aggregation, so the seed axis is held at
    one instance and the m axis is whatever the script provides.
    """
    archive = MAPElitesArchive(**asdict(ArchiveConfig()))
    config = EvolutionConfig(eval_seeds=eval_seeds)
    return RQEvolver(
        archive=archive, backend=ScriptedBackend(grouped), evolution_config=config
    )


def _program(pid, s_hat=0.7, h=1.2, rq=0.25):
    # Labelled, because the judge gate compares its own verdict against these.
    program = ProblemProgram(
        source_code=(
            "def generate(seed):\n"
            '    return f"what is {seed} plus four?", "4"\n\n\n'
            'GROUP = "algebra"\n'
            'SKILL = "counting"\n'
        ),
        program_id=pid,
    )
    program.s_hat, program.u_score, program.rq_score = s_hat, h, rq
    return program


def test_all_rejected_group_yields_none_and_keeps_scores():
    champion = _program("champ")
    evolver = _evolver([[_rejected(), _rejected()]])
    results = evolver.evaluate_programs([champion])
    assert results == [None]
    # prior scores untouched -- the champion is not zeroed out of its niche
    assert champion.s_hat == 0.7
    assert champion.u_score == 1.2
    assert champion.rq_score == 0.25
    assert any(e["event"] == "eval_rollout_failed" for e in evolver.events)


def test_mixed_groups_score_only_the_healthy_one():
    ok_prog, bad_prog = _program("ok"), _program("bad", s_hat=0.5, h=2.0, rq=0.5)
    evolver = _evolver(
        [
            [_accepted(True), _accepted(False)],       # healthy: s_hat 0.5
            [_rejected("worker_error"), _rejected("worker_error")],
        ]
    )
    results = evolver.evaluate_programs([ok_prog, bad_prog])
    assert results[0] is not None and results[0].s_hat == 0.5
    assert results[1] is None
    assert bad_prog.s_hat == 0.5 and bad_prog.u_score == 2.0  # untouched


def test_rejected_samples_excluded_from_p_hat():
    prog = _program("mixed")
    # 2 accepted (1 correct) + 2 rejected -> s_hat over ACCEPTED only = 0.5
    evolver = _evolver([[_accepted(True), _accepted(False), _rejected(), _rejected()]])
    result = evolver.evaluate_programs([prog])[0]
    assert result is not None
    assert result.s_hat == 0.5
    assert result.num_rollouts == 2


def test_p_hat_regrades_cleaned_first_conversation():
    prog = _program("cleaned")
    contaminated = RolloutRecord(
        response=(
            r"Correct solution: \boxed{4}."
            "\nAssistant: user\nUnrelated problem. "
            r"\boxed{9}"
        ),
        predicted_answer="9",
        correct=False,
        entropy=1.0,
    )
    evolver = _evolver([[contaminated]])

    result = evolver.evaluate_programs([prog])[0]

    assert result is not None
    assert result.s_hat == 1.0
    assert result.num_correct == 1


def test_openai_judge_rejects_when_it_fails_closed(monkeypatch):
    def fake_openai(messages, config):
        assert config.model == "gpt-5.4-mini"
        assert messages[0]["role"] == "system"
        return "GROUP: none\nSKILL: none\nFAILURE_REASON: disconnected condition"

    monkeypatch.setattr("rq_evolve.evolution.evaluate_messages_with_openai", fake_openai)
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(evaluator_provider="openai")
    child = _program("child")
    entry = {
        "task": type("Task", (), {"op": MUTATION_OP})(),
        "child": child,
        "inst": ProblemInstance(problem="p", answer="4", program_id="child", seed=0),
    }
    evolver._apply_judge([entry])
    assert entry["report"].status == "judge_rejected"
    assert entry["report"].child_id == "child"


def test_judge_receives_the_problem_and_answer_only():
    messages = build_judge_messages("Compute 2+2.", "4")
    content = messages[-1]["content"]
    assert "Compute 2+2." in content
    assert "4" in content
    assert "def generate" not in content


def test_openai_evaluator_preserves_target_order(monkeypatch):
    seen = []

    def fake_openai(messages, config):
        seen.append(messages[1]["content"])
        if "second" in messages[1]["content"]:
            return "GROUP: algebra\nSKILL: counting\nFAILURE_REASON: none"
        return "GROUP: none\nSKILL: none\nFAILURE_REASON: disconnected condition"

    monkeypatch.setattr("rq_evolve.evolution.evaluate_messages_with_openai", fake_openai)
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(
        evaluator_provider="openai",
        evaluator_concurrency=2,
    )
    entries = []
    for pid, problem in (("first", "first problem"), ("second", "second problem")):
        entries.append(
            {
                "task": type("Task", (), {"op": MUTATION_OP})(),
                "child": _program(pid),
                "inst": ProblemInstance(
                    problem=problem,
                    answer="4",
                    program_id=pid,
                    seed=0,
                ),
            }
        )

    evolver._apply_judge(entries)
    assert entries[0]["report"].status == "judge_rejected"
    assert "child" in entries[1]
    assert len(seen) == 2


def test_openai_evaluator_error_aborts_evolution(monkeypatch):
    def fake_openai(messages, config):
        raise RuntimeError("missing OPENAI_API_KEY")

    monkeypatch.setattr("rq_evolve.evolution.evaluate_messages_with_openai", fake_openai)
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(evaluator_provider="openai")
    child = _program("child")
    entry = {
        "task": type("Task", (), {"op": MUTATION_OP})(),
        "child": child,
        "inst": ProblemInstance(problem="p", answer="4", program_id="child", seed=0),
    }
    import pytest

    with pytest.raises(EvaluatorRuntimeError, match="OPENAI_API_KEY"):
        evolver._apply_judge([entry])


def test_openai_evaluator_missing_key_fails_preflight(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    import pytest

    with pytest.raises(EvaluatorConfigurationError, match="OPENAI_API_KEY"):
        validate_openai_evaluator_environment()


def test_project_dotenv_loads_key_without_logging_or_overwriting(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=from-dotenv\nOTHER_VALUE='hello'\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OTHER_VALUE", "from-shell")
    load_project_dotenv(tmp_path)
    assert __import__("os").environ["OPENAI_API_KEY"] == "from-dotenv"
    assert __import__("os").environ["OTHER_VALUE"] == "from-shell"


def test_reevaluate_champions_can_be_switched_off():
    """Champion rescoring is the archive's dominant sink -- across three 4B arms
    it removed 0.86-0.94 champions per insertion, leaving +19 net champions out
    of 268 insertions. The flag exists so that can be measured, not assumed.
    """
    from unittest.mock import patch

    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.config import EvolutionConfig
    from rq_evolve.evolution import RQEvolver

    for enabled in (True, False):
        evolver = RQEvolver(
            archive=MAPElitesArchive(),
            backend=SimpleNamespace(
                sync_weights=lambda: None,
                begin_session=lambda: None,
                end_session=lambda: None,
            ),
            evolution_config=EvolutionConfig(
                reevaluate_champions=enabled, inner_iterations=0
            ),
        )
        with patch.object(evolver, "reevaluate_champions") as spy:
            try:
                evolver.run_outer_iteration(0)
            except Exception:
                pass  # the stub backend cannot complete a batch; the spy is the point
            assert spy.called is enabled, enabled


def test_a_child_rejected_once_is_not_re_evaluated():
    """Mutation regenerates the same source; re-judging it teaches nothing.

    program_id is md5 of the source and the judge is deterministic at
    temperature 0, so a repeat is guaranteed the same verdict while still
    costing a 5-seed execution and a judge call. In a two-iteration probe one
    program came back 13 times and 34% of all slots went to repeats.
    """
    from rq_evolve.evolution import CandidateReport, RQEvolver

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver._memoize_rejections(
        [
            CandidateReport(status="verify_failed", op="mutate", child_id="aaa",
                            reason="assert fired"),
            CandidateReport(status="judge_rejected", op="mutate", child_id="bbb",
                            reason="label mismatch"),
            CandidateReport(status="no_code", op="mutate", child_id="ccc"),
        ]
    )
    assert set(evolver.rejected_children) == {"aaa", "bbb", "ccc"}
    assert "assert fired" in evolver.rejected_children["aaa"]


def test_policy_dependent_rejections_are_never_memoized():
    """A program nobody can solve TODAY must get another look tomorrow.

    s_hat_zero / rq_zero / rejected_non_elite / rollout_failed are verdicts
    about the current policy and the current occupant of the cell, not about
    the source. Memoizing them would make the archive monotone in a way the
    design does not intend -- the whole point of re-scoring is that a
    champion's fitness moves as the solver improves.
    """
    from rq_evolve.evolution import CandidateReport, RQEvolver

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver._memoize_rejections(
        [
            CandidateReport(status=status, op="mutate", child_id=f"id_{status}")
            for status in (
                "s_hat_zero",
                "rq_zero",
                "rejected_non_elite",
                "rollout_failed",
                "inserted",
            )
        ]
    )
    assert evolver.rejected_children == {}


def test_a_repeat_inside_one_batch_is_not_executed_twice():
    """32 parents drawn with replacement from ~10 cells produce repeats WITHIN
    a batch, which the cross-iteration memo cannot see: it is written when
    reports are finalized, after the batch is over. Measured 11 of 32 slots in
    one iteration. The repeat must skip the 5-seed execution, not just the
    judge, so the check lives beside the memo rather than in the caller.
    """
    from rq_evolve.evolution import RQEvolver
    from rq_evolve.prompts import MUTATION_OP, MutationTask
    from rq_evolve.program import ProblemProgram

    source = (
        "import random\n\n\n"
        "def generate(seed):\n"
        '    return f"What is {seed} + 1?", str(seed + 1)\n\n\n'
        'GROUP = "algebra"\nSKILL = "invariant"\n'
    )
    parent = ProblemProgram(source_code=source)
    # The guard reads only task.op and task.parent; which stage produced the
    # task is irrelevant to in-batch dedup, so build the task directly.
    task = MutationTask(op=MUTATION_OP, prompt="", parent=parent)

    from rq_evolve.config import EvolutionConfig

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.rejected_children = {}
    evolver.evolution_config = EvolutionConfig()
    executed = []
    evolver.verify_program = lambda program, **kw: (executed.append(program.program_id), (None, "x"))[1]

    output = f"```python\n{source}```"

    # extract_generator_code normalizes the block, so the id is whatever the
    # extracted source hashes to -- take it from the call, not from `source`.
    first, _, _, _ = evolver._make_child_from_output(task, output, in_flight=set())
    child_id = first.program_id
    assert executed == [child_id], "the first occurrence must be executed"

    child, inst, reason, src = evolver._make_child_from_output(
        task, output, in_flight={child_id}
    )
    assert executed == [child_id], "the repeat must NOT be executed again"
    assert inst is None and src is None
    assert "duplicate" in reason


def test_two_stage_mutation_returns_the_single_call_shape():
    """Stage 1 writes the problem, stage 2 its generator, and the pair comes
    back as (tasks, outputs) so retries, judging, scoring and reporting are
    untouched. A parent whose stage-1 reply does not parse yields no output,
    which the existing path already reports as a failed mutation."""
    from rq_evolve.config import EvolutionConfig
    from rq_evolve.evolution import RQEvolver
    from rq_evolve.program import ProblemProgram

    parent = ProblemProgram(
        source_code=(
            "import random\n\n\n"
            "def generate(seed):\n"
            '    problem = f"Count to {seed}. State only the integer."\n'
            '    return problem, "1"\n\n\n'
            'GROUP = "algebra"\nSKILL = "invariant"\n'
        )
    )
    child_code = (
        "```python\nimport random\n\n\n"
        "def generate(seed):\n"
        '    problem = f"How many? {seed} State only the integer."\n'
        '    return problem, "2"\n```'
    )
    plan = ("STRUCTURAL MUTATION: the target changes\n"
            "CHILD FAMILY: How many? State only the integer.\n"
            "GROUP: geometry\nSKILL: casework\n")

    calls = []

    class _Backend:
        def mutate(self, tasks):
            calls.append([t.stage for t in tasks])
            if tasks[0].stage == "family":
                return [plan, "nothing parseable here"]
            return [child_code]

    evolver = RQEvolver.__new__(RQEvolver)
    evolver.evolution_config = EvolutionConfig(two_stage_mutation=True)
    evolver.backend = _Backend()

    tasks, outputs, _ = evolver._mutate_in_two_stages([parent, parent])
    assert calls == [["family", "family"], ["generator"]], calls
    assert len(tasks) == len(outputs) == 2
    # The parsed one carries stage 1's labels, stapled on rather than asked of
    # stage 2 -- the file's tail is the one thing every prompt variant lost.
    assert 'GROUP = "geometry"' in outputs[0]
    assert 'SKILL = "casework"' in outputs[0]
    assert "algebra" not in outputs[0] and "invariant" not in outputs[0]
    # The unparsed one is simply absent, not a half-built child.
    assert outputs[1] is None
