"""Regression: all-rejected rollout groups must NEVER zero a program's scores.

A transient infra failure (chunk timeout, worker error) during
reevaluate_champions / bootstrap previously flowed into _score_from_rollouts,
which mutates p_hat/h_score/rq_score IN PLACE -- evicting a champion from its
true niche (the stale-p_hat archive-pollution failure mode). evaluate_instances
now yields None for those groups and callers keep prior scores.
"""

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


def _evolver(grouped):
    archive = MAPElitesArchive(**asdict(ArchiveConfig()))
    return RQEvolver(archive=archive, backend=ScriptedBackend(grouped))


def _program(pid, p_hat=0.7, h=1.2, rq=0.25):
    program = ProblemProgram(source_code="def generate(seed): pass", program_id=pid)
    program.p_hat, program.h_score, program.rq_score, program.fitness = p_hat, h, rq, rq
    return program


def test_all_rejected_group_yields_none_and_keeps_scores():
    champion = _program("champ")
    inst = ProblemInstance(problem="p", answer="4", program_id="champ", seed=0)
    evolver = _evolver([[_rejected(), _rejected()]])
    results = evolver.evaluate_instances([champion], [inst])
    assert results == [None]
    # prior scores untouched -- the champion is not zeroed out of its niche
    assert champion.p_hat == 0.7
    assert champion.h_score == 1.2
    assert champion.rq_score == 0.25
    assert any(e["event"] == "eval_rollout_failed" for e in evolver.events)


def test_mixed_groups_score_only_the_healthy_one():
    ok_prog, bad_prog = _program("ok"), _program("bad", p_hat=0.5, h=2.0, rq=0.5)
    insts = [
        ProblemInstance(problem="a", answer="4", program_id="ok", seed=0),
        ProblemInstance(problem="b", answer="4", program_id="bad", seed=0),
    ]
    evolver = _evolver(
        [
            [_accepted(True), _accepted(False)],       # healthy: p_hat 0.5
            [_rejected("worker_error"), _rejected("worker_error")],
        ]
    )
    results = evolver.evaluate_instances([ok_prog, bad_prog], insts)
    assert results[0] is not None and results[0].p_hat == 0.5
    assert results[1] is None
    assert bad_prog.p_hat == 0.5 and bad_prog.h_score == 2.0  # untouched


def test_rejected_samples_excluded_from_p_hat():
    prog = _program("mixed")
    inst = ProblemInstance(problem="p", answer="4", program_id="mixed", seed=0)
    # 2 accepted (1 correct) + 2 rejected -> p_hat over ACCEPTED only = 0.5
    evolver = _evolver([[_accepted(True), _accepted(False), _rejected(), _rejected()]])
    result = evolver.evaluate_instances([prog], [inst])[0]
    assert result is not None
    assert result.p_hat == 0.5
    assert result.num_rollouts == 2


def test_openai_evaluator_rejects_invalid(monkeypatch):
    def fake_openai(messages, config):
        assert config.model == "gpt-5.4-mini"
        assert messages[0]["role"] == "system"
        return "reason: disconnected condition\nverdict: INVALID"

    monkeypatch.setattr("rq_evolve.evolution.evaluate_messages_with_openai", fake_openai)
    evolver = _evolver([])
    evolver.evolution_config = EvolutionConfig(evaluator_provider="openai")
    child = _program("child")
    entry = {
        "task": type("Task", (), {"op": "in_depth"})(),
        "child": child,
        "inst": ProblemInstance(problem="p", answer="4", program_id="child", seed=0),
    }
    evolver._apply_evaluator([entry])
    assert entry["report"].status == "evaluator_rejected"
    assert entry["report"].child_id == "child"


def test_openai_evaluator_preserves_target_order(monkeypatch):
    seen = []

    def fake_openai(messages, config):
        seen.append(messages[1]["content"])
        if "second" in messages[1]["content"]:
            return "reason: ok\nverdict: VALID"
        return "reason: first invalid\nverdict: INVALID"

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
                "task": type("Task", (), {"op": "in_depth"})(),
                "child": _program(pid),
                "inst": ProblemInstance(
                    problem=problem,
                    answer="4",
                    program_id=pid,
                    seed=0,
                ),
            }
        )

    evolver._apply_evaluator(entries)
    assert entries[0]["report"].status == "evaluator_rejected"
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
        "task": type("Task", (), {"op": "in_depth"})(),
        "child": child,
        "inst": ProblemInstance(problem="p", answer="4", program_id="child", seed=0),
    }
    import pytest

    with pytest.raises(EvaluatorRuntimeError, match="OPENAI_API_KEY"):
        evolver._apply_evaluator([entry])


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
