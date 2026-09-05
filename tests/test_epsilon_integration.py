"""Exercise config-to-archive wiring and mutation-only admission accounting."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from rq_evolve.archive import MAPElitesArchive
from rq_evolve.config import ArchiveConfig, EvolutionConfig, load_config
from rq_evolve.evolution import CandidateReport, RQEvolver


def test_epsilon_config_reaches_archive_without_changing_parent_epsilon(tmp_path):
    (tmp_path / "base.yaml").write_text(
        "archive:\n  epsilon: 0.3\n  selection_strategy: random\n",
        encoding="utf-8",
    )
    child = tmp_path / "epsilon.yaml"
    child.write_text(
        "extends: base.yaml\narchive:\n"
        "  admission_strategy: epsilon_greedy\n  admission_epsilon: 0.4\n",
        encoding="utf-8",
    )
    config = load_config(child)
    archive = MAPElitesArchive(**asdict(config.archive))
    assert archive.admission_strategy == "epsilon_greedy"
    assert archive.admission_epsilon == 0.4
    assert archive.epsilon == 0.3
    assert archive.selection_strategy == "random"
    baseline = load_config(tmp_path / "base.yaml")
    assert baseline.archive.admission_strategy == "fitness"


@pytest.mark.parametrize("invalid", [-0.1, 1.1, float("nan"), float("inf"),
                                     -float("inf"), True, False, "bad", None])
def test_config_rejects_invalid_admission_probability(invalid):
    with pytest.raises(ValueError, match="admission_epsilon"):
        ArchiveConfig(admission_strategy="epsilon_greedy", admission_epsilon=invalid)


def test_outer_metrics_and_json_log_distinguish_epsilon_from_score_wins(tmp_path):
    archive = MAPElitesArchive(admission_strategy="epsilon_greedy", admission_epsilon=0.4)
    evolver = RQEvolver(
        archive=archive,
        backend=SimpleNamespace(sync_weights=lambda: None),
        evolution_config=EvolutionConfig(
            group_size=10, inner_iterations=6, inner_iteration_batch_size=6,
            reevaluate_champions=False,
        ),
    )
    reasons = ["score_improved", "epsilon_override", "epsilon_override",
               "epsilon_rejected", "empty_cell"]
    reports = [
        CandidateReport(
            status="rejected_non_elite" if reason == "epsilon_rejected" else "inserted",
            op="mutate",
            s_hat=0.5,
            archive_decision={"admission": {
                "strategy": "epsilon_greedy", "reason": reason,
            }},
        )
        for reason in reasons
    ]
    reports.append(CandidateReport(status="verify_failed", op="mutate"))
    evolver.inner_iteration_batch = lambda _size: reports
    evolver.refresh_dataset = lambda **_kwargs: None
    metrics = evolver.run_outer_iteration(1)
    assert metrics["epsilon_score_wins"] == 1
    assert metrics["epsilon_nonwinning_candidates"] == 3
    assert metrics["epsilon_overrides"] == 2
    assert metrics["epsilon_override_rate"] == pytest.approx(2 / 3)
    evolver.append_evolution_log(tmp_path, iteration=1, metrics=metrics)
    record = json.loads((tmp_path / "evolution_log.jsonl").read_text())
    assert record["archive_admission_strategy"] == "epsilon_greedy"
    assert record["archive_admission_epsilon"] == 0.4
    assert record["training_random_order"] is False
    assert record["reports"][1]["archive_decision"]["admission"]["reason"] == "epsilon_override"
