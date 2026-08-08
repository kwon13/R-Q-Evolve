import json

import pytest

from rq_evolve.evolved_performance import (
    EvolutionEvent,
    HighRQCandidate,
    benchmark_sha256,
    build_concept_change_rows,
    evolution_state_at_step,
    instance_sha256,
    load_benchmark,
    load_evolution_events,
    normalize_problem,
    select_interval_high_rq,
    select_prominent_high_rq,
    summarize_scored_rows,
)


def _benchmark_rows():
    rows = []
    for index, (program, seed, problem, answer) in enumerate(
        [
            ("a", 100, "What is 1 + 1?", "2"),
            ("a", 101, "What is 2 + 2?", "4"),
            ("b", 100, "What is 3 + 3?", "6"),
            ("b", 101, "What is 4 + 4?", "8"),
        ]
    ):
        signature = instance_sha256(problem, answer)
        rows.append(
            {
                "benchmark": "evolved_performance_seed_id_v1",
                "index": index,
                "sample_id": f"{program}:{index}",
                "program_name": program,
                "program_id": f"pid-{program}",
                "program_sha256": f"sha-{program}",
                "seed": seed,
                "problem": problem,
                "answer": answer,
                "instance_sha256": signature,
            }
        )
    return rows


def test_problem_and_benchmark_hashes_are_stable_under_formatting():
    assert normalize_problem("  a\n  b  ") == "a b"
    assert instance_sha256("a  b", "2") == instance_sha256("a\nb", "2")
    rows = _benchmark_rows()
    assert benchmark_sha256(rows) == benchmark_sha256([dict(row) for row in rows])


def test_load_benchmark_validates_hash_and_duplicates(tmp_path):
    rows = _benchmark_rows()
    jsonl = tmp_path / "benchmark.jsonl"
    manifest = tmp_path / "manifest.json"
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest.write_text(
        json.dumps(
            {
                "benchmark_sha256": benchmark_sha256(rows),
                "num_examples": len(rows),
            }
        ),
        encoding="utf-8",
    )
    loaded, _ = load_benchmark(jsonl, manifest)
    assert loaded == rows

    rows[1]["sample_id"] = rows[0]["sample_id"]
    jsonl.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest.write_text(
        json.dumps(
            {
                "benchmark_sha256": benchmark_sha256(rows),
                "num_examples": len(rows),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_benchmark(jsonl, manifest)


def test_score_is_macro_average_across_seed_programs():
    scored = [
        {"program_name": "a", "group": "algebra", "skill": "casework", "correct": True},
        {"program_name": "a", "group": "algebra", "skill": "casework", "correct": False},
        {"program_name": "b", "group": "sequence", "skill": "induction", "correct": True},
        {"program_name": "b", "group": "sequence", "skill": "induction", "correct": True},
    ]
    summary = summarize_scored_rows(scored)
    assert summary["correct"] == 3
    assert summary["micro_accuracy"] == pytest.approx(0.75)
    assert summary["macro_accuracy"] == pytest.approx(0.75)
    assert summary["score_percent"] == pytest.approx(75.0)
    assert summary["per_program"]["a"]["group"] == "algebra"
    assert summary["per_concept"]["algebra/casework"]["num_examples"] == 2
    assert summary["per_concept"]["sequence/induction"]["accuracy"] == 1.0


def test_concept_changes_aggregate_generators_with_the_same_labels():
    results = [
        {
            "global_step": 0,
            "score_percent": 50.0,
            "per_program": {},
            "per_concept": {
                "algebra/casework": {
                    "accuracy": 0.50,
                    "group": "algebra",
                    "skill": "casework",
                }
            },
        },
        {
            "global_step": 32,
            "score_percent": 60.0,
            "per_program": {},
            "per_concept": {
                "algebra/casework": {
                    "accuracy": 0.75,
                    "group": "algebra",
                    "skill": "casework",
                }
            },
        },
    ]
    changes = build_concept_change_rows(results)
    assert changes[0]["improved"][0]["program_name"] == "algebra/casework"
    assert changes[0]["improved"][0]["delta_pp"] == pytest.approx(25.0)


def test_concept_changes_rank_positive_accuracy_deltas():
    results = [
        {
            "global_step": 0,
            "score_percent": 50.0,
            "per_program": {
                "a": {"accuracy": 0.5, "group": "algebra", "skill": "casework"},
                "b": {"accuracy": 0.5, "group": "sequence", "skill": "induction"},
            },
        },
        {
            "global_step": 32,
            "score_percent": 55.0,
            "per_program": {
                "a": {"accuracy": 0.7, "group": "algebra", "skill": "casework"},
                "b": {"accuracy": 0.4, "group": "sequence", "skill": "induction"},
            },
        },
    ]
    changes = build_concept_change_rows(results)
    assert len(changes) == 1
    assert changes[0]["eps_delta_pp"] == pytest.approx(5.0)
    assert changes[0]["improved"][0]["program_name"] == "a"
    assert changes[0]["improved"][0]["delta_pp"] == pytest.approx(20.0)
    assert changes[0]["declined"][0]["program_name"] == "b"
    assert changes[0]["declined"][0]["delta_pp"] == pytest.approx(-10.0)


def test_load_evolution_events_joins_outer_metrics_to_global_steps(tmp_path):
    archive = tmp_path / "rq_archive"
    archive.mkdir()
    evolution_rows = [
        {
            "iteration": 1,
            "metrics": {"outer_iteration": 1, "attempted": 32, "inserted": 3},
        },
        {
            "iteration": 2,
            "metrics": {"outer_iteration": 2, "attempted": 30, "inserted": 4},
        },
    ]
    rollout_rows = [
        {"iteration": 1, "source_checkpoint": "/m@global_step_3"},
        {"iteration": 2, "source_checkpoint": "/m@global_step_7"},
    ]
    (archive / "evolution_log.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in evolution_rows), encoding="utf-8"
    )
    (archive / "rollout_metrics.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rollout_rows), encoding="utf-8"
    )
    events = load_evolution_events(tmp_path)
    assert events == [
        EvolutionEvent(1, 3, 32, 3, 32, 3),
        EvolutionEvent(2, 7, 30, 4, 62, 7),
    ]
    assert evolution_state_at_step(events, 0) == (None, 0, 0)
    assert evolution_state_at_step(events, 6) == (1, 32, 3)
    assert evolution_state_at_step(events, 7) == (2, 62, 7)


def test_selects_highest_inserted_rq_between_checkpoint_outer_states():
    events = [
        EvolutionEvent(1, 3, 32, 3, 32, 3),
        EvolutionEvent(2, 7, 32, 4, 64, 7),
        EvolutionEvent(3, 11, 32, 2, 96, 9),
    ]
    candidates = [
        HighRQCandidate(1, 3, "low", 2.0, 0.5, 8.0, "in_depth", "x"),
        HighRQCandidate(1, 3, "high", 9.0, 0.5, 36.0, "in_depth", "y"),
        HighRQCandidate(2, 7, "middle", 7.0, 0.4, 29.2, "in_breadth", "z"),
        HighRQCandidate(3, 11, "last", 8.0, 0.6, 33.3, "in_depth", "q"),
    ]
    selected = select_interval_high_rq(candidates, [0, 8, 12], events)
    assert [(step, item.child_id) for step, item in selected] == [
        (8, "high"),
        (12, "last"),
    ]


def test_selects_prominent_rq_independently_of_checkpoints():
    candidates = [
        HighRQCandidate(1, 3, "o1-low", 2.0, 0.5, 8.0, "in_depth", "x"),
        HighRQCandidate(1, 3, "o1-high", 9.0, 0.5, 36.0, "in_depth", "y"),
        HighRQCandidate(2, 7, "o2", 7.0, 0.4, 29.2, "in_breadth", "z"),
        HighRQCandidate(3, 11, "o3", 20.0, 0.6, 33.3, "in_depth", "q"),
        HighRQCandidate(4, 15, "o4", 12.0, 0.6, 33.3, "in_depth", "r"),
    ]
    selected = select_prominent_high_rq(
        candidates,
        quantile=0.50,
        max_count=2,
    )
    assert [item.child_id for item in selected] == ["o3", "o4"]

    # At most one candidate can represent any outer iteration.
    assert "o1-low" not in {item.child_id for item in selected}
