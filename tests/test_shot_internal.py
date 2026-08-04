import numpy as np
import pytest

from rq_evolve.expansion_trajectory import (
    compute_stalt,
    compute_stalt_from_transitions,
    spatiotemporal_transitions,
)
from rq_evolve.shot_internal import (
    build_solver_messages,
    canonical_shot_solution,
    conversation_hash,
    summarize_shot_records,
)


def _linear_row():
    return {
        "seed": 0,
        "problem": (
            "The following consistent integer linear system does not determine "
            "x, y, and z individually:\n"
            "(-3)x + (-1)y + (1)z = 3\n"
            "(6)x + (4)y + (2)z = 6\n"
            "Determine the uniquely fixed value of x + y + z. "
            "State only the integer."
        ),
        "answer": "3",
        "answer_correct": True,
        "necessity_holds": True,
        "valid": True,
        "necessity_facts": {
            "rowspace_weights": ["1/3", "1/3"],
            "aggregate_multiplier": 3,
        },
    }


def _modular_row():
    return {
        "seed": 0,
        "problem": (
            "Let x, y, and z be residue classes satisfying both congruences:\n"
            "6x + 3y + 0z is congruent to 3 modulo 7\n"
            "6x + 2y + 5z is congruent to 0 modulo 7\n"
            "Determine the unique residue of x + y + z modulo 7. "
            "State only its least nonnegative integer."
        ),
        "answer": "2",
        "answer_correct": True,
        "necessity_holds": True,
        "valid": True,
        "necessity_facts": {"multiplier": 5},
    }


def test_canonical_solutions_are_verified_and_condition_blind():
    linear = canonical_shot_solution(
        "linear_system_aggregate",
        _linear_row(),
    )
    assert "3(x+y+z)" in linear
    assert r"\boxed{3}" in linear

    modular = canonical_shot_solution(
        "modular_linear_system_aggregate",
        _modular_row(),
    )
    assert "inverse of 5 modulo 7 is 3" in modular
    assert r"\boxed{2}" in modular


def test_one_shot_messages_contain_no_condition_or_target_answer_leak():
    row = _linear_row()
    solution = canonical_shot_solution("linear_system_aggregate", row)
    target = "A separate held-out problem whose answer is not supplied."
    first = build_solver_messages(
        target,
        shot_problem=row["problem"],
        shot_solution=solution,
    )
    second = build_solver_messages(
        target,
        shot_problem=row["problem"],
        shot_solution=solution,
    )

    assert [message["role"] for message in first] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert first[-1]["content"] == target
    assert "plain" not in str(first).lower()
    assert "reasoning_shot" not in str(first).lower()
    assert conversation_hash(first) == conversation_hash(second)


def test_user_context_shot_uses_no_assistant_message():
    row = _linear_row()
    solution = canonical_shot_solution("linear_system_aggregate", row)
    target = "Solve a separate held-out target."

    messages = build_solver_messages(
        target,
        shot_problem=row["problem"],
        shot_solution=solution,
        shot_presentation="user_context",
    )

    assert [message["role"] for message in messages] == ["system", "user"]
    user_context = messages[1]["content"]
    assert row["problem"] in user_context
    assert solution in user_context
    assert target in user_context
    assert "Example problem:" in user_context
    assert "Example solution:" in user_context
    assert "Target problem:" in user_context


def test_streamed_transition_stalt_matches_full_hidden_tensor():
    rng = np.random.default_rng(7)
    hidden = rng.normal(size=(9, 5, 6))
    full = compute_stalt(hidden, tau=0.8)
    delta_time, delta_layer = spatiotemporal_transitions(hidden)
    streamed = compute_stalt_from_transitions(
        delta_time,
        delta_layer[1:],
        tau=0.8,
    )

    for key in (
        "stalt",
        "temporal_path_length",
        "mean_unweighted_temporal_amplitude",
        "mean_unweighted_layer_amplitude",
        "mean_layer_concentration_hhi",
    ):
        assert streamed[key] == full[key]
    np.testing.assert_allclose(
        streamed["token_wise_amplitude"],
        full["token_wise_amplitude"],
    )
    np.testing.assert_allclose(
        streamed["layer_saliency"],
        full["layer_saliency"],
    )


def test_shot_summary_reports_plain_reasoning_internal_contrasts():
    records = []
    for condition, correct, stalt in (
        ("no_shot", False, 1.0),
        ("plain_shot", True, 1.2),
        ("reasoning_shot", True, 1.5),
    ):
        records.append(
            {
                "operator": "in_depth",
                "condition": condition,
                "family_variant": (
                    None if condition == "no_shot" else "balanced"
                ),
                "correct": correct,
                "stalt": stalt,
                "temporal_path_length": stalt * 10,
                "generated_tokens": 11,
                "mean_layer_concentration_hhi": 0.2,
                "prompt_last_cosine_to_no_shot": 1.0,
                "conversation_hash": condition,
            }
        )
    summary = summarize_shot_records(records)

    assert len(summary["rows"]) == 3
    contrast = summary["contrasts"][0]
    assert contrast["accuracy_reasoning_minus_plain"] == 0.0
    assert contrast["stalt_reasoning_minus_plain"] == pytest.approx(0.3)
    assert contrast["reasoning_accuracy_gain_over_no_shot"] == 1.0
