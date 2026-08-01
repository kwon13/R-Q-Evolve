"""Tests for belief/desire attribution as a falsifiable probe.

The v5 plan asked for 22 fields and got behaviour descriptions back. These tests
pin the two properties that make v6 different: the planner makes exactly one
judgement that needs language understanding, and every other claim is checked in
Python -- including whether the attribution predicted what the solver did.
"""

import pytest

from rq_evolve.belief_probe import (
    BELIEF_PLAN_FIELDS,
    BELIEF_SCHEMA_VERSION,
    HYPOTHESIS_CATALOG,
    asserts_individual_values,
    catalog_payload,
    check_probe_diagnosticity,
    get_hypothesis,
    hypotheses_for,
    score_belief_attribution,
    validate_belief_plan,
)
from rq_evolve.mutation_compiler import (
    CompilationStatus,
    compile_belief_probe,
    compiled_family_instances,
    validate_compiled_family_semantics,
)
from rq_evolve.program import ProblemProgram

# The trace that motivated the design, from the recorded run. The addition is
# arithmetically correct; the error is that "to eliminate x" is applied to an
# equation with no x term, so the x is silently dropped.
RECORDED_WRONG_TRACE = (
    "We need to find the solution for x, y, and z, and then add them together.\n"
    "Let's start by solving the system of equations for x, y, and z.\n"
    "1. Add the first and second equations to eliminate x:\n"
    "   (x - y + z) + (4y + 3z) = -6 + 24\n"
    "   3y + 4z = 18\n"
    "7. Substitute the equation for y into the third original equation:\n"
    "   x = -48\n   y = 150\n   z = -108\n"
    "x + y + z = -48 + 150 - 108 = 2\n"
)

_OPERATORS = (
    ("in_depth", "algebra", "algebra.linear_system_sum", "linear_system_aggregate"),
    (
        "in_breadth",
        "algebra",
        "algebra.linear_system_sum",
        "modular_linear_system_aggregate",
    ),
)


def _parent() -> ProblemProgram:
    return ProblemProgram(
        source_code=(
            'def generate(seed):\n    return "Find an integer.", "1"\n'
            'CONCEPT_REASON = "t"\n'
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )


def _plan(hypothesis_id: str, operator: str, quote: str | None):
    return {
        "schema_version": BELIEF_SCHEMA_VERSION,
        "operator": operator,
        "attributed_hypothesis": hypothesis_id,
        "evidence_quote": quote,
    }


# ---------------------------------------------------------------------------
# The burden reduction is the point.
# ---------------------------------------------------------------------------


def test_plan_asks_the_model_for_exactly_one_judgement():
    """v5 asked for 22 fields; only two here carry model-generated meaning."""
    assert len(BELIEF_PLAN_FIELDS) == 4
    model_authored = set(BELIEF_PLAN_FIELDS) - {"schema_version", "operator"}
    assert model_authored == {"attributed_hypothesis", "evidence_quote"}


@pytest.mark.parametrize("family", sorted(HYPOTHESIS_CATALOG))
def test_hypothesis_choice_is_a_small_closed_set(family: str):
    """A closed choice cannot be 'echoed defaults' the way free config was."""
    entries = catalog_payload(family)
    assert 2 <= len(entries) <= 5
    assert len({entry["hypothesis_id"] for entry in entries}) == len(entries)
    for entry in entries:
        assert entry["kind"] in {"belief", "desire"}
        # Content, not behaviour: a proposition plus the precondition it omits.
        assert entry["proposition"].strip()
        assert entry["omitted_precondition"].strip()
    # The prediction function is never exposed to the planner.
    assert all("predicted_wrong_answer" not in entry for entry in entries)


# ---------------------------------------------------------------------------
# Grounding: an attribution must be tied to the evidence, verbatim.
# ---------------------------------------------------------------------------


def test_verbatim_quote_is_accepted():
    errors = validate_belief_plan(
        _plan(
            "unweighted_sum_then_divide",
            "in_depth",
            "Add the first and second equations to eliminate x",
        ),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=RECORDED_WRONG_TRACE,
    )
    assert errors == []


def test_paraphrase_is_rejected_as_evidence():
    """The v5 plan paraphrased and mislocated the error; that must not pass."""
    errors = validate_belief_plan(
        _plan(
            "unweighted_sum_then_divide",
            "in_depth",
            "The solver incorrectly adds the equations",
        ),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=RECORDED_WRONG_TRACE,
    )
    assert any("verbatim" in error for error in errors)


def test_missing_quote_is_rejected_when_a_trace_exists():
    errors = validate_belief_plan(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=RECORDED_WRONG_TRACE,
    )
    assert any("evidence_quote" in error for error in errors)


def test_plain_condition_attributes_without_a_quote():
    """No trace means guessing from priors -- the experimental manipulation."""
    errors = validate_belief_plan(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=None,
    )
    assert errors == []


def test_plain_condition_may_not_invent_a_quote():
    errors = validate_belief_plan(
        _plan("unweighted_sum_then_divide", "in_depth", "something it never saw"),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=None,
    )
    assert any("must be null" in error for error in errors)


def test_unknown_hypothesis_lists_the_valid_choices():
    errors = validate_belief_plan(
        _plan("solver_is_bad_at_math", "in_depth", None),
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=None,
    )
    joined = "; ".join(errors)
    assert "unknown attributed_hypothesis" in joined
    for hypothesis in hypotheses_for("linear_system_aggregate"):
        assert hypothesis.hypothesis_id in joined


def test_extra_fields_are_rejected_so_the_schema_stays_small():
    plan = _plan("unweighted_sum_then_divide", "in_depth", None)
    plan["failure_summary"] = "the solver added the equations"
    errors = validate_belief_plan(
        plan,
        operator="in_depth",
        generator_family="linear_system_aggregate",
        wrong_trace=None,
    )
    assert any("unexpected plan field: failure_summary" in e for e in errors)


# ---------------------------------------------------------------------------
# The attribution drives the mutation.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator,_g,_t,family", _OPERATORS)
def test_each_hypothesis_compiles_to_its_own_verified_probe(
    operator: str,
    _g: str,
    _t: str,
    family: str,
):
    hashes: dict[str, str] = {}
    for hypothesis in hypotheses_for(family):
        result = compile_belief_probe(
            _plan(hypothesis.hypothesis_id, operator, None),
            _parent(),
            operator,
        )
        assert result.status is CompilationStatus.COMPILED, result.reasons
        assert result.family_variant == hypothesis.probe_variant
        # The registry-owned probe still passes the family's own oracles.
        semantics = validate_compiled_family_semantics(result, range(6))
        assert semantics.valid, semantics.reasons
        hashes[hypothesis.hypothesis_id] = result.source_hash
    # Distinct beliefs must not collapse onto one construction, which is the
    # failure that made the whole comparison untestable under v5.
    assert len(set(hashes.values())) >= 2, hashes


def test_planner_never_picks_the_family_or_the_variant():
    """Family and variant are derived, so defaults cannot be echoed."""
    plan = _plan("division_step_skipped", "in_depth", None)
    assert "generator_family" not in plan
    assert "family_variant" not in plan
    assert "family_config" not in plan
    result = compile_belief_probe(plan, _parent(), "in_depth")
    assert result.generator_family == "linear_system_aggregate"
    assert result.family_variant == "heavy_division"


def test_unknown_hypothesis_never_compiles():
    result = compile_belief_probe(
        _plan("not_a_hypothesis", "in_depth", None),
        _parent(),
        "in_depth",
    )
    assert result.status is CompilationStatus.INVALID_SPEC
    assert "unknown attributed_hypothesis" in "; ".join(result.reasons)


# ---------------------------------------------------------------------------
# Diagnosticity: the probe must be able to tell belief from truth.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("operator,_g,_t,family", _OPERATORS)
def test_belief_probes_discriminate_on_enough_seeds(
    operator: str,
    _g: str,
    _t: str,
    family: str,
):
    """Enough seeds, not every seed.

    A route can coincide with the correct answer by arithmetic luck, and a route
    that divides by the aggregate multiplier is silent when the division is not
    whole. Neither makes the probe uninformative, so the gate is a fraction and
    the non-discriminating seeds are excluded from scoring instead.
    """
    seeds = range(10)
    for hypothesis in hypotheses_for(family):
        result = compile_belief_probe(
            _plan(hypothesis.hypothesis_id, operator, None),
            _parent(),
            operator,
        )
        instances = compiled_family_instances(result)(seeds)
        diagnosticity = check_probe_diagnosticity(hypothesis, instances)
        assert diagnosticity.valid, (
            hypothesis.hypothesis_id,
            diagnosticity.reasons,
        )
        discriminating = set(diagnosticity.discriminating_seeds)
        assert len(discriminating) >= hypothesis.min_discriminating_seeds
        for entry in diagnosticity.per_seed:
            if entry["seed"] in discriminating:
                # Every seed that counts really can refute the attribution.
                assert entry["discriminates"] is True
                if entry["belief_predicted_answer"] is not None:
                    assert (
                        entry["belief_predicted_answer"]
                        != entry["correct_answer"]
                    )


def test_a_probe_that_cannot_discriminate_is_rejected():
    """If the belief predicts the correct answer, the probe proves nothing."""
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    # Hand-built instances where (rhs[0] + rhs[1]) / k coincides with the answer.
    instances = {
        0: ({"rhs": [8, 6], "aggregate_multiplier": 2}, "7"),
        1: ({"rhs": [4, 2], "aggregate_multiplier": 2}, "3"),
    }
    diagnosticity = check_probe_diagnosticity(hypothesis, instances)
    assert diagnosticity.valid is False
    assert diagnosticity.discriminating_seeds == ()
    assert any("cannot discriminate" in r for r in diagnosticity.reasons)


def test_a_seed_whose_prediction_is_silent_is_excluded_not_counted_as_a_miss():
    """A non-whole division carries no evidence either way."""
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    instances = {
        0: ({"rhs": [5, 2], "aggregate_multiplier": 3}, "4"),   # 7/3 -> silent
        1: ({"rhs": [9, 3], "aggregate_multiplier": 3}, "1"),   # 12/3 = 4 != 1
        2: ({"rhs": [6, 6], "aggregate_multiplier": 3}, "2"),   # 12/3 = 4 != 2
        3: ({"rhs": [8, 1], "aggregate_multiplier": 3}, "5"),   # 9/3  = 3 != 5
    }
    diagnosticity = check_probe_diagnosticity(hypothesis, instances)
    assert diagnosticity.discriminating_seeds == (1, 2, 3)
    assert diagnosticity.valid is True
    silent = next(e for e in diagnosticity.per_seed if e["seed"] == 0)
    assert silent["belief_predicted_answer"] is None
    assert silent["discriminates"] is False


def test_desire_hypotheses_are_scored_from_the_derivation():
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "full_solution_required",
    )
    assert hypothesis.kind == "desire"
    assert hypothesis.predicted_wrong_answer is None
    assert hypothesis.response_signature is not None
    # The recorded trace commits to unique x, y and z on a system that has none.
    assert asserts_individual_values(RECORDED_WRONG_TRACE) is True
    assert asserts_individual_values(
        "Adding the equations gives x + y + z = 6, so the answer is 6."
    ) is False


# ---------------------------------------------------------------------------
# Outcome: did the attribution predict what the solver actually did?
# ---------------------------------------------------------------------------


def test_attribution_is_scored_from_rollouts_without_an_llm_judge():
    """One observation per instance: the answer the policy settles on.

    Each instance below has a clear modal answer, and the three instances settle
    on different outcomes, so hits, other errors and correct answers are all
    exercised. Non-modal rollouts are sampling noise and are not scored -- that
    is the whole point of the change.
    """
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    result = compile_belief_probe(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        _parent(),
        "in_depth",
    )
    instances = compiled_family_instances(result)(range(12))
    diagnosticity = check_probe_diagnosticity(hypothesis, instances)
    scored = list(diagnosticity.discriminating_seeds)[:3]
    instances = {seed: instances[seed] for seed in scored}

    def rollout(answer, correct, nll):
        return {
            "predicted_answer": str(answer),
            "correct": correct,
            "mean_negative_logprob": nll,
        }

    rollouts = {}
    for position, seed in enumerate(scored):
        data, answer = instances[seed]
        route = hypothesis.predicted_wrong_answer(data)
        assert route is not None and str(route) != answer
        if position == 0:
            # Settles on the attributed route, and confidently.
            rollouts[seed] = [rollout(route, False, 0.08)] * 3 + [
                rollout(answer, True, 0.30)
            ]
        elif position == 1:
            # Settles on a different error, hesitantly.
            rollouts[seed] = [rollout(99999, False, 0.50)] * 3 + [
                rollout(route, False, 0.09)
            ]
        else:
            # Settles on the correct answer.
            rollouts[seed] = [rollout(answer, True, 0.25)] * 3 + [
                rollout(route, False, 0.09)
            ]

    score = score_belief_attribution(
        hypothesis,
        instances,
        rollouts,
        discriminating_seeds=scored,
    )
    assert score["scoring_basis"] == "modal_answer_per_instance"
    assert score["scored_seeds"] == sorted(scored)
    assert score["num_instances_scored"] == 3
    assert score["num_wrong"] == 2
    assert score["num_attribution_hits"] == 1
    assert score["attribution_hit_rate"] == pytest.approx(0.5)
    # Every instance had a 3-of-4 majority.
    assert score["mean_modal_support"] == pytest.approx(0.75)
    # The belief signature: confidently wrong in the predicted way.
    assert score["mean_neg_logprob_on_hits"] == pytest.approx(0.08)
    assert score["mean_neg_logprob_on_other_errors"] == pytest.approx(0.50)
    assert score["hits_more_confident_than_correct"] is True


def test_a_wrong_attribution_scores_near_zero():
    """The measure must be able to say the planner guessed badly."""
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    result = compile_belief_probe(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        _parent(),
        "in_depth",
    )
    instances = compiled_family_instances(result)(range(3))
    rollouts = {
        seed: [
            {
                "predicted_answer": "123456",
                "correct": False,
                "mean_negative_logprob": 0.3,
            }
        ]
        for seed in instances
    }
    score = score_belief_attribution(hypothesis, instances, rollouts)
    assert score["num_attribution_hits"] == 0
    assert score["attribution_hit_rate"] == 0.0


def test_hit_rate_is_undefined_rather_than_zero_without_errors():
    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    result = compile_belief_probe(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        _parent(),
        "in_depth",
    )
    instances = compiled_family_instances(result)(range(2))
    rollouts = {
        seed: [
            {
                "predicted_answer": answer,
                "correct": True,
                "mean_negative_logprob": 0.2,
            }
        ]
        for seed, (_data, answer) in instances.items()
    }
    score = score_belief_attribution(hypothesis, instances, rollouts)
    assert score["num_wrong"] == 0
    assert score["attribution_hit_rate"] is None


def test_both_conditions_get_the_same_catalog_and_instructions():
    """Only trace presence may differ; the choice set must be identical."""
    from rq_evolve.belief_probe import build_belief_plan_prompt

    common = dict(
        parent_problem="Suppose x, y, z satisfy ... Find x + y + z.",
        generator_family="linear_system_aggregate",
        operator="in_depth",
    )
    reasoning = build_belief_plan_prompt(
        **common,
        wrong_trace=RECORDED_WRONG_TRACE,
        correct_trace="Add the three equations: 4x+4y+4z=24, so x+y+z=6.",
    )
    plain = build_belief_plan_prompt(**common, wrong_trace=None)

    for hypothesis in hypotheses_for("linear_system_aggregate"):
        assert hypothesis.hypothesis_id in reasoning
        assert hypothesis.hypothesis_id in plain
        # The probe mapping is never revealed, so the planner cannot choose a
        # hypothesis for its downstream construction instead of its content.
        assert hypothesis.probe_variant not in plain
        assert hypothesis.probe_rationale not in plain
    assert "evidence_quote" in reasoning and "evidence_quote" in plain
    assert '"evidence_quote": null' in plain
    assert RECORDED_WRONG_TRACE.strip() in reasoning
    assert "No reasoning traces are available" in plain
    # Small enough that an 8B model is choosing, not drafting.
    assert len(plain) < 4000


def test_prompt_asks_for_content_not_behaviour():
    from rq_evolve.belief_probe import build_belief_plan_prompt

    prompt = build_belief_plan_prompt(
        parent_problem="p",
        generator_family="linear_system_aggregate",
        operator="in_depth",
        wrong_trace=RECORDED_WRONG_TRACE,
    )
    assert "Do not describe what the solver did" in prompt
    assert "believed" in prompt and "wanted" in prompt
    # None of the v5 behaviourist field names survive.
    for legacy in (
        "predicted_pre_behavior",
        "predicted_post_behavior",
        "failure_summary",
        "target_reasoning_move",
        "family_config",
    ):
        assert legacy not in prompt


# ---------------------------------------------------------------------------
# Gate-then-select: attribution picks the niche, R_Q alone picks the champion.
# ---------------------------------------------------------------------------


def test_eligibility_is_a_conjunction_not_a_weighted_sum():
    from rq_evolve.belief_probe import evaluate_eligibility

    hypothesis = get_hypothesis(
        "linear_system_aggregate",
        "unweighted_sum_then_divide",
    )
    result = compile_belief_probe(
        _plan("unweighted_sum_then_divide", "in_depth", None),
        _parent(),
        "in_depth",
    )
    diagnosticity = check_probe_diagnosticity(
        hypothesis,
        compiled_family_instances(result)(range(5)),
    )
    passing = evaluate_eligibility(
        family_semantics_valid=True,
        diagnosticity=diagnosticity,
        plan_errors=[],
    )
    assert passing.eligible is True
    # Eligibility carries no score, so nothing can trade off against R_Q.
    assert not hasattr(passing, "score")

    for kwargs in (
        {"family_semantics_valid": False, "family_semantics_reasons": ["bad"]},
        {"diagnosticity": None},
        {"plan_errors": ["evidence_quote must appear verbatim"]},
    ):
        base = {
            "family_semantics_valid": True,
            "diagnosticity": diagnosticity,
            "plan_errors": [],
        }
        base.update(kwargs)
        assert evaluate_eligibility(**base).eligible is False, kwargs


def test_each_attribution_gets_its_own_archive_niche():
    """A new exploration axis, so search cannot fixate on one belief."""
    from rq_evolve.archive import MAPElitesArchive
    from rq_evolve.belief_probe import attribution_niche_key

    archive = MAPElitesArchive(diversity_axis="attributed_hypothesis")

    def program(hypothesis_id: str) -> ProblemProgram:
        item = ProblemProgram(
            source_code=(
                'def generate(seed):\n    return f"Q{seed}", str(seed)\n'
                'CONCEPT_REASON = "t"\n'
                'CONCEPT_GROUP = "algebra"\n'
                'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
            )
        )
        item.metadata["attributed_hypothesis"] = hypothesis_id
        return item

    ids = [h.hypothesis_id for h in hypotheses_for("linear_system_aggregate")]
    bins = {i: archive.program_to_div_bin(program(i), "Q0") for i in ids}
    assert len(set(bins.values())) == len(ids), bins
    # Binning is deterministic, so a cell keeps its meaning across generations.
    for hypothesis_id in ids:
        assert archive.program_to_div_bin(program(hypothesis_id), "Q0") == bins[
            hypothesis_id
        ]
    assert attribution_niche_key("algebra.linear_system_sum", ids[0]).endswith(
        ids[0]
    )


def test_grid_shape_is_condition_independent():
    """Both conditions must get identical archive capacity."""
    from rq_evolve.archive import MAPElitesArchive

    first = MAPElitesArchive(diversity_axis="attributed_hypothesis")
    second = MAPElitesArchive(diversity_axis="attributed_hypothesis")
    assert len(first.grid) == len(second.grid)
    # A program carrying no attribution still lands in a real cell rather than
    # being dropped, so the plain condition is never structurally excluded.
    unattributed = ProblemProgram(
        source_code=(
            'def generate(seed):\n    return f"Q{seed}", str(seed)\n'
            'CONCEPT_REASON = "t"\n'
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )
    assert 0 <= first.program_to_div_bin(unattributed, "Q0") < first.n_div_bins


def test_rq_alone_decides_the_champion_within_a_niche():
    """Attribution chooses the cell; it must not influence who wins it."""
    from rq_evolve.archive import MAPElitesArchive

    archive = MAPElitesArchive(diversity_axis="attributed_hypothesis")

    def program(tag: str) -> ProblemProgram:
        item = ProblemProgram(
            source_code=(
                "def generate(seed):\n"
                f'    return f"{tag} problem {{seed}} value {{seed * 7}}", str(seed * 7)\n'
                'CONCEPT_REASON = "t"\n'
                'CONCEPT_GROUP = "algebra"\n'
                'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
            )
        )
        item.metadata["attributed_hypothesis"] = "unweighted_sum_then_divide"
        return item

    weak, strong = program("weak"), program("strong")
    assert archive.try_insert(weak, h_value=2.0, problem_text="weak", rq_score=0.02)
    cell = (archive.h_to_bin(2.0), archive.program_to_div_bin(weak, "weak"))
    assert archive.grid[cell].champion is weak
    # Higher R_Q takes the cell even though the attribution is identical.
    assert archive.try_insert(
        strong, h_value=2.0, problem_text="strong", rq_score=0.09
    )
    assert archive.grid[cell].champion is strong
    assert archive.grid[cell].champion_rq == pytest.approx(0.09)
