"""Replay of the recorded mutation-comparison failure without a GPU.

The artifacts of the failing run are reproduced here as literals so the two
defects stay fixed without needing vLLM:

* ``in_breadth`` returned ``invalid_plan`` on *both* conditions with
  ``parameters[0].domain must be a non-empty string``. The plan mirrored the
  registry's own ``family_config_schema`` (``{"modulus": {"enum": [5, 7, 11]}}``)
  and wrote ``"domain": [5, 7, 11]``, so the validator rejected the registry's
  own vocabulary before the compiler ever ran.
* ``in_depth`` reached the evaluator and was rejected with
  ``target_move_required: NO`` on a problem whose answer was correct.
"""

import json

import pytest

from rq_evolve.metacognition import validate_mutation_plan
from rq_evolve.mutation_compiler import (
    CompilationStatus,
    compile_mutation_plan,
    family_contract_payload,
    validate_compiled_family_semantics,
)
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import build_evaluator_messages, parse_evaluator_verdict

# Verbatim ``02_plan_raw.txt`` payloads from
# rq_output/mutation_method_comparison_notebook/
#   09_linear_algebra_llm_seed=0_plan_temp=0.3_config=a771aa7221af44e4
RECORDED_IN_BREADTH_PLAIN_PLAN = {
    "schema_version": 5,
    "operator": "in_breadth",
    "failure_summary": None,
    "target_reasoning_move": (
        "Identifying the sum x + y + z by solving the system of equations."
    ),
    "correct_wrong_contrast": None,
    "target_concept_group": "number_theory",
    "target_concept_type": "number_theory.modular_linear_system_sum",
    "generator_family": "modular_linear_system_aggregate",
    "family_config": {"modulus": 7},
    "why_target_reasoning_move_is_necessary": (
        "The visible trigger is the need to solve a system of equations to find "
        "x + y + z. The bypass or naive route excluded by the construction is "
        "directly adding the equations without solving for individual "
        "variables. A checkable witness is the requirement to solve the system "
        "to find x + y + z."
    ),
    "preserve": [
        "The structure of the system of equations and the requirement to find "
        "x + y + z."
    ],
    "change": ["Changing the domain to modular arithmetic with a prime modulus."],
    "forbidden_changes": [
        "Changing the structure of the system of equations or the requirement "
        "to find x + y + z."
    ],
    # The field that killed the run: a list, mirroring the registry enum.
    "parameters": [{"name": "modulus", "domain": [5, 7, 11]}],
    "guards": [
        "necessity: The system of equations must be solved to find x + y + z.",
        "nondegeneracy: The modulus must be a prime number.",
    ],
    "max_sampling_attempts": 200,
    "answer_route": (
        "Solve the system of equations to find x + y + z, then compute the sum "
        "modulo the given modulus."
    ),
    "problem_output_contract": {
        "states": ["The system of equations in modular arithmetic."],
        "withholds": ["The answer and the intended reasoning move."],
        "closing_instruction": "State only the integer x + y + z.",
    },
    "answer_output_contract": {
        "type": "integer",
        "range": "0 to modulus - 1",
        "serialization": "str(sympy.Integer(answer))",
    },
    "predicted_pre_behavior": None,
    "predicted_post_behavior": None,
    "heldout_evaluation": (
        "Phase 1: compare pass rate on the fixed generated family before and "
        "after training. Phase 2 diagnostic: Check if the system of equations "
        "can be solved to find x + y + z."
    ),
}

RECORDED_IN_BREADTH_REASONING_PLAN = {
    **RECORDED_IN_BREADTH_PLAIN_PLAN,
    "failure_summary": (
        "The solver incorrectly adds the equations and then attempts to solve "
        "for x, y, and z individually, leading to an incorrect final answer."
    ),
    "target_reasoning_move": (
        "Adding the equations to directly find the sum x + y + z."
    ),
    "correct_wrong_contrast": (
        "The correct solution adds the equations directly to find x + y + z, "
        "while the incorrect solution attempts to solve for x, y, and z "
        "individually."
    ),
    "why_target_reasoning_move_is_necessary": (
        "The visible trigger is the direct addition of the equations to find "
        "x + y + z. The bypass or naive route is attempting to solve for x, y, "
        "and z individually, which is unnecessary and incorrect. The witness is "
        "the direct addition of the equations, which uniquely determines the "
        "requested integer."
    ),
    "guards": [
        "necessity: The sum of the coefficients in each equation must be "
        "congruent to the right-hand side modulo the modulus.",
        "necessity: The modulus must be a prime number.",
    ],
    "answer_route": (
        "Sum the coefficients of x, y, and z in the equations and take the "
        "result modulo the modulus."
    ),
    "answer_output_contract": {
        "type": "integer",
        "range": [0, 6],
        "serialization": "str(sympy.Integer(answer))",
    },
    "predicted_pre_behavior": (
        "The solver will incorrectly attempt to solve for x, y, and z "
        "individually."
    ),
    "predicted_post_behavior": (
        "The solver will correctly add the equations to find x + y + z."
    ),
    "heldout_evaluation": (
        "Phase 1: compare pass rate on the fixed generated family before and "
        "after training. Phase 2 diagnostic: Check if the solver correctly adds "
        "the equations to find x + y + z."
    ),
}

# Verbatim ``evaluator.json`` raw_output values from the same run. Both say
# "the answer is correct" yet answer NO, and one invents a "third equation" that
# the compiled two-equation problem never had.
RECORDED_EVALUATOR_OUTPUTS = (
    "reason: The problem is internally coherent and the supplied answer is "
    "correct. The system of equations does not determine x, y, and z "
    "individually, but the sum of the coefficients in each equation must equal "
    "the same multiple of x + y + z. The answer is 0, which is the correct "
    "value of x + y + z.\n\ntarget_move_required: NO\n\nverdict: VALID",
    "reason: The problem is internally coherent, and the supplied answer "
    "correctly solves the visible problem. The generator source code correctly "
    "computes the sum of x + y + z by adding the equations and dividing by the "
    "coefficient of x + y + z in the third equation.\n\n"
    "target_move_required: NO\n\nverdict: VALID",
)

# The plan the in_depth planner actually produced: it describes the *parent's*
# three-equation shape, which the two-equation compiled family never builds.
RECORDED_IN_DEPTH_PLAN_MOVE = (
    "The target reasoning move is to recognize that adding all three equations "
    "directly gives a multiple of x + y + z, which simplifies the problem "
    "significantly."
)


def _linear_parent() -> ProblemProgram:
    return ProblemProgram(
        source_code=(
            'def generate(seed):\n    return "Find an integer.", "1"\n'
            'CONCEPT_REASON = "t"\n'
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )


@pytest.mark.parametrize(
    "plan,reasoning_informed",
    [
        (RECORDED_IN_BREADTH_PLAIN_PLAN, False),
        (RECORDED_IN_BREADTH_REASONING_PLAN, True),
    ],
    ids=["plain", "reasoning"],
)
def test_recorded_in_breadth_plans_no_longer_fail_validation(
    plan: dict,
    reasoning_informed: bool,
):
    errors = validate_mutation_plan(
        plan,
        "in_breadth",
        reasoning_informed=reasoning_informed,
        parent_concept_group="algebra",
        parent_concept_type="algebra.linear_system_sum",
    )
    assert errors == []


@pytest.mark.parametrize(
    "plan",
    [RECORDED_IN_BREADTH_PLAIN_PLAN, RECORDED_IN_BREADTH_REASONING_PLAN],
    ids=["plain", "reasoning"],
)
def test_recorded_in_breadth_plans_reach_the_registered_compiler(plan: dict):
    """Previously both conditions died at invalid_plan and never compiled."""
    result = compile_mutation_plan(plan, _linear_parent(), "in_breadth")
    assert result.status is CompilationStatus.COMPILED, result.reasons
    assert result.generator_family == "modular_linear_system_aggregate"
    semantics = validate_compiled_family_semantics(result, range(5))
    assert semantics.valid, semantics.reasons


def test_domain_still_rejects_empty_and_non_scalar_enumerations():
    """Accepting enum lists must not accept junk."""
    for bad_domain in ([], [[5]], [{"a": 1}], ["  "], None, 7, True):
        plan = {
            **RECORDED_IN_BREADTH_PLAIN_PLAN,
            "parameters": [{"name": "modulus", "domain": bad_domain}],
        }
        errors = validate_mutation_plan(
            plan,
            "in_breadth",
            reasoning_informed=False,
            parent_concept_group="algebra",
            parent_concept_type="algebra.linear_system_sum",
        )
        assert any("domain" in error for error in errors), bad_domain


def test_domain_still_requires_a_parameter_name():
    plan = {
        **RECORDED_IN_BREADTH_PLAIN_PLAN,
        "parameters": [{"name": "", "domain": [5, 7, 11]}],
    }
    errors = validate_mutation_plan(
        plan,
        "in_breadth",
        reasoning_informed=False,
        parent_concept_group="algebra",
        parent_concept_type="algebra.linear_system_sum",
    )
    assert any("name" in error for error in errors)


@pytest.mark.parametrize("recorded", RECORDED_EVALUATOR_OUTPUTS)
def test_recorded_evaluator_rejections_are_still_honoured(recorded: str):
    """The gate is unchanged: a recorded NO must still reject the candidate."""
    is_valid, reason = parse_evaluator_verdict(recorded, require_target_move=True)
    assert is_valid is False
    assert "target reasoning move is not necessary" in reason


def test_in_depth_evaluator_prompt_no_longer_asks_about_the_parents_shape():
    """The mismatch that produced the NO verdicts is gone from the question.

    The recorded plan claims "adding all three equations"; the compiled family
    prints two. The contract now supplies the claim under test, and the two
    problem-shape statements no longer contradict each other in the prompt.
    """
    result = compile_mutation_plan(
        {
            "schema_version": 5,
            "operator": "in_depth",
            "generator_family": "linear_system_aggregate",
            "family_config": {},
            "target_reasoning_move": RECORDED_IN_DEPTH_PLAN_MOVE,
            "target_concept_group": "algebra",
            "target_concept_type": "algebra.linear_system_sum",
        },
        _linear_parent(),
        "in_depth",
    )
    assert result.status is CompilationStatus.COMPILED
    semantics = validate_compiled_family_semantics(result, range(5))
    assert semantics.valid, semantics.reasons
    contract = family_contract_payload(result, semantics)

    namespace: dict = {}
    exec(compile(result.source_code, "<test>", "exec"), namespace)
    problem, answer = namespace["generate"](0)
    messages = build_evaluator_messages(
        problem,
        {"schema_version": 5, "target_reasoning_move": RECORDED_IN_DEPTH_PLAN_MOVE},
        answer_text=answer,
        program_source=result.source_code,
        family_contract=contract,
    )
    user = messages[1]["content"]
    # The authoritative claim matches the two-equation construction.
    assert contract["target_reasoning_move"] in user
    # The parent-shaped prose is gone from the evaluator input entirely.
    assert RECORDED_IN_DEPTH_PLAN_MOVE not in user
    assert "all three equations" not in user
    assert "Mutation plan" not in user
    # The compiled problem really does print exactly two equations.
    assert problem.count(" = ") == 2
    # And the deterministic gate has already certified both axes, with the
    # certification visible to the evaluator.
    verification = contract["deterministic_verification"]
    assert verification["answer_oracle_agrees"] is True
    assert verification["necessity_holds"] is True
    assert "deterministic_verification" in user
    assert json.dumps(verification["seeds"]) == "[0, 1, 2, 3, 4]"


def test_plan_may_select_a_family_variant():
    """The new optional field must be accepted, and junk still rejected."""
    plan = {
        **RECORDED_IN_BREADTH_PLAIN_PLAN,
        "family_variant": "hard_inverse",
    }
    errors = validate_mutation_plan(
        plan,
        "in_breadth",
        reasoning_informed=False,
        parent_concept_group="algebra",
        parent_concept_type="algebra.linear_system_sum",
    )
    assert errors == []
    result = compile_mutation_plan(plan, _linear_parent(), "in_breadth")
    assert result.status is CompilationStatus.COMPILED
    assert result.family_variant == "hard_inverse"
    assert result.variant_selected_by_plan is True
    # The recorded plan echoes "modulus": 7, but the registry-owned variant wins
    # so the variant choice cannot be neutralized by a copied default. The
    # override is recorded rather than applied silently.
    assert result.family_config["modulus"] == 11
    assert result.family_config["multiplier_min"] == 4
    assert result.variant_overridden_plan_keys == ("modulus",)
    assert validate_compiled_family_semantics(result, range(5)).valid

    for bad in ("", "   ", 7, [], {}):
        errors = validate_mutation_plan(
            {**RECORDED_IN_BREADTH_PLAIN_PLAN, "family_variant": bad},
            "in_breadth",
            reasoning_informed=False,
            parent_concept_group="algebra",
            parent_concept_type="algebra.linear_system_sum",
        )
        assert any("family_variant" in error for error in errors), bad


def test_recorded_plans_without_a_variant_default_and_are_flagged():
    """The previous run's plans omit the field; they must still compile."""
    assert "family_variant" not in RECORDED_IN_BREADTH_PLAIN_PLAN
    result = compile_mutation_plan(
        RECORDED_IN_BREADTH_PLAIN_PLAN,
        _linear_parent(),
        "in_breadth",
    )
    assert result.status is CompilationStatus.COMPILED
    assert result.family_variant == "balanced"
    assert result.variant_selected_by_plan is False
