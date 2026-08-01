import json

from rq_evolve.code_utils import (
    lint_generator_source,
    lint_metacognitive_generator_source,
    lint_problem_instance,
)
from rq_evolve.mutation_compiler import (
    CompilationStatus,
    DEFAULT_MUTATION_FAMILY_REGISTRY,
    MUTATION_FAMILY_REGISTRY_VERSION,
    MutationSpec,
    compile_mutation_plan,
    compile_mutation_spec,
    registered_family_catalog,
    registered_family_descriptor,
    validate_linear_system_aggregate_instance,
    validate_modular_linear_system_aggregate_instance,
)
from rq_evolve.program import ProblemProgram


def _parent(
    group: str = "algebra",
    concept_type: str = "algebra.linear_system_sum",
) -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return "Find an integer.", "1"
CONCEPT_REASON = "test"
CONCEPT_GROUP = "{group}"
CONCEPT_TYPE = "{concept_type}"
'''
    )


def _assert_mechanical_contract(source: str) -> None:
    assert lint_generator_source(source) == []
    assert (
        lint_metacognitive_generator_source(
            source,
            require_canonical_instance_data=True,
        )
        == []
    )


def test_registry_catalog_and_forced_parent_operator_descriptors():
    catalog = DEFAULT_MUTATION_FAMILY_REGISTRY.catalog()
    assert catalog["version"] == MUTATION_FAMILY_REGISTRY_VERSION
    assert {
        entry["generator_family"] for entry in catalog["families"]
    } == {
        "linear_system_aggregate",
        "modular_linear_system_aggregate",
    }

    parent = _parent()
    depth = registered_family_descriptor(parent, "in_depth")
    breadth = registered_family_descriptor(parent, "in_breadth")
    assert depth is not None
    assert depth.generator_family == "linear_system_aggregate"
    assert depth.default_config["coefficient_min"] == -3
    assert breadth is not None
    assert breadth.generator_family == "modular_linear_system_aggregate"
    assert breadth.concept_group == "number_theory"

    prompt_catalog = json.loads(registered_family_catalog(parent, "in_breadth"))
    assert prompt_catalog["supported"] is True
    assert (
        prompt_catalog["family"]["generator_family"]
        == "modular_linear_system_aggregate"
    )
    assert prompt_catalog["family"]["family_config_defaults"] == {
        "modulus": 7,
        "multiplier_min": 2,
    }


def test_linear_plan_compiles_deterministically_and_executes_across_seeds():
    plan = {
        "schema_version": 5,
        "operator": "in_depth",
        "generator_family": "linear_system_aggregate",
        "family_config": {
            "coefficient_min": -4,
            "coefficient_max": 4,
            "solution_min": -2,
            "solution_max": 2,
        },
        "target_reasoning_move": "identify a row-space invariant",
        "target_concept_group": "algebra",
        "target_concept_type": "algebra.linear_system_sum",
    }
    first = compile_mutation_plan(plan, _parent(), "in_depth")
    second = compile_mutation_plan(plan, _parent(), "in_depth")
    assert first.status is CompilationStatus.COMPILED
    assert first.source_code == second.source_code
    assert first.source_hash == second.source_hash
    assert first.source_hash is not None
    # The compiler normalizes the plan's partial config against registry
    # defaults, so every knob the family owns is present and recorded.
    assert first.family_config == {
        **plan["family_config"],
        "aggregate_multiplier_min": 2,
        "aggregate_multiplier_max": 4,
        "combination_weight_min": 1,
        "combination_weight_max": 3,
    }
    assert first.concept_group == "algebra"
    assert first.concept_type == "algebra.linear_system_sum"
    _assert_mechanical_contract(first.source_code)

    program = first.to_problem_program(parent_id="parent", generation=2)
    assert program.metadata["compiler_source_hash"] == first.source_hash
    assert program.metadata["generator_family"] == "linear_system_aggregate"
    instances = [program.execute(seed) for seed in range(5)]
    assert all(instance is not None for instance in instances)
    assert all(
        lint_problem_instance(instance) == []
        for instance in instances
        if instance is not None
    )
    assert len(
        {instance.problem for instance in instances if instance is not None}
    ) == 5
    assert all(
        instance.answer.lstrip("-").isdigit()
        for instance in instances
        if instance is not None
    )


def test_linear_validator_proves_partial_identifiability():
    valid = {
        "rows": [[1, 0, 0], [0, 1, 1]],
        "rhs": [2, 5],
        "target": [1, 1, 1],
        "witnesses": [[2, 1, 4], [2, 2, 3]],
        "answer": 7,
    }
    result = validate_linear_system_aggregate_instance(valid)
    assert result.valid, result.reasons
    assert result.facts["rank"] == 2
    assert result.facts["nullity"] == 1
    assert result.facts["answer"] == 7

    not_identified = {
        **valid,
        "target": [0, 0, 1],
    }
    rejected = validate_linear_system_aggregate_instance(not_identified)
    assert not rejected.valid
    assert any(
        "rowspace" in reason or "nullspace" in reason
        for reason in rejected.reasons
    )


def test_modular_plan_compiles_with_prime_config_and_brute_force_oracle():
    plan = {
        "schema_version": 5,
        "operator": "in_breadth",
        "generator_family": "modular_linear_system_aggregate",
        "family_config": {"modulus": 11},
        "target_concept_group": "number_theory",
        "target_concept_type": "number_theory.modular_linear_system_sum",
    }
    result = compile_mutation_plan(plan, _parent(), "in_breadth")
    assert result.status is CompilationStatus.COMPILED
    assert result.family_config == {"modulus": 11, "multiplier_min": 2}
    assert "pow(data[\"multiplier\"], -1, modulus)" in result.source_code
    assert "for x in range(modulus)" in result.source_code
    _assert_mechanical_contract(result.source_code)

    program = result.to_problem_program()
    instances = [program.execute(seed) for seed in range(5)]
    assert all(instance is not None for instance in instances)
    assert len(
        {instance.problem for instance in instances if instance is not None}
    ) == 5


def test_modular_validator_checks_gcd_inverse_and_small_brute_force():
    valid = {
        "modulus": 7,
        "multiplier": 3,
        "rows": [[1, 2, 3], [2, 1, 0]],
        "rhs": [3, 4],
        "answer": 0,
    }
    result = validate_modular_linear_system_aggregate_instance(valid)
    assert result.valid, result.reasons
    assert result.facts["modular_inverse"] == 5
    assert result.facts["answer"] == 0
    assert result.facts["solution_count"] >= 2
    assert result.facts["brute_answers"] == (0,)

    noninvertible = {
        "modulus": 6,
        "multiplier": 2,
        "rows": [[1, 0, 1], [1, 2, 1]],
        "rhs": [0, 0],
    }
    rejected = validate_modular_linear_system_aggregate_instance(noninvertible)
    assert not rejected.valid
    assert any("invertible" in reason for reason in rejected.reasons)


def test_config_changes_are_visible_in_hash_and_invalid_config_is_rejected():
    default = compile_mutation_spec(
        MutationSpec("linear_system_aggregate", "in_depth")
    )
    changed = compile_mutation_spec(
        MutationSpec(
            "linear_system_aggregate",
            "in_depth",
            {"coefficient_min": -5, "coefficient_max": 5},
        )
    )
    assert default.compiled and changed.compiled
    assert default.source_hash != changed.source_hash

    bad_modulus = compile_mutation_spec(
        MutationSpec(
            "modular_linear_system_aggregate",
            "in_breadth",
            {"modulus": 12},
        )
    )
    assert bad_modulus.status is CompilationStatus.INVALID_SPEC
    assert any("prime" in reason for reason in bad_modulus.reasons)


def test_labels_operators_and_parent_contract_are_registry_owned():
    spoofed = compile_mutation_spec(
        MutationSpec(
            "modular_linear_system_aggregate",
            "in_breadth",
            target_concept_group="algebra",
        )
    )
    assert spoofed.status is CompilationStatus.INVALID_SPEC
    assert any("registry-derived" in reason for reason in spoofed.reasons)

    wrong_operator = compile_mutation_spec(
        MutationSpec("linear_system_aggregate", "in_breadth")
    )
    assert wrong_operator.status is CompilationStatus.INVALID_SPEC

    same_group_breadth = compile_mutation_spec(
        MutationSpec("modular_linear_system_aggregate", "in_breadth"),
        parent=_parent(
            "number_theory",
            "number_theory.modular_linear_system_sum",
        ),
    )
    assert same_group_breadth.status is CompilationStatus.INVALID_SPEC
    assert any("must change" in reason for reason in same_group_breadth.reasons)


def test_legacy_and_unknown_families_return_clear_unsupported_results():
    legacy = compile_mutation_plan(
        {"schema_version": 4, "operator": "in_depth"},
        _parent(),
        "in_depth",
    )
    assert legacy.status is CompilationStatus.UNSUPPORTED
    assert legacy.source_code is None
    assert any("quarantine" in reason for reason in legacy.reasons)

    free_form = compile_mutation_spec(
        MutationSpec("free_form", "in_depth")
    )
    assert free_form.status is CompilationStatus.UNSUPPORTED
    assert any("quarantined free-form" in reason for reason in free_form.reasons)
