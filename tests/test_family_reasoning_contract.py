"""Regression tests for the registered-family reasoning contract.

Background: a run of the mutation comparison produced ``evaluator_rejected`` on
both ``in_depth`` conditions with the reason "target reasoning move is not
necessary" even though every generated answer was mathematically correct. Two
independent defects caused it, and both are covered here:

1. The ``linear_system_aggregate`` compiler always built rows whose plain sum was
   exactly the target functional ``(1, 1, 1)``, so ``x + y + z`` was readable as
   ``rhs[0] + rhs[1]``. The declared row-space move was genuinely bypassable.
2. The evaluator judged the *planner's* free-form ``target_reasoning_move``
   against a problem a fixed compiler built, so its necessity question was about
   a construction nobody had produced.

The fix keeps the evaluator's ``target_move_required`` gate exactly as strict and
instead makes the family non-degenerate and the judged claim compiler-owned.
"""

import hashlib
import json
import re

import pytest
import sympy

from rq_evolve.mutation_compiler import (
    DEFAULT_MUTATION_FAMILY_REGISTRY,
    CompilationStatus,
    MutationSpec,
    check_linear_system_aggregate_necessity,
    check_modular_linear_system_aggregate_necessity,
    compile_mutation_spec,
    family_contract_payload,
    registered_family_catalog,
    validate_compiled_family_semantics,
)
from rq_evolve.prompts import build_evaluator_messages, parse_evaluator_verdict

_FAMILIES = (
    ("linear_system_aggregate", "in_depth"),
    ("modular_linear_system_aggregate", "in_breadth"),
)


def _compiled(family: str, operator: str):
    result = compile_mutation_spec(
        MutationSpec(generator_family=family, operator=operator)
    )
    assert result.status is CompilationStatus.COMPILED, result.reasons
    return result


def _namespace(result):
    namespace: dict = {}
    exec(compile(result.source_code, "<test>", "exec"), namespace)
    return namespace


def _parse_linear_problem(problem: str) -> tuple[list[list[int]], list[int]]:
    rows: list[list[int]] = []
    rhs: list[int] = []
    pattern = re.compile(
        r"\((-?\d+)\)x \+ \((-?\d+)\)y \+ \((-?\d+)\)z = (-?\d+)"
    )
    for line in problem.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            first, second, third, value = (int(g) for g in match.groups())
            rows.append([first, second, third])
            rhs.append(value)
    assert len(rows) == 2, problem
    return rows, rhs


def _parse_modular_problem(problem: str):
    modulus = int(re.search(r"modulo (\d+)", problem).group(1))
    rows: list[list[int]] = []
    rhs: list[int] = []
    pattern = re.compile(
        r"(-?\d+)x \+ (-?\d+)y \+ (-?\d+)z is congruent to (-?\d+) modulo (\d+)"
    )
    for line in problem.splitlines():
        match = pattern.fullmatch(line.strip())
        if match:
            values = [int(g) for g in match.groups()]
            rows.append(values[:3])
            rhs.append(values[3])
    assert len(rows) == 2, problem
    return modulus, rows, rhs


# --------------------------------------------------------------------------
# 1. The degeneracy that made the evaluator's "NO" correct is gone.
# --------------------------------------------------------------------------


def test_linear_family_never_lets_the_plain_row_sum_answer_the_problem():
    """The exact defect behind the run: row1 + row2 was always (1, 1, 1)."""
    result = _compiled("linear_system_aggregate", "in_depth")
    namespace = _namespace(result)
    for seed in range(60):
        rows, rhs = _parse_linear_problem(namespace["generate"](seed)[0])
        column_sums = [rows[0][index] + rows[1][index] for index in range(3)]
        assert column_sums != [1, 1, 1], (
            f"seed={seed} reintroduces the answer-by-inspection bypass: "
            f"x + y + z would be rhs[0] + rhs[1] = {sum(rhs)}"
        )


def test_linear_family_hides_nontrivial_rowspace_weights():
    """The move must require finding weights, not just adding the equations."""
    result = _compiled("linear_system_aggregate", "in_depth")
    namespace = _namespace(result)
    observed: set[tuple[str, str]] = set()
    for seed in range(40):
        data = namespace["build_instance_data"](seed)
        weights = namespace["_rowspace_weights"](data)
        assert not weights.free_symbols
        assert [str(value) for value in weights] != ["1", "1"]
        assert all(sympy.simplify(value) != 0 for value in weights)
        assert data["aggregate_multiplier"] >= 2
        observed.add(tuple(str(value) for value in weights))
    # Several distinct weight pairs, so the solver cannot memorise one recipe.
    assert len(observed) >= 5


def test_no_single_printed_equation_leaks_the_aggregate():
    result = _compiled("linear_system_aggregate", "in_depth")
    namespace = _namespace(result)
    target = sympy.Matrix([[1, 1, 1]])
    for seed in range(40):
        rows, _ = _parse_linear_problem(namespace["generate"](seed)[0])
        for row in rows:
            assert sympy.Matrix([row]).col_join(target).rank() == 2


def test_modular_family_always_requires_a_real_modular_inversion():
    """multiplier == 1 previously made the summed congruence the answer."""
    result = _compiled("modular_linear_system_aggregate", "in_breadth")
    namespace = _namespace(result)
    for seed in range(60):
        data = namespace["build_instance_data"](seed)
        assert data["multiplier"] % data["modulus"] >= 2
        modulus, rows, _ = _parse_modular_problem(namespace["generate"](seed)[0])
        column_sums = [
            (rows[0][index] + rows[1][index]) % modulus for index in range(3)
        ]
        assert column_sums != [1, 1, 1]


# --------------------------------------------------------------------------
# 2. Independent oracles: correctness and necessity judged separately.
# --------------------------------------------------------------------------


def test_linear_answers_survive_an_independent_symbolic_oracle():
    """Re-solve from the printed text only; ignore the generator's own oracle."""
    result = _compiled("linear_system_aggregate", "in_depth")
    namespace = _namespace(result)
    x, y, z = sympy.symbols("x y z")
    for seed in range(40):
        problem, answer = namespace["generate"](seed)
        rows, rhs = _parse_linear_problem(problem)
        equations = [
            rows[index][0] * x + rows[index][1] * y + rows[index][2] * z
            - rhs[index]
            for index in range(2)
        ]
        solutions = sympy.linsolve(equations, [x, y, z])
        assert solutions, f"seed={seed} produced an inconsistent system"
        point = next(iter(solutions))
        aggregate = sympy.simplify(point[0] + point[1] + point[2])
        assert not aggregate.free_symbols, "aggregate must be uniquely fixed"
        assert int(aggregate) == int(answer)
        # Partial identifiability is what the problem text claims, so it must
        # actually hold: at least one variable stays free.
        assert any(component.free_symbols for component in point)


def test_modular_answers_survive_a_bounded_exhaustive_oracle():
    result = _compiled("modular_linear_system_aggregate", "in_breadth")
    namespace = _namespace(result)
    for seed in range(30):
        problem, answer = namespace["generate"](seed)
        modulus, rows, rhs = _parse_modular_problem(problem)
        solutions = [
            (a, b, c)
            for a in range(modulus)
            for b in range(modulus)
            for c in range(modulus)
            if all(
                sum(rows[index][column] * (a, b, c)[column] for column in range(3))
                % modulus
                == rhs[index]
                for index in range(2)
            )
        ]
        aggregates = {sum(point) % modulus for point in solutions}
        assert aggregates == {int(answer) % modulus}
        assert len(solutions) > 1, "system must stay underdetermined"
        # No seed may pin every variable, else solving directly bypasses the move.
        pinned = sum(
            1
            for column in range(3)
            if len({point[column] for point in solutions}) == 1
        )
        assert pinned < 3


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_deterministic_semantic_gate_passes_and_separates_the_two_judgements(
    family: str,
    operator: str,
):
    result = _compiled(family, operator)
    semantics = validate_compiled_family_semantics(result, range(8))
    assert semantics.valid, semantics.reasons
    assert semantics.reasons == ()
    assert len(semantics.per_seed) == 8
    for entry in semantics.per_seed:
        # "the answer is right" and "the move is required" stay distinct fields.
        assert entry["answer_correct"] is True
        assert entry["necessity_holds"] is True


def test_semantic_gate_rejects_the_old_degenerate_construction():
    """The pre-fix instance shape must now fail the necessity check."""
    degenerate = {
        # (2, 2, -4) + (-1, -1, 5) == (1, 1, 1): the old compiler's invariant.
        "rows": [[2, 2, -4], [-1, -1, 5]],
        "rhs": [-4, 5],
        "target": [1, 1, 1],
        "witnesses": [[0, 1, 0], [3, -2, 0]],
    }
    necessity = check_linear_system_aggregate_necessity(degenerate)
    assert necessity.valid is False
    assert any("inspection" in reason for reason in necessity.reasons)
    assert necessity.facts["plain_row_sum"] == [1, 1, 1]


def test_semantic_gate_rejects_a_single_equation_leak():
    leaking = {
        "rows": [[1, 1, 1], [1, 3, 0]],
        "rhs": [4, 6],
        "target": [1, 1, 1],
        "witnesses": [[0, 2, 2], [3, 1, 0]],
    }
    necessity = check_linear_system_aggregate_necessity(leaking)
    assert necessity.valid is False
    assert any("proportional" in reason for reason in necessity.reasons)


def test_semantic_gate_rejects_a_unit_modular_multiplier():
    necessity = check_modular_linear_system_aggregate_necessity(
        {
            "modulus": 7,
            "multiplier": 1,
            "rows": [[3, 5, 2], [5, 3, 6]],
            "rhs": [1, 2],
        }
    )
    assert necessity.valid is False
    assert any("plain sum" in reason for reason in necessity.reasons)


def test_semantic_gate_requires_a_compiled_source():
    unsupported = compile_mutation_spec(
        MutationSpec(generator_family="free_form.anything", operator="in_depth")
    )
    semantics = validate_compiled_family_semantics(unsupported, range(3))
    assert semantics.valid is False
    assert any("compiled source" in reason for reason in semantics.reasons)


# --------------------------------------------------------------------------
# 3. The registry owns the move; the evaluator judges the compiler's claim.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_registry_owns_the_target_reasoning_move(family: str, operator: str):
    definition = DEFAULT_MUTATION_FAMILY_REGISTRY.get(family)
    contract = definition.reasoning_contract
    assert contract.target_reasoning_move.strip()
    assert contract.necessity_witness.strip()
    assert contract.excluded_bypasses
    result = _compiled(family, operator)
    assert result.reasoning_contract is contract
    program = result.to_problem_program()
    recorded = program.metadata["family_reasoning_contract"]
    assert recorded["target_reasoning_move"] == contract.target_reasoning_move


def test_planner_catalog_publishes_the_move_the_family_guarantees():
    """The planner must be able to see what the compiler will actually build."""
    from rq_evolve.program import ProblemProgram

    parent = ProblemProgram(
        source_code=(
            'def generate(seed):\n    return "Find an integer.", "1"\n'
            'CONCEPT_REASON = "t"\n'
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )
    catalog = registered_family_catalog(parent, "in_depth")
    assert "reasoning_contract" in catalog
    assert "target_reasoning_move" in catalog
    assert "excluded_bypasses" in catalog


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_evaluator_prompt_judges_the_contract_not_the_plan_prose(
    family: str,
    operator: str,
):
    result = _compiled(family, operator)
    semantics = validate_compiled_family_semantics(result, range(3))
    contract = family_contract_payload(result, semantics)
    assert contract is not None
    # A plan whose prose describes the *parent* (three equations) -- exactly the
    # mismatch that produced the false "not necessary" verdicts.
    plan = {
        "schema_version": 5,
        "operator": operator,
        "generator_family": family,
        "target_reasoning_move": (
            "recognize that adding all three equations gives a multiple of "
            "x + y + z"
        ),
    }
    messages = build_evaluator_messages(
        "problem body",
        plan,
        answer_text="7",
        program_source=result.source_code,
        family_contract=contract,
    )
    user = messages[1]["content"]
    assert "Verified family contract" in user
    assert contract["target_reasoning_move"] in user
    assert "deterministic_verification" in user
    # The plan is dropped entirely, not demoted: labelling it "ignore this" did
    # not stop it from driving verdicts on byte-identical problems.
    assert "Mutation plan" not in user
    assert "Background only" not in user
    assert plan["target_reasoning_move"] not in user
    system = messages[0]["content"]
    # The gate itself is unchanged: NO still means INVALID.
    assert "target_move_required: YES or NO" in system


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_contract_block_is_a_parsable_json_object(family: str, operator: str):
    """It was serialized as a Python repr inside a JSON string, unreadable."""
    result = _compiled(family, operator)
    contract = family_contract_payload(
        result,
        validate_compiled_family_semantics(result, range(3)),
    )
    assert isinstance(contract, dict)
    messages = build_evaluator_messages(
        "problem body",
        None,
        program_source=result.source_code,
        family_contract=contract,
    )
    user = messages[1]["content"]
    marker = "Verified family contract:"
    block = user[user.index(marker) + len(marker):].lstrip()
    parsed, _ = json.JSONDecoder().raw_decode(block)
    assert isinstance(parsed, dict), "contract must be a JSON object"
    assert parsed["target_reasoning_move"] == contract["target_reasoning_move"]
    assert isinstance(parsed["excluded_bypasses"], list)
    assert isinstance(parsed["family_config"], dict)
    assert parsed["deterministic_verification"]["necessity_holds"] is True
    # No Python repr leaked through a str() fallback.
    assert "'" not in block.split('"target_reasoning_move"')[0]


def test_contract_payload_reports_the_two_verification_axes_separately():
    result = _compiled("linear_system_aggregate", "in_depth")
    semantics = validate_compiled_family_semantics(result, range(5))
    contract = family_contract_payload(result, semantics)
    verification = contract["deterministic_verification"]
    assert verification["valid"] is True
    assert verification["answer_oracle_agrees"] is True
    assert verification["necessity_holds"] is True
    assert verification["seeds"] == [0, 1, 2, 3, 4]
    assert contract["compiler_source_hash"] == result.source_hash


def test_free_form_family_gets_no_contract_and_keeps_plan_only_prompt():
    unsupported = compile_mutation_spec(
        MutationSpec(generator_family="free_form.whatever", operator="in_depth")
    )
    assert family_contract_payload(unsupported) is None
    messages = build_evaluator_messages(
        "problem body",
        {"schema_version": 5, "target_reasoning_move": "do the thing"},
        family_contract=None,
    )
    user = messages[1]["content"]
    assert "Verified family contract" not in user
    assert "Mutation plan:" in user


# --------------------------------------------------------------------------
# 4. The evaluator gate was not weakened.
# --------------------------------------------------------------------------


def test_target_move_required_no_is_still_a_rejection():
    """Reproduces the recorded evaluator output from the failing run verbatim."""
    recorded = (
        "reason: The problem is internally coherent and the supplied answer is "
        "correct. The system of equations does not determine x, y, and z "
        "individually, but the sum of the coefficients in each equation must "
        "equal the same multiple of x + y + z. The answer is 0, which is the "
        "correct value of x + y + z.\n\n"
        "target_move_required: NO\n\n"
        "verdict: VALID"
    )
    is_valid, reason = parse_evaluator_verdict(recorded, require_target_move=True)
    assert is_valid is False
    assert "target reasoning move is not necessary" in reason


def test_missing_target_move_line_is_still_a_rejection():
    is_valid, reason = parse_evaluator_verdict(
        "reason: looks fine\n\nverdict: VALID",
        require_target_move=True,
    )
    assert is_valid is False
    assert "not explicitly judged necessary" in reason


def test_explicit_yes_with_valid_verdict_still_passes():
    is_valid, _ = parse_evaluator_verdict(
        "reason: the row-space combination is required\n\n"
        "target_move_required: YES\n\n"
        "verdict: VALID",
        require_target_move=True,
    )
    assert is_valid is True


def test_invalid_verdict_wins_even_with_yes():
    is_valid, _ = parse_evaluator_verdict(
        "reason: answer is wrong\n\ntarget_move_required: YES\n\n"
        "verdict: INVALID",
        require_target_move=True,
    )
    assert is_valid is False


# --------------------------------------------------------------------------
# 5. plain vs. reasoning stays a fair comparison.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_both_conditions_receive_an_identical_contract_and_compiler_action(
    family: str,
    operator: str,
):
    """Neither condition gets a different family, config space, or contract."""
    plain_plan = {
        "schema_version": 5,
        "operator": operator,
        "generator_family": family,
        "family_config": {},
        "target_reasoning_move": "plain-condition narration",
    }
    reasoning_plan = {
        **plain_plan,
        "target_reasoning_move": "reasoning-condition narration",
    }
    plain = compile_mutation_spec(plain_plan)
    reasoning = compile_mutation_spec(reasoning_plan)
    assert plain.status is CompilationStatus.COMPILED
    assert reasoning.status is CompilationStatus.COMPILED
    # Same source, same hash, same config space, same contract: the only thing
    # that differs between conditions is which family/config the planner picks.
    assert plain.source_code == reasoning.source_code
    assert plain.source_hash == reasoning.source_hash
    assert dict(plain.family_config) == dict(reasoning.family_config)
    assert plain.reasoning_contract == reasoning.reasoning_contract
    plain_contract = family_contract_payload(
        plain,
        validate_compiled_family_semantics(plain, range(4)),
    )
    reasoning_contract = family_contract_payload(
        reasoning,
        validate_compiled_family_semantics(reasoning, range(4)),
    )
    assert dict(plain_contract) == dict(reasoning_contract)


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_evaluator_input_is_byte_identical_across_conditions(
    family: str,
    operator: str,
):
    """The measured quantity must not depend on which condition wrote the plan.

    With the plan included as "background", plain narration passed 4/5 seeds and
    reasoning narration 0/5 on byte-identical problems and source. Same family
    and variant must now mean the same evaluator input.
    """
    result = _compiled(family, operator)
    contract = family_contract_payload(
        result,
        validate_compiled_family_semantics(result, range(4)),
    )
    namespace = _namespace(result)
    problem, answer = namespace["generate"](0)
    digests = set()
    for plan in (
        {
            "schema_version": 5,
            "operator": operator,
            "failure_summary": None,
            "target_reasoning_move": "plain-condition narration",
            "predicted_pre_behavior": None,
        },
        {
            "schema_version": 5,
            "operator": operator,
            "failure_summary": "the solver dropped the division step",
            "target_reasoning_move": "reasoning-condition narration",
            "predicted_pre_behavior": "solver will divide incorrectly",
        },
        None,
    ):
        messages = build_evaluator_messages(
            problem,
            plan,
            answer_text=answer,
            program_source=result.source_code,
            family_contract=contract,
        )
        payload = messages[0]["content"] + messages[1]["content"]
        digests.add(hashlib.sha256(payload.encode()).hexdigest())
    assert len(digests) == 1, "plan narration must not reach the evaluator"


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_every_registered_variant_compiles_to_a_distinct_verified_family(
    family: str,
    operator: str,
):
    """A variant choice must actually change the generated problems.

    Both conditions previously compiled to one source hash, so no evaluator fix
    could make "reasoning beats plain" testable.
    """
    contract = DEFAULT_MUTATION_FAMILY_REGISTRY.get(family).reasoning_contract
    assert len(contract.variants) >= 2
    hashes: dict[str, str] = {}
    problem_sets: dict[str, set[str]] = {}
    for variant in contract.variants:
        result = compile_mutation_spec(
            MutationSpec(
                generator_family=family,
                operator=operator,
                family_variant=variant.variant_id,
            )
        )
        assert result.status is CompilationStatus.COMPILED, result.reasons
        assert result.family_variant == variant.variant_id
        assert result.variant_selected_by_plan is True
        assert result.variant_targets_failure_mode == variant.targets_failure_mode
        # Registry-owned overrides still pass the family's own oracles.
        semantics = validate_compiled_family_semantics(result, range(8))
        assert semantics.valid, (variant.variant_id, semantics.reasons)
        hashes[variant.variant_id] = result.source_hash
        namespace = _namespace(result)
        problem_sets[variant.variant_id] = {
            namespace["generate"](seed)[0] for seed in range(20)
        }
    assert len(set(hashes.values())) == len(hashes), hashes
    # Distinct variants must not merely relabel the same problems.
    ids = list(problem_sets)
    for index, first in enumerate(ids):
        for second in ids[index + 1:]:
            assert not (problem_sets[first] & problem_sets[second]), (
                first,
                second,
            )


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_absent_variant_defaults_and_is_recorded_as_not_plan_selected(
    family: str,
    operator: str,
):
    """A planner that ignores the choice must be visible, not silently equal."""
    result = compile_mutation_spec(
        MutationSpec(generator_family=family, operator=operator)
    )
    assert result.status is CompilationStatus.COMPILED
    contract = DEFAULT_MUTATION_FAMILY_REGISTRY.get(family).reasoning_contract
    assert result.family_variant == contract.default_variant_id
    assert result.variant_selected_by_plan is False
    metadata = result.to_problem_program().metadata
    assert metadata["family_variant"] == contract.default_variant_id
    assert metadata["variant_selected_by_plan"] is False


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_unknown_variant_is_a_terminal_spec_error_listing_valid_ids(
    family: str,
    operator: str,
):
    result = compile_mutation_spec(
        MutationSpec(
            generator_family=family,
            operator=operator,
            family_variant="does_not_exist",
        )
    )
    assert result.status is CompilationStatus.INVALID_SPEC
    joined = "; ".join(result.reasons)
    assert "unknown family_variant" in joined
    contract = DEFAULT_MUTATION_FAMILY_REGISTRY.get(family).reasoning_contract
    for variant_id in contract.variant_ids:
        assert variant_id in joined


def test_registry_variant_wins_over_an_echoed_plan_config():
    """A copied default must not silently neutralize the variant choice.

    In the observed run both planners echoed the published defaults verbatim, so
    plan-wins layering would have cancelled the only consequential choice.
    """
    result = compile_mutation_spec(
        {
            "schema_version": 5,
            "operator": "in_depth",
            "generator_family": "linear_system_aggregate",
            "family_variant": "heavy_division",
            # Exactly the echo the planner produced last run.
            "family_config": {
                "aggregate_multiplier_max": 4,
                "coefficient_max": 3,
                "coefficient_min": -3,
                "combination_weight_max": 3,
                "solution_max": 3,
                "solution_min": -3,
            },
        }
    )
    assert result.status is CompilationStatus.COMPILED
    # heavy_division owns the multiplier bounds, so they survive the echo.
    assert result.family_config["aggregate_multiplier_min"] == 4
    assert result.family_config["aggregate_multiplier_max"] == 7
    assert result.variant_overridden_plan_keys == ("aggregate_multiplier_max",)
    assert validate_compiled_family_semantics(result, range(6)).valid


def test_plan_may_still_set_knobs_the_variant_does_not_own():
    result = compile_mutation_spec(
        {
            "schema_version": 5,
            "operator": "in_depth",
            "generator_family": "linear_system_aggregate",
            "family_variant": "heavy_division",
            "family_config": {"solution_min": -5, "solution_max": 5},
        }
    )
    assert result.status is CompilationStatus.COMPILED
    assert result.family_config["solution_min"] == -5
    assert result.family_config["solution_max"] == 5
    assert result.family_config["aggregate_multiplier_max"] == 7
    assert result.variant_overridden_plan_keys == ()
    assert validate_compiled_family_semantics(result, range(6)).valid


def test_variant_catalog_is_published_to_the_planner():
    from rq_evolve.program import ProblemProgram

    parent = ProblemProgram(
        source_code=(
            'def generate(seed):\n    return "Find an integer.", "1"\n'
            'CONCEPT_REASON = "t"\n'
            'CONCEPT_GROUP = "algebra"\n'
            'CONCEPT_TYPE = "algebra.linear_system_sum"\n'
        )
    )
    catalog = json.loads(registered_family_catalog(parent, "in_depth"))
    variants = catalog["family"]["family_variants"]
    assert len(variants) >= 2
    assert catalog["family"]["default_family_variant"] == "balanced"
    for entry in variants:
        assert entry["variant_id"]
        # Each variant must say which failure it targets, so an evidence-holding
        # planner has a documented basis for choosing it.
        assert entry["targets_failure_mode"]
        assert isinstance(entry["family_config"], dict)


def test_linear_necessity_still_holds_for_every_variant_at_scale():
    contract = DEFAULT_MUTATION_FAMILY_REGISTRY.get(
        "linear_system_aggregate"
    ).reasoning_contract
    target = sympy.Matrix([[1, 1, 1]])
    for variant in contract.variants:
        result = compile_mutation_spec(
            MutationSpec(
                generator_family="linear_system_aggregate",
                operator="in_depth",
                family_variant=variant.variant_id,
            )
        )
        namespace = _namespace(result)
        for seed in range(30):
            data = namespace["build_instance_data"](seed)
            rows, _ = _parse_linear_problem(namespace["generate"](seed)[0])
            assert [rows[0][i] + rows[1][i] for i in range(3)] != [1, 1, 1]
            for row in rows:
                assert sympy.Matrix([row]).col_join(target).rank() == 2
            weights = namespace["_rowspace_weights"](data)
            assert [str(value) for value in weights] != ["1", "1"]
            assert data["aggregate_multiplier"] >= 2


@pytest.mark.parametrize("family,operator", _FAMILIES)
def test_build_instance_data_matches_generate_on_every_seed(
    family: str,
    operator: str,
):
    """The structured gate must inspect exactly what the solver will see."""
    result = _compiled(family, operator)
    namespace = _namespace(result)
    renderer = (
        namespace["_render_linear_problem"]
        if family == "linear_system_aggregate"
        else namespace["_render_modular_problem"]
    )
    for seed in range(30):
        problem, _ = namespace["generate"](seed)
        assert renderer(namespace["build_instance_data"](seed)) == problem
