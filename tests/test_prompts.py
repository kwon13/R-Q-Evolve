"""Untargeted two-stage mutation and local descriptor prompt contracts."""

import re

import pytest

from rq_evolve import prompts
from rq_evolve.concepts import DOMAINS, PROBLEM_TYPES
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    MUTATION_OP,
    _render_template,
    _template_context,
    build_family_task,
    build_generator_task,
    parse_family_plan,
)


def _program(*, legacy_type_declaration: bool = False) -> ProblemProgram:
    legacy = 'PROBLEM_TYPE = "function"\n' if legacy_type_declaration else ""
    return ProblemProgram(
        source_code=(
            'DOMAIN = "algebra"\n'
            + legacy
            + "\n"
            + "def generate(seed):\n"
            + '    return f"What is 3 plus {seed}?", str(3 + seed)\n'
        ),
        metadata={"problem_type": "function"},
    )


def _plan() -> dict[str, str]:
    return {
        "STRUCTURAL MUTATION": "Replace evaluation by a finite configuration.",
        "CHILD FAMILY": "How many subsets of [[item_count]] objects have even size?",
        "WHY FINITE": "There are finitely many subsets of the stated set.",
    }


def _text(task) -> str:
    return "\n".join(message["content"] for message in task.messages)


def _contains_token(text: str, token: str) -> bool:
    return bool(
        re.search(rf"(?<![a-z_]){re.escape(token)}(?![a-z_])", text, re.I)
    )


def test_stage_one_has_no_closed_descriptor_vocabulary_or_target():
    text = _text(build_family_task(_program()))
    for label in (*DOMAINS, *PROBLEM_TYPES):
        assert not _contains_token(text, label), label
    assert "target" not in text.lower()
    assert "target_cell" not in text
    assert "no destination" in text.lower()


def test_stage_one_requires_collision_free_child_placeholders_in_all_examples():
    system = build_family_task(_program()).messages[0]["content"]
    assert "[[descriptive_name]]" in system
    assert "Do not use Python-format {name}" in system
    for name in (
        "modulus",
        "residue",
        "board_size",
        "vertex_count",
        "neighbor_count",
    ):
        assert f"[[{name}]]" in system
    for old in ("{modulus}", "{residue}", "{board_size}", "{vertex_count}"):
        assert old not in system


def test_stage_two_uses_header_contract_without_a_target():
    task = build_generator_task(_program(), _plan())
    system, user = (message["content"] for message in task.messages)
    normalized_system = " ".join(system.split())

    assert "DOMAIN: token" in system
    assert "MODE: expression|boolean|set" in system
    assert "CORE:" in system
    assert "INVALID: <specific reason>" in system
    assert "never a requested destination" in normalized_system
    assert "Do not output PROBLEM_TYPE" in system
    assert "runtime derives PROBLEM_TYPE deterministically" in system
    for domain in DOMAINS:
        assert _contains_token(system, domain), domain

    # Neither the parent coordinate nor a desired child coordinate is
    # substituted into the stage-2 user turn.
    assert 'DOMAIN = "algebra"' not in user
    assert "target_cell" not in _text(task)


def test_stage_two_receives_no_parent_statement_source_or_example():
    user = build_generator_task(
        _program(legacy_type_declaration=True), _plan()
    ).messages[-1]["content"]
    assert "What is 3 plus" not in user
    assert "def generate(seed):" not in user
    assert 'DOMAIN = "algebra"' not in user
    assert 'PROBLEM_TYPE = "function"' not in user
    assert "REFERENCE" not in user
    assert "PARENT" not in user


def test_stage_one_sees_the_problem_but_not_source_code():
    task = build_family_task(_program())
    assert "What is 3 plus 0?" in task.messages[-1]["content"]
    assert "def generate" not in task.messages[-1]["content"]


def test_stage_two_receives_only_fixed_family_finiteness_and_placeholders():
    user = build_generator_task(_program(), _plan()).messages[-1]["content"]
    assert _plan()["CHILD FAMILY"] in user
    assert _plan()["WHY FINITE"] in user
    assert "- item_count" in user
    assert "FIXED CHILD FAMILY:" in user
    assert "WHY FINITE:" in user
    assert "PLACEHOLDER NAMES" in user


def test_stage_two_does_not_execute_or_inspect_the_parent():
    broken = ProblemProgram(
        source_code=(
            'DOMAIN = "geometry"\n'
            "PARENT_SECRET_SENTINEL = 'must not be rendered'\n"
            "raise RuntimeError('must not execute')\n"
        )
    )
    task = build_generator_task(broken, _plan())
    assert "PARENT_SECRET_SENTINEL" not in _text(task)
    assert "must not execute" not in _text(task)


def test_stage_two_lists_unique_placeholders_in_first_seen_order():
    plan = {
        **_plan(),
        "CHILD FAMILY": (
            "Compute [[first_value]] + [[second_value]] + [[first_value]]."
        ),
    }
    user = build_generator_task(_program(), plan).messages[-1]["content"]
    assert user.count("- first_value") == 1
    assert user.count("- second_value") == 1
    assert user.index("- first_value") < user.index("- second_value")


def test_stage_two_rejects_a_family_without_valid_placeholders_before_call():
    plan = {**_plan(), "CHILD FAMILY": "Compute the fixed value 17."}
    with pytest.raises(ValueError, match="valid .* placeholder syntax"):
        build_generator_task(_program(), plan)


def test_a_parent_that_cannot_run_still_produces_a_prompt():
    broken = ProblemProgram(source_code='DOMAIN = "algebra"\n')
    assert "did not run here" in build_family_task(broken).messages[-1]["content"]


def test_target_cell_is_retired_and_fails_loudly():
    with pytest.raises(ValueError, match="target_cell is retired"):
        build_family_task(_program(), target_cell=(0, 0))


def test_both_stages_use_the_one_untargeted_operator():
    family = build_family_task(_program())
    generator = build_generator_task(_program(), _plan())
    assert family.op == generator.op == MUTATION_OP == "mutate"
    assert family.stage == "family" and generator.stage == "generator"
    assert [m["role"] for m in family.messages] == ["system", "user"]
    assert [m["role"] for m in generator.messages] == ["system", "user"]


def test_family_plan_contains_mathematics_only():
    system = build_family_task(_program()).messages[0]["content"]
    for line in ("STRUCTURAL MUTATION:", "CHILD FAMILY:", "WHY FINITE:"):
        assert line in system
    for line in ("DOMAIN:", "PROBLEM_TYPE:", "GROUP:", "SKILL:"):
        assert line not in system


def test_finiteness_is_requested_and_parsed_separately():
    without = parse_family_plan(
        "STRUCTURAL MUTATION: change the object\n"
        "CHILD FAMILY: How many subsets of [[n]] objects have even size?"
    )
    assert without is None
    with_finite = parse_family_plan(
        "STRUCTURAL MUTATION: change the object\n"
        "CHILD FAMILY: How many subsets of [[n]] objects have even size?\n"
        "WHY FINITE: the power set is finite"
    )
    assert with_finite["WHY FINITE"] == "the power set is finite"
    assert "WHY FINITE" not in with_finite["CHILD FAMILY"]


def test_missing_finiteness_is_rejected_before_stage_two_call():
    plan = {key: value for key, value in _plan().items() if key != "WHY FINITE"}
    with pytest.raises(ValueError, match="WHY FINITE"):
        build_generator_task(_program(), plan)


@pytest.mark.parametrize(
    "family",
    (
        "Compute the fixed value 17.",
        "Compute [[bad-name]].",
        "Compute [[missing_end].",
        "Compute {legacy_name}.",
    ),
)
def test_family_plan_rejects_invalid_or_legacy_placeholders(family):
    assert parse_family_plan(
        "STRUCTURAL MUTATION: change the object\n"
        f"CHILD FAMILY: {family}\n"
        "WHY FINITE: the requested computation is explicitly bounded"
    ) is None


def test_family_plan_allows_one_letter_braced_set_notation():
    family = "Find all x in {x} with 0 <= x < [[upper_bound]]."
    plan = parse_family_plan(
        "STRUCTURAL MUTATION: change the requested object\n"
        f"CHILD FAMILY: {family}\n"
        "WHY FINITE: the interval is explicitly bounded"
    )
    assert plan is not None and plan["CHILD FAMILY"] == family


@pytest.mark.parametrize(
    "family",
    (
        "Compute [[n]] and [[bad-name]].",
        "Compute [[n]] using {legacy_name}.",
    ),
)
def test_generator_task_shares_strict_family_placeholder_validation(family):
    with pytest.raises(ValueError, match="valid .* placeholder syntax"):
        build_generator_task(_program(), {**_plan(), "CHILD FAMILY": family})


def test_stage_prompts_include_luna_audit_preflight_checks():
    family_system = build_family_task(_program()).messages[0]["content"]
    generator_system = build_generator_task(_program(), _plan()).messages[0][
        "content"
    ]
    assert "at least one complete" in family_system
    assert "A reply missing any of these three fields is invalid" in family_system
    assert "Before replying, verify all five conditions" in generator_system
    assert "return INVALID rather than guessing" in generator_system
    assert "maximum/minimum value is" in generator_system


def test_stage_two_states_core_and_materialized_verifier_contracts():
    system = build_generator_task(_program(), _plan()).messages[0]["content"]
    normalized = " ".join(system.split())
    assert "exactly one function named `def build_instance(rng):`" in normalized
    assert "Do not define `generate`" in system
    assert "Do not import `random`" in system
    assert "Do not render the natural-language problem" in system
    assert '`return parameters, answer, check`' in system
    assert "materially independent route" in system
    assert "fully materialized, non-callable" in system
    assert "Bound every sample, loop, search, and collection" in system
    assert system.count("```python") == 1
    assert system.count("```") == 2
    for mode in ("expression", "boolean", "set"):
        assert _contains_token(system, mode)
    assert not _contains_token(system, "one_of")
    assert "predicates" in system and "callable" in system


def test_stage_two_explains_deterministic_mode_to_type_mapping():
    system = build_generator_task(_program(), _plan()).messages[0]["content"]
    expected = {
        "boolean": "decision",
        "set": "search",
        "How-many": "counting",
        "maximum/minimum": "optimization",
        "other expression": "function",
    }
    for contract, problem_type in expected.items():
        assert contract in system
        assert _contains_token(system, problem_type)


def test_generator_task_keeps_provenance_audit_only():
    provenance = {"structural_inspiration": {"donor": "secret-donor"}}
    task = build_generator_task(_program(), _plan(), provenance=provenance)
    assert task.provenance == provenance
    assert "secret-donor" not in _text(task)


def test_legacy_template_context_strips_all_coordinate_declarations():
    context = _template_context(_program(legacy_type_declaration=True))
    assert set(context) == {"parent_source", "parent_problem"}
    assert 'DOMAIN = "algebra"' not in context["parent_source"]
    assert 'PROBLEM_TYPE = "function"' not in context["parent_source"]


def test_no_placeholder_survives_rendered_mutation_prompts():
    for task in (
        build_family_task(_program()),
        build_generator_task(_program(), _plan()),
    ):
        for message in task.messages:
            assert "$" not in message["content"]


def test_an_unsupplied_placeholder_is_an_error():
    with pytest.raises(KeyError, match="parent_source"):
        _render_template("use $parent_source", {})


def test_substituted_dollar_sign_is_not_mistaken_for_a_placeholder():
    assert _render_template("$text", {"text": "cost = $5"}) == "cost = $5"


def test_remote_descriptor_task_and_verdict_apis_are_absent():
    retired = (
        "JUDGE_FIELDS",
        "JudgeVerdict",
        "build_judge_messages",
        "build_judge_task",
        "judge_system_prompt",
        "parse_judge_verdict",
        "judge_accepts",
    )
    assert [name for name in retired if hasattr(prompts, name)] == []
