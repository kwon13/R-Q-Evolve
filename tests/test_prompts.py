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
        "CHILD FAMILY": "How many subsets of {item_count} objects have even size?",
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


def test_stage_two_self_declares_exactly_one_domain_without_a_target():
    task = build_generator_task(_program(), _plan())
    system, user = (message["content"] for message in task.messages)
    normalized_system = " ".join(system.split())

    assert "exactly one top-level literal assignment" in system
    assert 'DOMAIN = "<value>"' in system
    assert "not a requested destination" in normalized_system
    assert "Do not declare PROBLEM_TYPE" in system
    assert "runtime derives the problem type deterministically" in system
    for domain in DOMAINS:
        assert _contains_token(system, domain), domain

    # The options are symmetric: neither the parent coordinate nor a desired
    # child coordinate is substituted into the stage-2 user turn.
    assert 'DOMAIN = "algebra"' not in user
    assert "target_cell" not in _text(task)


def test_parent_coordinate_declarations_are_removed_from_stage_two():
    user = build_generator_task(
        _program(legacy_type_declaration=True), _plan()
    ).messages[-1]["content"]
    assert "def generate(seed):" in user
    assert 'DOMAIN = "algebra"' not in user
    assert 'PROBLEM_TYPE = "function"' not in user


def test_stage_one_sees_the_problem_but_not_source_code():
    task = build_family_task(_program())
    assert "What is 3 plus 0?" in task.messages[-1]["content"]
    assert "def generate" not in task.messages[-1]["content"]


def test_stage_two_receives_the_fixed_child_family_verbatim():
    assert _plan()["CHILD FAMILY"] in build_generator_task(
        _program(), _plan()
    ).messages[-1]["content"]


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


def test_finiteness_is_parsed_but_optional():
    without = parse_family_plan(
        "STRUCTURAL MUTATION: change the object\n"
        "CHILD FAMILY: How many subsets of {n} objects have even size?"
    )
    assert without is not None and "WHY FINITE" not in without
    with_finite = parse_family_plan(
        "STRUCTURAL MUTATION: change the object\n"
        "CHILD FAMILY: How many subsets of {n} objects have even size?\n"
        "WHY FINITE: the power set is finite"
    )
    assert with_finite["WHY FINITE"] == "the power set is finite"
    assert "WHY FINITE" not in with_finite["CHILD FAMILY"]


def test_stage_two_states_generator_and_live_verifier_contracts():
    system = build_generator_task(_program(), _plan()).messages[0]["content"]
    assert "define generate(seed)" in system
    assert "assert answer == check" in system
    assert "(problem, str(answer), verifier)" in system
    for mode in ("expression", "boolean", "set"):
        assert f'"mode": "{mode}"' in system
    assert '"mode": "one_of"' not in system
    assert "predicate" in system and "callable" in system


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
