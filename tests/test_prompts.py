import pytest

from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    _render_template,
    _template_context,
    build_fix_task,
    build_mutation_task,
    mutation_system_prompt,
    parse_evaluator_verdict,
)


def _program(group: str = "algebra", skill: str = "transformation") -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return "What is 3 + {{seed}}?", str(3 + seed)


GROUP = "{group}"
SKILL = "{skill}"
'''
    )


def _conversation(task) -> str:
    return "\n".join(message["content"] for message in task.messages)


def _offered(parent, key: str) -> set[str]:
    """The choice list the template substitutes for the moving axis.

    Read from the substitution context, not by parsing the rendered sentence:
    the instruction line also names the parent's own value ("different from
    ...") so any text scan would see it as offered. Reading the context also
    survives rewording, which these prompts are actively going through.
    """
    return {v.strip() for v in _template_context("in_depth", parent)[key].split(",")}


# --- system prompt ---------------------------------------------------------


def test_each_operator_defines_the_axis_it_must_cross_blind():
    """The held axis has the parent as an instance; the moved axis has nothing.

    in_depth keeps GROUP and moves SKILL, so it carries the SKILL definitions;
    in_breadth is the mirror. A label the model can pick but not read a
    definition for is unfalsifiable.
    """
    parent = _program("algebra", "transformation")

    a = build_mutation_task("in_depth", parent).messages[-1]["content"]
    for skill in SKILLS:
        assert f"- {skill}:" in a, skill
    assert "- number_theory:" not in a, "in_depth holds GROUP; it has the parent"

    b = build_mutation_task("in_breadth", parent).messages[-1]["content"]
    for group in GROUPS:
        assert f"- {group}:" in b, group
    assert "- casework:" not in b, "in_breadth holds SKILL; it has the parent"


def test_the_definitions_have_one_source_shared_with_the_evaluator():
    """Two copies drift; the evaluator must judge against the text the
    mutation was written from."""
    from rq_evolve.prompts import skill_definition

    parent = _program("algebra", "transformation")
    a = build_mutation_task("in_depth", parent).messages[-1]["content"]
    for skill in SKILLS:
        assert skill_definition(skill) in a, skill


def test_system_prompt_declares_both_axis_vocabularies():
    system = mutation_system_prompt()
    for group in GROUPS:
        assert group in system, group
    assert "independent" in system


def test_system_prompt_teaches_the_self_check_rather_than_demanding_it():
    """The assert is the only mechanical problem-text/answer agreement check."""
    system = mutation_system_prompt()
    assert "assert answer == check" in system
    assert "recomputed from the words of problem_text" in system
    assert 'GROUP = "<one of the GROUP vocabulary>"' in system
    assert 'SKILL = "<one of the SKILL vocabulary>"' in system


def test_system_prompt_no_longer_teaches_the_retry_loop_strategy():
    """Seeds curate their parameter pools; they never reject-and-resample.

    Demanding MAX_ATTEMPTS while every in-context parent omits it produced 0/8
    compliance -- the parent, not the instruction list, is what an 8B base model
    imitates.
    """
    system = mutation_system_prompt()
    assert "MAX_ATTEMPTS" not in system
    assert "continue" not in system


def test_system_prompt_no_longer_asks_for_the_retired_labels():
    system = mutation_system_prompt()
    assert "CONCEPT_TYPE" not in system
    assert "CONCEPT_REASON" not in system


# --- operator A: same GROUP, different SKILL -------------------------------


def test_in_depth_holds_group_and_offers_every_other_skill():
    parent = _program("algebra", "transformation")
    user = build_mutation_task("in_depth", parent).messages[-1]["content"]

    assert 'GROUP = "algebra"' in user, "the held axis is named explicitly"
    offered = _offered(parent, "allowed_skills")
    assert offered == set(SKILLS) - {"transformation"}, offered


# --- operator B: same SKILL, different GROUP -------------------------------


def test_in_breadth_holds_skill_and_offers_every_other_group():
    parent = _program("algebra", "induction")
    user = build_mutation_task("in_breadth", parent).messages[-1]["content"]

    assert 'SKILL = "induction"' in user, "the held axis is named explicitly"
    offered = _offered(parent, "allowed_groups")
    assert offered == set(GROUPS) - {"algebra"}, offered


# --- shared rendering rules ------------------------------------------------


def test_parent_source_is_inlined_and_shots_are_omitted():
    parent = _program()
    for op in ("in_depth", "in_breadth"):
        user = build_mutation_task(op, parent).messages[-1]["content"]
        assert "def generate(seed)" in user
        assert "$few_shot_examples" not in user
        assert "Few-shot examples:" not in user


def test_no_placeholder_survives_into_a_rendered_prompt():
    parent = _program()
    for op in ("in_depth", "in_breadth"):
        conversation = _conversation(build_mutation_task(op, parent))
        for name in (
            "$parent_group",
            "$parent_skill",
            "$allowed_skills",
            "$allowed_groups",
            "$parent_source",
            "$parent_p_hat",
        ):
            assert name not in conversation, (op, name)


def test_an_unsupplied_placeholder_is_an_error_not_a_silent_passthrough():
    """A literal `$parent_skill` in the prompt reads as a plausible instruction."""
    with pytest.raises(KeyError, match="parent_skill"):
        _render_template("hold $parent_group, move $parent_skill", {"parent_group": "algebra"})


def test_a_dollar_sign_from_substituted_content_is_not_flagged():
    assert _render_template("$parent_source", {"parent_source": "cost = $5"}) == "cost = $5"


def test_fix_task_replays_the_original_conversation():
    parent = _program()
    task = build_mutation_task("in_depth", parent)
    fix_task = build_fix_task(task, "bad code", "missing SKILL")

    assert [m["role"] for m in fix_task.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert fix_task.messages[0]["content"] == task.messages[0]["content"]
    assert fix_task.messages[1]["content"] == task.messages[-1]["content"]
    assert fix_task.messages[2]["content"] == "bad code"
    assert "missing SKILL" in fix_task.messages[3]["content"]
    assert fix_task.op == task.op


def test_evaluator_verdict_passes_only_on_explicit_valid():
    valid, reason = parse_evaluator_verdict("reason: coherent\nverdict: VALID")
    assert valid is True, reason

    for output in (
        "reason: contradictory conditions\nverdict: INVALID",
        "reason: unclear",
        "",
    ):
        valid, _ = parse_evaluator_verdict(output)
        assert valid is False, output
