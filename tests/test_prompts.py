"""The mutation prompt and the judge contract.

The operator pair is gone. Nothing in the mutation prompt names a target GROUP
or SKILL any more, so these tests check the opposite of what they used to: that
the child is handed both full vocabularies and no instruction about where to
land, and that the judge is handed the visible problem and nothing else.
"""

import pytest

from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    MUTATION_OP,
    _render_template,
    _template_context,
    build_fix_task,
    build_judge_messages,
    build_mutation_task,
    judge_accepts,
    judge_system_prompt,
    mutation_system_prompt,
    parse_judge_verdict,
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


def _user(parent) -> str:
    return build_mutation_task(parent).messages[-1]["content"]


# --- the mutation prompt ---------------------------------------------------


def test_the_child_may_read_every_label_it_may_choose():
    """A label the model can pick but not read a definition for is unfalsifiable.

    With no axis held fixed there is nothing to withhold: the retired operators
    each dropped the vocabulary of the axis they pinned, on the reasoning that
    the parent was a worked instance of it.
    """
    user = _user(_program("algebra", "transformation"))
    for group in GROUPS:
        assert f"{group}:" in user, group
    for skill in SKILLS:
        assert f"{skill}:" in user, skill


def test_the_definitions_have_one_source_on_disk():
    parent = _program()
    context = _template_context(parent)
    for group in GROUPS:
        assert f"{group}:" in context["allowed_groups"]
    for skill in SKILLS:
        assert f"{skill}:" in context["allowed_skills"]


def test_the_parent_labels_travel_but_no_target_is_named():
    """The parent is context. Ordering a label change is what caused the drift.

    Demanding "now produce SKILL=invariant" made the label a target the child
    was written to satisfy, so the archived coordinates stopped describing the
    problem. The prompt states where the parent sits and asks for mathematical
    distinctness instead.
    """
    user = _user(_program("geometry", "extremal_principle"))
    assert 'GROUP="geometry"' in user and 'SKILL="extremal_principle"' in user
    # The whole vocabulary is offered, so no single cell is being demanded --
    # asserted on the content rather than on a sentence, because the wording of
    # these prompts is actively being tuned.
    for skill in SKILLS:
        assert f"{skill}:" in user, skill
    assert "read its GROUP and SKILL off the" in user


def test_the_parent_source_is_inlined():
    parent = _program()
    assert "def generate(seed):" in _user(parent)


def test_the_task_carries_the_single_operator():
    task = build_mutation_task(_program())
    assert task.op == MUTATION_OP == "mutate"
    assert [m["role"] for m in task.messages] == ["system", "user"]


def test_the_system_prompt_is_read_from_disk_verbatim():
    """Verbatim really means verbatim: compare against the file, not a phrase."""
    from pathlib import Path

    from rq_evolve.prompts import MUTATION_SYSTEM_PROMPT_FILE, PROMPT_TEMPLATE_DIR

    on_disk = (PROMPT_TEMPLATE_DIR / MUTATION_SYSTEM_PROMPT_FILE).read_text().strip()
    assert mutation_system_prompt() == on_disk
    assert "$" not in on_disk, "the system turn is not templated"


def test_the_system_prompt_carries_the_shape_the_linter_enforces():
    """Two thirds of rejected children died on the assert contract or on the
    module shape. The prompt states both as code, not only as prose."""
    text = mutation_system_prompt()
    assert "def generate(seed):" in text
    assert "GROUP = " in text and "SKILL = " in text
    assert "assert answer == check" in text


# --- template rendering ----------------------------------------------------


def test_no_placeholder_survives_into_a_rendered_prompt():
    for group, skill in (("algebra", "counting"), ("geometry", "induction")):
        task = build_mutation_task(_program(group, skill))
        for message in task.messages:
            assert "$" not in message["content"], message["content"][:200]


def test_an_unsupplied_placeholder_is_an_error_not_a_silent_passthrough():
    """safe_substitute would ship a literal $parent_skill that reads as prose."""
    with pytest.raises(KeyError, match="parent_skill"):
        _render_template("use $parent_skill here", {"parent_group": "algebra"})


def test_a_dollar_sign_from_substituted_content_is_not_flagged():
    rendered = _render_template("$parent_source", {"parent_source": "cost = $5"})
    assert rendered == "cost = $5"


# --- the self-fix conversation ---------------------------------------------


def test_fix_task_replays_the_original_conversation():
    task = build_mutation_task(_program())
    fix = build_fix_task(task, "```python\nbroken\n```", "execute failed at seed=0")

    roles = [m["role"] for m in fix.messages]
    assert roles == ["system", "user", "assistant", "user"]
    assert fix.messages[1]["content"] == task.messages[-1]["content"]
    assert "broken" in fix.messages[2]["content"]
    assert "execute failed at seed=0" in fix.messages[3]["content"]
    assert fix.op == task.op


# --- the judge -------------------------------------------------------------


def test_the_judge_sees_the_problem_and_answer_and_nothing_else():
    messages = build_judge_messages("How many divisors does 5040 have?", "60")
    user = messages[1]["content"]
    assert "How many divisors does 5040 have?" in user
    assert "60" in user
    assert "$" not in user
    assert messages[0]["content"] == judge_system_prompt()


def test_the_judge_rubric_names_both_vocabularies_and_the_output_contract():
    rubric = judge_system_prompt()
    for label in (*GROUPS, *SKILLS):
        assert label in rubric, label
    for field in ("GROUP:", "GROUP_EVIDENCE:", "SKILL:", "SKILL_WITNESS:",
                  "CLOSEST_ALTERNATIVE:", "WHY_NOT_ALTERNATIVE:", "FAILURE_REASON:"):
        assert field in rubric, field


def test_a_well_formed_verdict_parses_into_its_seven_fields():
    verdict = parse_judge_verdict(
        "GROUP: number_theory\n"
        "GROUP_EVIDENCE: congruences decide membership\n"
        "SKILL: casework\n"
        "SKILL_WITNESS: three residue regimes argued differently\n"
        "CLOSEST_ALTERNATIVE: counting\n"
        "WHY_NOT_ALTERNATIVE: no structural counting principle appears\n"
        "FAILURE_REASON: none"
    )
    assert verdict.group == "number_theory" and verdict.skill == "casework"
    assert "congruences" in verdict.group_evidence
    assert "residue" in verdict.skill_witness
    assert verdict.closest_alternative == "counting"


@pytest.mark.parametrize(
    "reply",
    [
        "**GROUP:** number_theory\n**SKILL:** casework",
        "- GROUP : number_theory\n- SKILL : casework",
        "GROUP: `number_theory`\nSKILL: `casework`",
        "GROUP: Number_Theory\nSKILL: CASEWORK",
    ],
)
def test_label_decoration_does_not_decide_the_verdict(reply):
    """A base model decorates the field name far more often than it misjudges."""
    verdict = parse_judge_verdict(reply)
    assert judge_accepts(verdict, "number_theory", "casework")[0] is True


@pytest.mark.parametrize(
    "reply",
    ["GROUP: numbertheory\nSKILL: casework", "GROUP: number_theory\nSKILL: case work",
     "GROUP: none\nSKILL: casework", "", "no idea"],
)
def test_leniency_on_labels_does_not_loosen_the_values(reply):
    verdict = parse_judge_verdict(reply)
    assert judge_accepts(verdict, "number_theory", "casework")[0] is False


def test_both_axes_must_agree():
    good = parse_judge_verdict("GROUP: algebra\nSKILL: invariant")
    assert judge_accepts(good, "algebra", "invariant")[0] is True
    assert judge_accepts(good, "algebra", "counting")[0] is False
    assert judge_accepts(good, "geometry", "invariant")[0] is False
