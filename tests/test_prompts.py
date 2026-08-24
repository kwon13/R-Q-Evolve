"""The two-stage mutation prompts and the judge contract.

Single-stage mutation (one prompt that rewrote the whole program) is gone,
and with it ``build_mutation_task``/``build_fix_task`` and their templates.
Mutation is now two calls: stage 1 (``build_family_task``) mutates the problem
FAMILY in prose and commits to GROUP/SKILL, stage 2 (``build_generator_task``)
writes the generator for that fixed family with the parent inlined as a worked
example of the statement-to-program mapping. These tests pin what each stage
may see -- the vocabularies, the parent's problem, the parent's label-stripped
source -- and what neither stage may see: the parent's own cell.
"""

import pytest

from rq_evolve.concepts import GROUPS, SKILLS
from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    MUTATION_OP,
    _render_template,
    _template_context,
    build_family_task,
    build_generator_task,
    build_judge_messages,
    judge_accepts,
    judge_system_prompt,
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


def _plan() -> dict:
    return {
        "CHILD FAMILY": "Let n = {n}. How many? State only the integer.",
        "STRUCTURAL MUTATION": "a different target",
        "GROUP": "geometry",
        "SKILL": "casework",
    }


def _family_user(parent) -> str:
    return build_family_task(parent).messages[-1]["content"]


def _gen_user(parent) -> str:
    return build_generator_task(parent, _plan()).messages[-1]["content"]


# --- stage 1: the problem, in prose ----------------------------------------


def test_the_child_may_read_every_label_it_may_choose():
    """A label the model can pick but not read a definition for is
    unfalsifiable, so stage 1 carries both full vocabularies."""
    user = _family_user(_program("algebra", "transformation"))
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


def test_stage_one_shows_the_problem_and_no_code():
    """The object being mutated is a PROBLEM; the program only emits it.

    Showing source framed the task as editing code, and a base model shown
    code rewrites code -- child/parent source similarity sat at 0.99 under
    whole-program rewriting. Stage 1 therefore sees the family and one
    rendered instance, and no program at all.
    """
    parent = _program()
    user = _family_user(parent)
    statement = parent.execute(seed=0).problem
    assert statement.strip() in user
    assert "def generate" not in user


def test_the_parents_cell_never_reaches_either_stage():
    """Showing the parent's labels anchored the child to them: children
    declared the parent's own cell 96% of the time. Stage 1 carries no label
    declarations; stage 2 shows the source with its tail stripped, and its
    system prompt says the labels are added afterwards."""
    parent = _program("geometry", "extremal_principle")
    family_user = _family_user(parent)
    assert 'GROUP = "' not in family_user and 'SKILL = "' not in family_user

    task = build_generator_task(parent, _plan())
    gen_user, gen_system = task.messages[1]["content"], task.messages[0]["content"]
    assert 'GROUP = "' not in gen_user and 'SKILL = "' not in gen_user
    assert "Do not write GROUP or SKILL assignment lines" in gen_system


def test_the_parent_source_is_inlined_in_stage_two():
    assert "def generate(seed):" in _gen_user(_program())


def test_the_fixed_child_family_reaches_stage_two_verbatim():
    """Stage 2 does not re-decide the problem; it receives stage 1's family."""
    assert _plan()["CHILD FAMILY"] in _gen_user(_program())


def test_a_parent_that_cannot_run_still_produces_a_prompt():
    """A resume can load a snapshot whose source no longer executes here. That
    is not something to discover while building a prompt."""
    broken = ProblemProgram(
        source_code='GROUP = "algebra"\nSKILL = "invariant"\n'  # no generate()
    )
    assert "did not run here" in _family_user(broken)


def test_both_stages_carry_the_single_operator():
    family = build_family_task(_program())
    generator = build_generator_task(_program(), _plan())
    assert family.op == generator.op == MUTATION_OP == "mutate"
    for task, stage in ((family, "family"), (generator, "generator")):
        assert [m["role"] for m in task.messages] == ["system", "user"]
        assert task.stage == stage


def test_the_system_prompts_are_properly_rendered():
    """Stage 1 is verbatim; Stage 2 renders skill definitions."""
    from rq_evolve.prompts import (
        FAMILY_SYSTEM_PROMPT_FILE,
        GENERATOR_SYSTEM_PROMPT_FILE,
        PROMPT_TEMPLATE_DIR,
    )

    family_task = build_family_task(_program())
    on_disk_family = (PROMPT_TEMPLATE_DIR / FAMILY_SYSTEM_PROMPT_FILE).read_text()
    assert family_task.messages[0]["content"] == on_disk_family

    gen_task = build_generator_task(_program(), _plan())
    gen_system = gen_task.messages[0]["content"]
    assert "$" not in gen_system
    assert "INFERRED_SKILL:" in gen_system
    for skill in SKILLS:
        assert f"{skill}:" in gen_system


def test_stage_one_asks_for_its_labelled_lines():
    """The labels are committed in stage 1, while the solution is still in
    view, and the harness staples them onto the program afterwards
    (set_label_declarations).

    WHY FINITE is asked for but NOT required by the parser: 31% of archived
    champions were judged ill-posed, nearly all of them a "find the maximum X"
    whose bounding clause went missing in the mutation, and naming that clause
    is the cheapest way to make the model notice. It stays optional because
    stage-1 parse failures are already the largest single loss.
    """
    system = build_family_task(_program()).messages[0]["content"]
    for line in ("STRUCTURAL MUTATION:", "CHILD FAMILY:", "WHY FINITE:",
                 "GROUP:", "SKILL:"):
        assert line in system, line


def test_the_finiteness_field_is_parsed_but_never_required():
    from rq_evolve.prompts import parse_family_plan

    without = parse_family_plan(
        "STRUCTURAL MUTATION: a different target\n"
        "CHILD FAMILY: Let n = {n}. How many? \n"
        "GROUP: geometry\nSKILL: casework"
    )
    assert without is not None and "WHY FINITE" not in without

    withit = parse_family_plan(
        "STRUCTURAL MUTATION: a different target\n"
        "CHILD FAMILY: Let n = {n}. How many? \n"
        "WHY FINITE: the set of n items is finite\n"
        "GROUP: geometry\nSKILL: casework"
    )
    # And the header must be a field boundary, or the prose is swallowed into
    # CHILD FAMILY and silently becomes part of the child's problem statement.
    assert withit["WHY FINITE"].startswith("the set of n items")
    assert "WHY FINITE" not in withit["CHILD FAMILY"]


def test_stage_two_states_the_shape_the_linter_enforces():
    """Two thirds of rejected children died on the assert contract or on the
    module shape. The stage-2 prompt states both as code, not only as prose."""
    system = build_generator_task(_program(), _plan()).messages[0]["content"]
    assert "def generate(seed):" in system
    assert "assert answer == check" in system
    assert "```python" in system


# --- template rendering ----------------------------------------------------


def test_no_placeholder_survives_into_a_rendered_prompt():
    for group, skill in (("algebra", "counting"), ("geometry", "induction")):
        parent = _program(group, skill)
        for task in (build_family_task(parent), build_generator_task(parent, _plan())):
            for message in task.messages:
                assert "$" not in message["content"], message["content"][:200]


def test_an_unsupplied_placeholder_is_an_error_not_a_silent_passthrough():
    """safe_substitute would ship a literal $parent_skill that reads as prose."""
    with pytest.raises(KeyError, match="parent_skill"):
        _render_template("use $parent_skill here", {"parent_group": "algebra"})


def test_a_dollar_sign_from_substituted_content_is_not_flagged():
    rendered = _render_template("$parent_source", {"parent_source": "cost = $5"})
    assert rendered == "cost = $5"


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


def test_part_one_labels_are_read_back_off_the_reply():
    """The program the model returns has no labels; these are where they live."""
    from rq_evolve.prompts import parse_declared_labels

    reply = (
        "PART 1\n\nMUTATION:\nPreserve: the grid\n\n"
        "CHILD PROBLEM:\nHow many? State only the integer.\n\n"
        "GROUP: geometry\nSKILL: extremal_principle\n\n"
        "---\n\nPART 2\n\n```python\ndef generate(seed):\n    pass\n```\n"
    )
    assert parse_declared_labels(reply) == ("geometry", "extremal_principle")


def test_labels_outside_the_vocabulary_are_not_invented():
    """verify_program reports them through validate_label_decl; guessing here
    would hide a child that never chose a legal label."""
    from rq_evolve.prompts import parse_declared_labels

    assert parse_declared_labels("GROUP: arithmetic\nSKILL: substitution") == (None, None)
    assert parse_declared_labels("no labels at all") == (None, None)


def test_parse_inferred_labels_extracts_skill():
    """Stage 2 outputs INFERRED_SKILL after the code block."""
    from rq_evolve.prompts import parse_inferred_labels

    reply = "```python\ndef generate(seed):\n    pass\n```\nINFERRED_SKILL: casework\n"
    assert parse_inferred_labels(reply) == (None, "casework")

    # With decoration and case
    assert parse_inferred_labels("INFERRED_SKILL: `induction`") == (None, "induction")
    assert parse_inferred_labels("INFERRED_SKILL: \"invariant\"") == (None, "invariant")
    assert parse_inferred_labels("INFERRED_SKILL: extremal_principle") == (None, "extremal_principle")

    # Invalid / missing
    assert parse_inferred_labels("INFERRED_SKILL: unknown_technique") == (None, None)
    assert parse_inferred_labels("```python\ncode\n```") == (None, None)
