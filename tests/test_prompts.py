from pathlib import Path

from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    build_fix_task,
    build_metacognitive_plan_task,
    build_mutation_task,
    build_plain_plan_task,
    build_planned_mutation_task,
    parse_evaluator_verdict,
    parse_mutation_plan,
)


def _program(group: str, value: int) -> ProblemProgram:
    return ProblemProgram(
        source_code=f'''
def generate(seed):
    return "What is {value} + {{seed}}?", str({value} + seed)


CONCEPT_GROUP = "{group}"
CONCEPT_TYPE = "{group}.toy"
'''
    )


def test_in_depth_template_uses_parent_source():
    parent = _program("algebra", 3)
    task = build_mutation_task("in_depth", parent)
    assert "Task: write a deeper variant" in task.prompt
    assert "PARENT_PROGRAM_EXAMPLE" not in task.prompt
    assert "def generate(seed)" in task.prompt
    assert "algebra" in task.prompt
    assert [message["role"] for message in task.messages] == ["system", "user"]
    assert "Think silently" in task.messages[0]["content"]
    assert "Required structural skeleton" in task.messages[0]["content"]
    assert "for _ in range(MAX_ATTEMPTS)" in task.messages[0]["content"]
    assert "answer = <one executable integer-valued computation>" in task.messages[0]["content"]
    assert "instance_data" in task.messages[0]["content"]
    assert "answer_insight" not in task.prompt
    assert "answer_brute" not in task.prompt
    assert "brute" not in task.prompt.lower()
    assert "PARENT_PROGRAM_EXAMPLE" not in task.messages[-1]["content"]
    assert task.messages[-1]["content"].count('CONCEPT_GROUP = "algebra"') == 1
    assert task.messages[-1]["content"].count('CONCEPT_TYPE = "algebra.toy"') == 1
    assert task.messages[-1]["content"].rstrip().endswith("CHILD_PROGRAM:")


def test_in_breadth_template_uses_breadth_shots():
    parent = _program("algebra", 3)
    task = build_mutation_task("in_breadth", parent)
    assert "Task: write a generator in a different mathematical domain" in task.prompt
    assert "CHILD_PROGRAM_EXAMPLE" not in task.prompt
    assert "chord geometry" not in task.prompt
    assert "CHILD_PROGRAM_EXAMPLE" not in task.messages[-1]["content"]
    assert "answer_insight" not in task.prompt
    assert "answer_brute" not in task.prompt
    assert "brute" not in task.prompt.lower()


def test_metacognitive_plan_prompt_uses_single_answer_route_schema():
    parent = _program("algebra", 3)
    for op in ("in_depth", "in_breadth"):
        task = build_metacognitive_plan_task(
            op,
            parent,
            evidence=[
                {"role": "success", "response": "correct step"},
                {"role": "failure", "response": "wrong step"},
            ],
            meta_progress={},
        )
        conversation = "\n".join(message["content"] for message in task.messages)

        assert [message["role"] for message in task.messages] == ["system", "user"]
        assert "schema_version 5" in conversation
        assert "generator_family" in conversation
        assert "family_config" in conversation
        assert "registered" in conversation.lower()
        assert "why_target_reasoning_move_is_necessary" in conversation
        assert "mathematically or information-critical" in conversation
        assert "Target label contract:" in conversation
        assert "Target-move necessity contract:" in conversation
        assert "target_concept_group" in conversation
        assert "answer_route" in conversation
        assert "PARENT_PROGRAM_EXAMPLE" not in conversation
        assert "boundary plus one" not in conversation
        assert "decoy" not in conversation.lower()
        assert "brute_route" not in conversation
        assert "decoy_route" not in conversation
        assert "answer_brute" not in conversation


def test_plain_and_reasoning_planners_share_schema_and_sampling_contract():
    parent = _program("algebra", 3)
    plain = build_plain_plan_task(
        "in_depth",
        parent,
        meta_progress={},
        max_output_tokens=1024,
        temperature=0.3,
        top_p=0.95,
    )
    reasoning = build_metacognitive_plan_task(
        "in_depth",
        parent,
        evidence=[
            {
                "role": "success",
                "response": r"Correct: \boxed{3}.",
                "predicted_answer": "3",
            },
            {
                "role": "failure",
                "response": r"Wrong: \boxed{4}.",
                "predicted_answer": "4",
            },
        ],
        meta_progress={},
        max_output_tokens=1024,
        temperature=0.3,
        top_p=0.95,
    )

    assert plain.max_output_tokens == reasoning.max_output_tokens == 1024
    assert plain.temperature == reasoning.temperature == 0.3
    assert plain.top_p == reasoning.top_p == 0.95
    for field in (
        '"schema_version": 5',
        "generator_family",
        "family_config",
        "target_reasoning_move",
        "target_concept_group",
        "target_concept_type",
        "why_target_reasoning_move_is_necessary",
        "answer_route",
    ):
        assert field in plain.prompt
        assert field in reasoning.prompt
    assert "No Solver evidence is available" in plain.prompt
    assert r"\boxed{3}" not in plain.prompt
    assert r"\boxed{3}" in reasoning.prompt


def test_breadth_plan_prompt_forbids_parent_group_and_fixes_registered_route():
    parent = _program("algebra", 3)
    for task in (
        build_plain_plan_task("in_breadth", parent, meta_progress={}),
        build_metacognitive_plan_task(
            "in_breadth",
            parent,
            evidence=[
                {"role": "success", "response": "correct"},
                {"role": "failure", "response": "wrong"},
            ],
            meta_progress={},
        ),
    ):
        conversation = "\n".join(message["content"] for message in task.messages)
        assert 'target_concept_group "algebra" is forbidden' in conversation
        assert '"generator_family": "modular_linear_system_aggregate"' in conversation
        assert 'target_concept_group="number_theory"' in conversation
        assert (
            'target_concept_type="number_theory.modular_linear_system_sum"'
            in conversation
        )
        allowed_clause = conversation.split(
            'target_concept_group "algebra" is forbidden.',
            1,
        )[1].split(".", 1)[0]
        assert "algebra" not in allowed_clause


def test_planned_evaluator_requires_explicit_target_move_necessity():
    valid, reason = parse_evaluator_verdict(
        "reason: coherent\n"
        "target_move_required: YES\n"
        "verdict: VALID",
        require_target_move=True,
    )
    assert valid is True, reason

    for output in (
        "reason: coherent\nverdict: VALID",
        "reason: shortcut works\n"
        "target_move_required: NO\n"
        "verdict: VALID",
    ):
        valid, _ = parse_evaluator_verdict(
            output,
            require_target_move=True,
        )
        assert valid is False


def test_planned_code_prompt_has_no_comparison_or_decoy_contract():
    parent = _program("algebra", 3)
    for op in ("in_depth", "in_breadth"):
        shot = Path(
            f"prompt_templates/shots/metacognitive_{op}.txt"
        ).read_text(encoding="utf-8")
        plan, reason = parse_mutation_plan(shot, op)
        assert plan is not None, reason

        task = build_planned_mutation_task(op, parent, plan)
        conversation = "\n".join(message["content"] for message in task.messages)

        assert "single executable `answer_route`" in conversation
        assert "instance_data" in conversation
        assert "stale aliases" in conversation
        assert "brute" not in conversation.lower()
        assert "decoy" not in conversation.lower()
        assert "answer_brute" not in conversation
        assert "brute_route" not in conversation
        assert "decoy_route" not in conversation

        fix_task = build_fix_task(task, "bad code", "missing answer")
        fix_conversation = "\n".join(
            message["content"] for message in fix_task.messages
        )
        assert "answer_brute" not in fix_conversation
        assert "decoy_route" not in fix_conversation
