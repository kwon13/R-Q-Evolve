from pathlib import Path

from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    build_metacognitive_plan_task,
    build_mutation_task,
    build_planned_mutation_task,
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


def test_metacognitive_plan_prompt_uses_single_answer_route_schema():
    parent = _program("algebra", 3)
    task = build_metacognitive_plan_task(
        "in_depth",
        parent,
        evidence=[
            {"role": "success", "response": "correct step"},
            {"role": "failure", "response": "wrong step"},
        ],
        meta_progress={},
    )
    conversation = "\n".join(message["content"] for message in task.messages)

    assert "schema_version 3" in conversation
    assert "answer_route" in conversation
    assert "brute" not in conversation.lower()
    assert "decoy" not in conversation.lower()
    assert "brute_route" not in conversation
    assert "decoy_route" not in conversation
    assert "answer_brute" not in conversation


def test_planned_code_prompt_has_no_comparison_or_decoy_contract():
    parent = _program("algebra", 3)
    shot = Path(
        "prompt_templates/shots/metacognitive_in_depth.txt"
    ).read_text(encoding="utf-8")
    plan, reason = parse_mutation_plan(shot, "in_depth")
    assert plan is not None, reason

    task = build_planned_mutation_task("in_depth", parent, plan)
    conversation = "\n".join(message["content"] for message in task.messages)

    assert "single executable `answer_route`" in conversation
    assert "brute" not in conversation.lower()
    assert "decoy" not in conversation.lower()
    assert "answer_brute" not in conversation
    assert "brute_route" not in conversation
    assert "decoy_route" not in conversation
