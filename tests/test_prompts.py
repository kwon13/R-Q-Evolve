from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import (
    build_fix_task,
    build_mutation_task,
    parse_evaluator_verdict,
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


def test_mutation_prompt_carries_no_plan_vocabulary():
    """The mutation prompt is one-stage: nothing may mention a plan or a route."""
    parent = _program("algebra", 3)
    for op in ("in_depth", "in_breadth"):
        conversation = "\n".join(
            message["content"]
            for message in build_mutation_task(op, parent).messages
        ).lower()
        for token in (
            "mutation plan",
            "target_reasoning_move",
            "answer_route",
            "generator_family",
            "reasoning trace",
        ):
            assert token not in conversation, token


def test_fix_task_replays_the_original_conversation():
    parent = _program("algebra", 3)
    task = build_mutation_task("in_depth", parent)
    fix_task = build_fix_task(task, "bad code", "missing answer")

    assert [m["role"] for m in fix_task.messages] == [
        "system",
        "user",
        "assistant",
        "user",
    ]
    assert fix_task.messages[0]["content"] == task.messages[0]["content"]
    assert fix_task.messages[1]["content"] == task.messages[-1]["content"]
    assert fix_task.messages[2]["content"] == "bad code"
    assert "missing answer" in fix_task.messages[3]["content"]
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
