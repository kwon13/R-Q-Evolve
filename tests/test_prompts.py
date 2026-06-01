from rq_evolve.program import ProblemProgram
from rq_evolve.prompts import build_mutation_task


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
    assert "PARENT_PROGRAM_EXAMPLE" in task.prompt
    assert "def generate(seed)" in task.prompt
    assert "algebra" in task.prompt


def test_in_breadth_template_uses_breadth_shots():
    parent = _program("algebra", 3)
    task = build_mutation_task("in_breadth", parent)
    assert "Task: write a generator in a different mathematical domain" in task.prompt
    assert "MUTATED_PROGRAM_EXAMPLE" in task.prompt
    assert "chord geometry" in task.prompt


def test_crossover_template_uses_both_parent_sources():
    parent = _program("algebra", 3)
    parent_b = _program("geometry", 4)
    task = build_mutation_task("crossover", parent, parent_b)
    assert "Parent A" in task.prompt
    assert "Parent B" in task.prompt
    assert "CROSSOVER_CHILD_PROGRAM_EXAMPLE" in task.prompt
    assert "geometry" in task.prompt
