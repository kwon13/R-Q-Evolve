

def test_the_parents_label_lines_are_deleted():
    """Whatever sits in the parent's tail is what the child copies.

    Real labels -> 97% of children reproduced the parent's cell.
    `GROUP = "..."` -> children declared the cell `...`.
    The skeleton placeholder -> children declared `<one of the allowed GROUPS>`.
    Deletion is the only form that cannot be copied; it is safe only because
    PART 1 now commits to both labels before any code is written.
    """
    from rq_evolve.code_utils import strip_label_declarations

    source = (
        "import random\n\n\n"
        "def generate(seed):\n"
        '    return "q", "1"\n\n\n'
        'GROUP = "algebra"\n'
        'SKILL = "invariant"\n'
    )
    out = strip_label_declarations(source)
    assert "GROUP" not in out and "SKILL" not in out
    assert out.rstrip().endswith('return "q", "1"')
    assert "def generate(seed):" in out and "import random" in out


def test_stripping_labels_leaves_unparseable_source_alone():
    from rq_evolve.code_utils import strip_label_declarations

    broken = "def generate(seed:\n    pass\n"
    assert strip_label_declarations(broken) == broken


def _bomb_source() -> str:
    """Mixed nesting ("[1,[1,[1,...") explodes CPython's PEG-parser arena and
    raises MemoryError with host RAM to spare -- pure paren runs get the cheap
    "too many nested parentheses" SyntaxError, but this shape does not. One
    such reply killed the 2026-08-23 run mid-iteration."""
    return (
        "def generate(seed):\n"
        "    x = " + "[1," * 50_000 + "1" + "]" * 50_000 + "\n"
        "    return \"q\", \"1\"\n"
    )


def test_a_parser_bomb_is_a_failed_extraction_not_a_dead_trainer():
    """A degenerate generation must cost its own candidate and nothing else."""
    from rq_evolve.code_utils import extract_generator_code

    assert extract_generator_code(f"```python\n{_bomb_source()}```\n") is None


def test_a_parser_bomb_is_absorbed_at_every_downstream_gate():
    """The extraction guard alone is defense by call order: had the same reply
    first reached the AST contract, the lint, or -- worst -- the label read on
    an archive-restored champion, the trainer would have died the same way,
    and a bomb inside a snapshot would replay the crash on every restart.
    Every parse now routes through safe_ast_parse, so each gate reports the
    source as unparseable instead."""
    from rq_evolve.ast_contract import check_generator_contract
    from rq_evolve.code_utils import lint_generator_source, strip_label_declarations
    from rq_evolve.program import ProblemProgram

    bomb = _bomb_source() + '\nGROUP = "algebra"\nSKILL = "counting"\n'

    # The archive-restore path: labels are read straight off the source.
    restored = ProblemProgram(source_code=bomb)
    assert restored.declared_group() is None
    assert restored.declared_skill() is None

    assert check_generator_contract(bomb) == []
    assert any("syntax error" in r for r in lint_generator_source(bomb))
    assert strip_label_declarations(bomb) == bomb  # unchanged, not raised
