

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


def test_a_parser_bomb_is_a_failed_extraction_not_a_dead_trainer():
    """Mixed nesting ("[1,[1,[1,..." for thousands of tokens) explodes
    CPython's PEG-parser arena and raises MemoryError with host RAM to spare --
    pure paren runs get the cheap "too many nested parentheses" SyntaxError,
    but this shape does not. One such reply killed the 2026-08-23 run mid-
    iteration. A degenerate generation must cost its own candidate and nothing
    else."""
    from rq_evolve.code_utils import extract_generator_code

    bomb = (
        "```python\n"
        "def generate(seed):\n"
        "    x = " + "[1," * 50_000 + "1" + "]" * 50_000 + "\n"
        "    return \"q\", \"1\"\n"
        "```\n"
    )
    assert extract_generator_code(bomb) is None
