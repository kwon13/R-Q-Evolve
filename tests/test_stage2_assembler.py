"""Fail-closed contracts for the data-only stage-2 compiler."""

import pytest

from rq_evolve.code_utils import (
    TRUSTED_ASSEMBLER_VERSION,
    compile_stage2_reply,
    extract_problem_statement_template,
    extract_problem_template,
    lint_generator_source,
)
from rq_evolve.program import ProblemProgram


def _reply(core: str, *, domain: str = "algebra", mode: str = "expression") -> str:
    return (
        f"DOMAIN: {domain}\n"
        f"MODE: {mode}\n"
        "CORE:\n"
        "```python\n"
        f"{core.strip()}\n"
        "```"
    )


def _compile(
    core: str,
    family: str = "Let n = [[n]]. Compute the sum from 1 through n.",
    *,
    domain: str = "algebra",
    mode: str = "expression",
) -> str:
    source, reason = compile_stage2_reply(
        _reply(core, domain=domain, mode=mode), family
    )
    assert source is not None, reason
    assert reason is None
    return source


EXPRESSION_CORE = """
def build_instance(rng):
    n = rng.randint(3, 20)
    answer = n * (n + 1) // 2
    check = sum(range(1, n + 1))
    parameters = {"n": n}
    return parameters, answer, check
"""


BOOLEAN_CORE = """
def build_instance(rng):
    n = rng.randint(3, 20)
    answer = n % 2 == 0
    check = (n // 2) * 2 == n
    parameters = {"n": n}
    return parameters, answer, check
"""


SET_CORE = """
def build_instance(rng):
    n = rng.randint(4, 10)
    answer = [x for x in range(1, n + 1) if x % 2 == 0]
    check = []
    for x in range(n, 0, -1):
        if x % 2 == 0:
            check.append(x)
    parameters = {"n": n}
    return parameters, answer, check
"""


@pytest.mark.parametrize(
    ("mode", "core", "family", "expected_mode"),
    [
        (
            "expression",
            EXPRESSION_CORE,
            "Let n = [[n]]. Compute the sum from 1 through n.",
            "expression",
        ),
        (
            "boolean",
            BOOLEAN_CORE,
            "Is [[n]] even? Answer Yes or No.",
            "boolean",
        ),
        (
            "set",
            SET_CORE,
            "Find all even integers from 1 through [[n]].",
            "set",
        ),
    ],
)
def test_all_modes_compile_and_execute(mode, core, family, expected_mode):
    source = _compile(core, family, mode=mode)
    program = ProblemProgram(source)
    instances = [program.execute(seed) for seed in range(3)]
    assert all(
        instance is not None for instance in instances
    ), program.last_execution_error
    assert {instance.verifier["mode"] for instance in instances} == {expected_mode}
    assert len({instance.problem for instance in instances}) >= 2


def test_set_answer_and_check_are_compared_as_unordered_members():
    source = _compile(
        SET_CORE,
        "Find all even integers from 1 through [[n]].",
        mode="set",
    )
    instance = ProblemProgram(source).execute(0)
    assert instance is not None
    assert instance.verifier["elements"] == sorted(instance.verifier["elements"])


def test_compiled_source_embeds_the_trusted_contract_and_one_domain():
    family = "Let n = [[n]]. Compute the sum from 1 through n."
    source = _compile(EXPRESSION_CORE, family, domain="number_theory")
    assert f"TRUSTED_ASSEMBLER_VERSION = {TRUSTED_ASSEMBLER_VERSION!r}" in source
    assert f"FAMILY_TEMPLATE = {family!r}" in source
    assert source.count("DOMAIN = 'number_theory'") == 1
    assert source.count("def generate(seed):") == 1
    assert "exact built-in data only" in source


def test_compiled_family_template_is_the_next_generation_and_donor_template():
    family = "Let n = [[n]]. Compute the sum from 1 through n."
    source = _compile(EXPRESSION_CORE, family)
    assert extract_problem_template(source) == family
    assert extract_problem_statement_template(source) == family


def test_legacy_problem_template_extraction_is_unchanged():
    source = """
def generate(seed):
    n = seed + 1
    problem = f"Compute {n} plus one."
    return problem, str(n + 1)
"""
    assert extract_problem_template(source) == 'problem = f"Compute {n} plus one."'
    assert extract_problem_statement_template(source) == "Compute {n} plus one."


def test_closed_think_block_is_removed_before_exact_protocol_parsing():
    reply = "<think>private reasoning</think>\n" + _reply(EXPRESSION_CORE)
    source, reason = compile_stage2_reply(
        reply, "Let n = [[n]]. Compute the sum from 1 through n."
    )
    assert source is not None, reason
    assert "private reasoning" not in source


def test_invalid_requires_and_preserves_a_specific_single_line_reason():
    source, reason = compile_stage2_reply(
        "INVALID: supplied family is not finitely gradable", "Compute [[n]]."
    )
    assert source is None
    assert reason.endswith("supplied family is not finitely gradable")

    for malformed in ("INVALID", "INVALID: ", "INVALID: first\nsecond"):
        source, reason = compile_stage2_reply(malformed, "Compute [[n]].")
        assert source is None
        assert "exact DOMAIN/MODE/CORE" in reason


@pytest.mark.parametrize(
    "reply",
    [
        "<think>never closed\n" + _reply(EXPRESSION_CORE),
        _reply(EXPRESSION_CORE) + "\n```python\npass\n```",
        "DOMAIN: algebra\nMODE: expression\nCORE:\n" + EXPRESSION_CORE,
        "DOMAIN: algebra\nMODE: one_of\nCORE:\n```python\npass\n```",
    ],
)
def test_malformed_or_multiple_fence_protocol_fails_closed(reply):
    source, reason = compile_stage2_reply(reply, "Compute [[n]].")
    assert source is None
    assert reason


def test_parser_bomb_and_syntax_error_are_normal_rejections():
    for core in ("[" * 5000, "def build_instance(rng):\n    return ("):
        source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
        assert source is None
        assert reason


@pytest.mark.parametrize(
    ("family", "fragment"),
    [
        ("Compute n.", "at least one"),
        ("Compute [[not valid]].", "malformed"),
        ("Compute [[n].", "malformed"),
    ],
)
def test_family_template_requires_well_formed_placeholders(family, fragment):
    source, reason = compile_stage2_reply(_reply(EXPRESSION_CORE), family)
    assert source is None
    assert fragment in reason


def test_parameters_keys_must_exactly_equal_the_placeholder_set():
    core = EXPRESSION_CORE.replace('{"n": n}', '{"n": n, "extra": n}')
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]] and [[other]].")
    assert source is None
    assert "exactly match family placeholders" in reason


@pytest.mark.parametrize(
    "core",
    [
        EXPRESSION_CORE + "\n\ndef build_instance(rng):\n    pass",
        EXPRESSION_CORE.replace(
            "def build_instance(rng):",
            "def generate(seed):\n    return None\n\ndef build_instance(rng):",
        ),
    ],
)
def test_duplicate_builder_and_generated_generate_shadow_are_rejected(core):
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "build_instance" in reason or "generate" in reason


@pytest.mark.parametrize(
    "signature",
    [
        "@staticmethod\ndef build_instance(rng)",
        "def build_instance(rng=None)",
        "def build_instance(rng: object)",
        "def build_instance(*, rng)",
        "def build_instance(rng, *args)",
    ],
)
def test_builder_signature_is_exact_and_inert(signature):
    core = EXPRESSION_CORE.replace("def build_instance(rng)", signature)
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert reason


@pytest.mark.parametrize(
    "injection",
    [
        "import random as r\n",
        "from random import randint\n",
        "import os\n",
    ],
)
def test_random_and_disallowed_imports_are_rejected_even_through_aliases(injection):
    source, reason = compile_stage2_reply(
        _reply(injection + EXPRESSION_CORE), "Compute [[n]]."
    )
    assert source is None
    assert "not allowed" in reason or "random" in reason


def test_random_import_cannot_hide_inside_the_builder():
    core = EXPRESSION_CORE.replace(
        "n = rng.randint(3, 20)",
        "import random as hidden_random\n    n = hidden_random.randint(3, 20)",
    )
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "random" in reason


def test_nested_duplicate_builder_is_rejected():
    core = (
        EXPRESSION_CORE
        + """

def helper():
    def build_instance(rng):
        return {}, 1, 1
    return build_instance
"""
    )
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "exactly one top-level build_instance" in reason


def test_import_cannot_replace_the_supplied_rng_name():
    core = EXPRESSION_CORE.replace(
        "n = rng.randint(3, 20)",
        "from fractions import Fraction as rng\n    n = 3",
    )
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "rng" in reason


@pytest.mark.parametrize(
    ("old", "new", "fragment"),
    [
        ("n = rng.randint(3, 20)", "n = randint(3, 20)", "random"),
        ("n = rng.randint(3, 20)", "n = random.randint(3, 20)", "random"),
        ("n = rng.randint(3, 20)", "rng = object()\n    n = 3", "rng"),
        ("n = rng.randint(3, 20)", "rng.seed(None)\n    n = 3", "seed"),
        ("n = rng.randint(3, 20)", "rng.setstate(())\n    n = 3", "setstate"),
    ],
)
def test_only_the_worker_supplied_rng_may_drive_randomness(old, new, fragment):
    source, reason = compile_stage2_reply(
        _reply(EXPRESSION_CORE.replace(old, new)), "Compute [[n]]."
    )
    assert source is None
    assert fragment in reason


@pytest.mark.parametrize(
    "name",
    (
        "DOMAIN",
        "PROBLEM_TYPE",
        "GROUP",
        "SKILL",
        "MODE",
        "problem",
        "verifier",
        "__rq_fake",
    ),
)
def test_core_cannot_bind_descriptors_or_trusted_reserved_names(name):
    core = EXPRESSION_CORE.replace(
        "n = rng.randint(3, 20)", f"{name} = 1\n    n = rng.randint(3, 20)"
    )
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert name in reason or "reserved" in reason


def test_parameters_assignment_must_immediately_precede_the_final_return():
    core = EXPRESSION_CORE.replace(
        'parameters = {"n": n}\n    return parameters, answer, check',
        'parameters = {"n": n}\n    answer += 0\n    return parameters, answer, check',
    )
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "immediately precede" in reason


@pytest.mark.parametrize(
    "core",
    [
        EXPRESSION_CORE.replace("check = sum(range(1, n + 1))", "check = answer"),
        EXPRESSION_CORE.replace(
            "answer = n * (n + 1) // 2\n    check = sum(range(1, n + 1))",
            "answer = n + 1\n    check = n + 1",
        ),
    ],
)
def test_candidate_cannot_forge_the_independent_check(core):
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "independent-check contract" in reason


def test_loop_bound_placeholder_is_connected_to_loop_carried_answer():
    core = """
def build_instance(rng):
    string_length = rng.randint(4, 11)
    previous, current = 1, 2
    for _ in range(string_length - 1):
        previous, current = current, previous + current
    answer = current
    check = 0
    for mask in range(1 << string_length):
        if all(((mask >> position) & 3) != 3 for position in range(string_length - 1)):
            check += 1
    parameters = {"string_length": string_length}
    return parameters, answer, check
"""
    source, reason = compile_stage2_reply(
        _reply(core),
        "Compute the number of binary strings of length [[string_length]] "
        "that contain no two consecutive 1s.",
    )
    assert reason is None
    assert source is not None


def test_a_bare_sampled_answer_is_rejected_even_with_a_different_check_shape():
    core = """
def build_instance(rng):
    n = rng.randint(3, 20)
    answer = n
    check = sum([n])
    parameters = {"n": n}
    return parameters, answer, check
"""
    source, reason = compile_stage2_reply(_reply(core), "Compute [[n]].")
    assert source is None
    assert "bare-answer contract" in reason


def test_callable_nested_in_plain_container_is_rejected_before_rendering():
    core = """
def helper():
    return 1

def build_instance(rng):
    n = rng.randint(3, 20)
    items = [helper, n]
    answer = len(items)
    check = sum(1 for _ in items)
    parameters = {"items": items}
    return parameters, answer, check
"""
    source = _compile(core, "The list is [[items]]. Compute its length.")
    program = ProblemProgram(source)
    assert program.execute(0) is None
    assert "exact built-in data only" in str(program.last_execution_error)


def test_custom_imported_object_is_rejected_before_its_string_conversion():
    core = """
from fractions import Fraction

def build_instance(rng):
    n = rng.randint(3, 20)
    value = Fraction(n, 2)
    answer = n + 1
    check = sum((n, 1))
    parameters = {"value": value}
    return parameters, answer, check
"""
    source = _compile(core, "Let q = [[value]]. Compute twice q plus one.")
    assert source.index("__rq_validate_plain(parameters)") < source.index(
        "problem_parts = []"
    )
    program = ProblemProgram(source)
    assert program.execute(0) is None
    assert "exact built-in data only" in str(program.last_execution_error)


def test_payload_bounds_are_enforced_at_runtime():
    core = """
def build_instance(rng):
    n = rng.randint(3, 20)
    text = "x" * 2049
    answer = len(text)
    check = sum(1 for _ in text)
    parameters = {"text": text}
    return parameters, answer, check
"""
    source = _compile(core, "The text is [[text]]. Compute its length.")
    program = ProblemProgram(source)
    assert program.execute(0) is None
    assert "string is too long" in str(program.last_execution_error)


def test_lint_generator_source_rejects_duplicate_runtime_entrypoints():
    source = (
        'def generate(seed):\n    return "Compute one.", "1"\n\n'
        'def generate(seed):\n    return "Compute two.", "2"\n'
    )
    assert any("exactly one" in reason for reason in lint_generator_source(source))
