"""Fail-closed local Domain and computational-problem-type guards."""

import pytest

from rq_evolve.code_utils import (
    MAX_PROBLEM_TEXT_CHARS,
    lint_mutation_generator_source,
    lint_problem_instance,
    validated_domain_declaration,
)
from rq_evolve.problem_type import (
    annotate_problem_type,
    problem_type_contract_errors,
)
from rq_evolve.program import ProblemInstance


def _instance(problem: str) -> ProblemInstance:
    return ProblemInstance(problem=problem, answer="42", program_id="p", seed=0)


def _mutation_lint(source: str) -> list[str]:
    return lint_mutation_generator_source(
        source,
        require_assert=False,
        reject_trivial_assert=False,
        reject_unbounded_sampling=False,
        require_answer_routes=False,
        require_canonical_instance_data=False,
        require_mechanical_shape=False,
    )


def test_a_runaway_problem_statement_is_rejected_locally():
    ok = _instance("Count the divisors of 5040. State only the integer.")
    assert lint_problem_instance(ok) == []
    runaway = _instance("Consider " + "1, " * 4000 + "and count them.")
    assert any("too long" in reason for reason in lint_problem_instance(runaway))


def test_problem_length_bound_leaves_normal_statements_far_below_it():
    assert MAX_PROBLEM_TEXT_CHARS >= 4000
    assert lint_problem_instance(_instance("x" * (MAX_PROBLEM_TEXT_CHARS - 1))) == []


def test_one_literal_top_level_domain_declaration_is_accepted():
    source = (
        "import random\n"
        'DOMAIN = "number_theory"\n\n'
        "def generate(seed):\n"
        "    rng = random.Random(seed)\n"
        "    n = rng.randint(1, 9)\n"
        '    return f"Evaluate {n} + 1.", str(n + 1), {"mode": "expression"}\n'
    )
    domain, errors = validated_domain_declaration(source)
    assert domain == "number_theory"
    assert errors == []
    assert _mutation_lint(source) == []


@pytest.mark.parametrize(
    "source, fragment",
    [
        (
            "def generate(seed):\n    return ('Evaluate 1.', '1')\n",
            "exactly one",
        ),
        (
            'DOMAIN = "algebra"\nDOMAIN = "geometry"\n'
            "def generate(seed):\n    return ('Evaluate 1.', '1')\n",
            "exactly one",
        ),
        (
            'DOMAIN = "topology"\n'
            "def generate(seed):\n    return ('Evaluate 1.', '1')\n",
            "must be one of",
        ),
        (
            'name = "algebra"\nDOMAIN = name\n'
            "def generate(seed):\n    return ('Evaluate 1.', '1')\n",
            "must be one of",
        ),
        (
            'DOMAIN = "algebra"\ncopy = DOMAIN\n'
            "def generate(seed):\n    return ('Evaluate 1.', '1')\n",
            "may not be read",
        ),
        (
            'DOMAIN = "algebra"\n'
            "def generate(seed):\n"
            '    DOMAIN = "geometry"\n'
            "    return ('Evaluate 1.', '1')\n",
            "exactly one",
        ),
    ],
)
def test_domain_declaration_fails_closed(source, fragment):
    _domain, errors = validated_domain_declaration(source)
    assert any(fragment in error for error in errors)


@pytest.mark.parametrize(
    "marker",
    (
        'PROBLEM_TYPE = "function"',
        'GROUP = "algebra"',
        'SKILL = "counting"',
        '# PROBLEM_TYPE: function',
    ),
)
def test_generated_source_forbids_non_domain_descriptor_markers(marker):
    source = (
        'DOMAIN = "algebra"\n'
        f"{marker}\n"
        "def generate(seed):\n"
        '    return "Evaluate 1 + 1.", "2", {"mode": "expression"}\n'
    )
    errors = _mutation_lint(source)
    assert any("PROBLEM_TYPE/GROUP/SKILL" in error for error in errors)


def test_nested_problem_type_assignment_is_also_forbidden():
    source = (
        'DOMAIN = "algebra"\n'
        "def generate(seed):\n"
        '    PROBLEM_TYPE = "function"\n'
        '    return "Evaluate 1 + 1.", "2", {"mode": "expression"}\n'
    )
    assert any("PROBLEM_TYPE" in error for error in _mutation_lint(source))


@pytest.mark.parametrize(
    "statement, verifier, answer, expected",
    [
        (
            "Determine whether 17 is prime. Answer Yes or No.",
            {"mode": "boolean"},
            "Yes",
            "decision",
        ),
        (
            "Find all integers x in the finite set {0,1,2,3,4} such that x^2=x.",
            {"mode": "set", "elements": ["0", "1"]},
            "{0,1}",
            "search",
        ),
        (
            "How many positive divisors does 12 have?",
            {"mode": "expression"},
            "6",
            "counting",
        ),
        (
            "Find the maximum value of x(6-x) for 0 <= x <= 6.",
            {"mode": "expression"},
            "9",
            "optimization",
        ),
        (
            "Evaluate 3^4.",
            {"mode": "expression"},
            "81",
            "function",
        ),
    ],
)
def test_statement_and_verifier_assign_each_problem_type_deterministically(
    statement, verifier, answer, expected
):
    first = annotate_problem_type(statement)
    second = annotate_problem_type(statement)
    assert first == second
    assert first.problem_type == expected
    assert first.confidence == "high"
    assert first.evidence
    assert problem_type_contract_errors(first, verifier, answer) == []


@pytest.mark.parametrize(
    "statement, reason",
    [
        ("", "empty_statement"),
        ("Prove that there are infinitely many primes.", "proof_or_justification"),
        ("Find an integer satisfying the stated conditions.", "generic_find_or_determine"),
        ("Let n be a positive integer.", "no_output_contract_cue"),
    ],
)
def test_ambiguous_or_non_exact_requests_abstain(statement, reason):
    annotation = annotate_problem_type(statement)
    assert annotation.problem_type is None
    assert annotation.confidence == "none"
    assert annotation.review_reason == reason
    assert annotation.needs_review is True
    assert problem_type_contract_errors(
        annotation, {"mode": "expression"}, "1"
    )


@pytest.mark.parametrize(
    "statement, verifier, answer, fragment",
    [
        (
            "Determine whether 17 is prime. Answer Yes or No.",
            {"mode": "expression"},
            "Yes",
            "incompatible",
        ),
        (
            "Determine whether 17 is prime. Answer Yes or No.",
            {"mode": "boolean"},
            "1",
            "Yes or No",
        ),
        (
            "Find all integers x such that 0 <= x <= 1.",
            {"mode": "one_of", "answers": ["0", "1"]},
            "0",
            "incompatible",
        ),
        (
            "How many positive divisors does 12 have?",
            {"mode": "expression"},
            "-1",
            "nonnegative integer",
        ),
        (
            "Evaluate 3^4.",
            {"mode": "boolean"},
            "Yes",
            "incompatible",
        ),
    ],
)
def test_statement_verifier_disagreement_fails_closed(
    statement, verifier, answer, fragment
):
    annotation = annotate_problem_type(statement)
    errors = problem_type_contract_errors(annotation, verifier, answer)
    assert any(fragment in error for error in errors)
