from rq_evolve.problem_type import (
    PROBLEM_TYPES,
    annotate_problem_type,
    integer_answer,
    problem_type_contract_errors,
    top_level_domains,
)


def test_problem_type_vocabulary_is_the_reference_five_way_split():
    assert PROBLEM_TYPES == (
        "decision",
        "search",
        "counting",
        "optimization",
        "function",
    )


def test_statement_only_problem_type_examples():
    examples = {
        "Does there exist an integer n satisfying n^2 = 2?": "decision",
        "Find all integers n such that n^2 = 9.": "search",
        "How many integers n satisfy 1 <= n <= 20?": "counting",
        "Find the smallest positive integer divisible by 6 and 15.": "optimization",
        "Compute the remainder when 2^20 is divided by 7.": "function",
    }
    for statement, expected in examples.items():
        result = annotate_problem_type(statement)
        assert result.problem_type == expected, (statement, result)
        assert not result.needs_review


def test_specific_output_contract_has_precedence():
    assert (
        annotate_problem_type("Find the maximum number of selected vertices.").problem_type
        == "optimization"
    )
    assert (
        annotate_problem_type("Is there a maximum value of this expression?").problem_type
        == "decision"
    )
    assert (
        annotate_problem_type("How many possible values are there?").problem_type
        == "counting"
    )
    assert (
        annotate_problem_type(
            "How many graphs are there on ten labeled vertices?"
        ).problem_type
        == "counting"
    )
    assert (
        annotate_problem_type(
            "Find all integers n such that a polynomial has degree at least 1."
        ).problem_type
        == "search"
    )
    assert (
        annotate_problem_type(
            "The minimum area is 4 and the maximum area is 9. Compute their sum."
        ).problem_type
        == "function"
    )
    assert (
        annotate_problem_type(
            "After what least number of moves can it happen that all values differ?"
        ).problem_type
        == "optimization"
    )
    assert (
        annotate_problem_type(
            "Find the smallest positive integer n such that n^2 is divisible by 72."
        ).problem_type
        == "optimization"
    )
    assert (
        annotate_problem_type(
            "For which positive integers n is there a solution to the equation?"
        ).problem_type
        == "search"
    )


def test_named_gcd_lcm_terms_are_function_requests_not_optimization():
    cases = (
        "Determine the greatest common divisor of 84 and 126.",
        "Find the greatest common factor of 84 and 126.",
        "Compute the least common multiple of 12 and 18.",
    )
    for statement in cases:
        annotation = annotate_problem_type(statement)
        assert annotation.problem_type == "function", (statement, annotation)

    # A genuine extremization of a gcd remains optimization.
    assert (
        annotate_problem_type(
            "Find the greatest possible common divisor of two distinct integers "
            "whose sum is 60."
        ).problem_type
        == "optimization"
    )


def test_nested_request_words_do_not_override_the_outer_contract():
    cases = {
        "Find the maximum number of solutions to the equation.": "optimization",
        "Find all x such that, after you compute x^2, it is odd.": "search",
        "Is the number of integer solutions even?": "decision",
        "What is the number of valid arrangements?": "counting",
    }
    for statement, expected in cases.items():
        assert annotate_problem_type(statement).problem_type == expected


def test_statement_type_is_cross_checked_with_verifier_and_answer():
    decision = annotate_problem_type("Is 10 even? Answer Yes or No.")
    assert problem_type_contract_errors(
        decision, {"mode": "boolean"}, "Yes"
    ) == []
    assert "incompatible" in problem_type_contract_errors(
        decision, {"mode": "expression"}, "1"
    )[0]

    search = annotate_problem_type("Find all integers x with x^2 = 1.")
    assert problem_type_contract_errors(
        search, {"mode": "set", "elements": ["-1", "1"]}, r"\{-1,1\}"
    ) == []
    assert "incompatible" in problem_type_contract_errors(
        search, {"mode": "expression"}, "1"
    )[0]

    count = annotate_problem_type("How many integers are in the interval?")
    assert problem_type_contract_errors(
        count, {"mode": "expression"}, "12"
    ) == []
    assert "nonnegative integer" in problem_type_contract_errors(
        count, {"mode": "expression"}, "1/2"
    )[0]


def test_proof_and_generic_requests_abstain():
    proof = annotate_problem_type("Prove that there are infinitely many primes.")
    assert proof.needs_review and proof.review_reason == "proof_or_justification"

    generic = annotate_problem_type("Determine x.")
    assert generic.needs_review and generic.review_reason == "generic_find_or_determine"


def test_omni_top_level_domains_are_multilabel_and_deduplicated():
    paths = [
        "Mathematics -> Algebra -> Inequalities",
        "Mathematics -> Discrete Mathematics -> Combinatorics",
        "Mathematics -> Algebra -> Other",
        "Mathematics -> Other",
    ]
    assert top_level_domains(paths) == ("Algebra", "Discrete Mathematics")


def test_current_generated_answer_integer_gate():
    assert integer_answer("-17")
    assert not integer_answer(r"\frac{1}{2}")
    assert not integer_answer(r"\text{Yes}")
