"""The structural contract on a generator's answer cross-check.

Every defect snippet here is reduced from a program that actually reached the
archive in ``rq_output/rq_evolve_base_8b``, and every tombstone records a rule
that was tried, measured against the clean corpus, and rejected.
"""

from __future__ import annotations

import ast
import glob
import re
from pathlib import Path

import pytest

from rq_evolve.ast_contract import _rename, check_generator_contract, check_problem_text

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "mutation_pairs"


def _codes(source: str) -> list[str]:
    return [finding.code for finding in check_generator_contract(source)]


# The champion that motivated this module: rank 1 by R_Q (54.4) in the final
# archive of the 105-iteration run. The comment claims an independent route and
# the two branches are identical, so the assert holds on every seed.
VACUOUS_CHAMPION = '''
import random

GROUP = "number_theory"
SKILL = "transformation"


def generate(seed):
    rng = random.Random(seed)
    p = rng.choice([3, 5, 7, 11, 13, 17])
    n = rng.randint(2, 10)

    if n >= p:
        answer = 1
    else:
        answer = 0

    # Independent route: consider Q(x) = P(x) - x^2.
    if n >= p:
        check = 1
    else:
        check = 0
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Consider polynomials P(x) of degree {n} such that P(k) = k^2 mod {p}. "
        f"State only the integer."
    )
    return problem, str(answer)
'''

SOUND_CROSS_CHECK = '''
import random

GROUP = "sequence"
SKILL = "induction"


def generate(seed):
    rng = random.Random(seed)
    p = rng.choice([2, 3])
    q = rng.randint(1, 5)
    c = rng.randint(1, 5)
    n = rng.randint(6, 10)

    val = c
    for _ in range(n):
        val = p * val + q
    answer = val

    # Closed form instead of iterating the recurrence.
    check = p ** n * c + q * (p ** n - 1) // (p - 1)
    assert answer == check, f"answer={answer} check={check}"

    problem = f"a(0) = {c}, a(n) = {p} * a(n-1) + {q}. Find a({n})."
    return problem, str(answer)
'''


def test_a_check_that_recomputes_the_answer_the_same_way_is_not_a_check():
    """The rank-1 champion by R_Q carried an assert that proved nothing.

    53 of the 151 programs that entered the archive in that run had this shape,
    and five of the top ten by R_Q. The fitness function cannot see it (median
    R_Q 22.5 against 20.7) and the evaluator passed it with the full source in
    its prompt.
    """
    assert "A3v" in _codes(VACUOUS_CHAMPION)


def test_single_iterable_minmax_is_not_folded_as_an_idempotent_call():
    expression = ast.parse("max(range(n))", mode="eval").body
    normalized = _rename(expression, {})
    assert isinstance(normalized, ast.Call)
    assert isinstance(normalized.func, ast.Name)
    assert normalized.func.id == "max"


def test_a_genuinely_independent_cross_check_survives():
    """Without this the rule would be vacuous -- it would reject everything."""
    assert check_generator_contract(SOUND_CROSS_CHECK) == []


def test_duplicate_symbolic_differentiation_is_still_vacuous():
    source = '''
import random
import sympy


def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 5)
    b = rng.randint(-5, 5)
    x0 = rng.randint(-3, 3)
    x = sympy.symbols("x")
    polynomial = a * x ** 2 + b * x
    answer = int(sympy.diff(polynomial, x).subs(x, x0))
    check = int(sympy.diff(polynomial, x).subs(x, x0))
    assert answer == check
    return f"Compute the derivative at {x0}.", str(answer)
'''
    assert "A3v" in _codes(source)


def test_coefficient_formula_and_symbolic_derivative_are_independent():
    source = '''
import random
import sympy


def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 5)
    b = rng.randint(-5, 5)
    c = rng.randint(-5, 5)
    x0 = rng.randint(-3, 3)
    answer = 2 * a * x0 + b
    x = sympy.symbols("x")
    polynomial = a * x ** 2 + b * x + c
    check = int(sympy.diff(polynomial, x).subs(x, x0))
    assert answer == check
    return f"Compute the derivative at {x0}.", str(answer)
'''
    assert check_generator_contract(source) == []


def test_fraction_antiderivative_and_symbolic_integral_are_independent():
    source = '''
import random
import sympy
from fractions import Fraction


def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(1, 4)
    b = rng.randint(-4, 4)
    left = rng.randint(-3, 0)
    right = rng.randint(1, 4)
    exact = Fraction(a * (right ** 3 - left ** 3), 3)
    exact += Fraction(b * (right ** 2 - left ** 2), 2)
    answer = str(exact)
    x = sympy.symbols("x")
    symbolic = sympy.integrate(a * x ** 2 + b * x, (x, left, right))
    check = str(symbolic)
    assert answer == check
    return f"Evaluate the integral from {left} to {right}.", answer
'''
    assert check_generator_contract(source) == []


def test_the_same_computation_through_a_renamed_variable_is_still_vacuous():
    """A static gate inside a selection loop is adversarial.

    Routing the duplicate computation through one extra variable must not buy
    a pass, or the archive will simply evolve toward that shape.
    """
    dodged = VACUOUS_CHAMPION.replace(
        "    if n >= p:\n        check = 1\n    else:\n        check = 0\n",
        "    if n >= p:\n        relay = 1\n    else:\n        relay = 0\n"
        "    check = relay\n",
    )
    assert dodged != VACUOUS_CHAMPION
    assert "A3v" in _codes(dodged)


def test_a_generator_with_no_assert_at_all_is_flagged():
    source = '''
def generate(seed):
    import random
    rng = random.Random(seed)
    n = rng.randint(2, 40)
    answer = n * (n + 1) // 2
    return f"Sum the first {n} positive integers.", str(answer)
'''
    assert "A1" in _codes(source)


def test_guards_and_invariants_are_not_a_cross_check():
    """A2: asserts exist, but none compares the answer to another route."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(2, 40)
    answer = n * (n + 1) // 2
    assert n > 0, f"n={n}"
    assert answer == int(answer), f"answer={answer}"
    return f"Sum the first {n} positive integers.", str(answer)
'''
    assert "A2" in _codes(source)


def test_a_check_read_off_the_answer_is_flagged():
    """A4d, reduced from a geometry champion whose Pick's-theorem branch was
    unreachable and whose live branch aliased the check to the answer."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    area = rng.randint(10, 90)
    boundary = rng.randint(2, 8) * 2
    interior_points = int(area - boundary / 2 + 1)

    if area > 10 ** 9:
        pick_interior = int(area - boundary / 2 + 1)
    else:
        pick_interior = interior_points
    assert interior_points == pick_interior, "pick"
    return f"Lattice polygon of area {area}, boundary {boundary}.", str(interior_points)
'''
    assert "A4d" in _codes(source)


def test_a_check_against_a_constant_is_flagged():
    """A5c. The lying failure message is what lets this survive a human skim."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    side = rng.randint(2, 9)
    perimeter = 4 * side
    answer = 3
    check = perimeter
    assert answer == 3, f"answer={answer} check={check}"
    return f"A square has side {side}. State only the integer.", str(answer)
'''
    assert "A5c" in _codes(source)


def test_a_stated_parameter_that_cannot_reach_the_answer_is_flagged():
    """P1. The statement advertises a radius the answer never consults."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    radius = rng.randint(2, 9)
    n = rng.randint(3, 12)
    answer = n * (n - 3) // 2
    check = sum(1 for i in range(n) for j in range(i + 2, n) if (i, j) != (0, n - 1))
    assert answer == check, f"answer={answer} check={check}"
    problem = f"A circle of radius {radius} contains a convex {n}-gon. Count its diagonals."
    return problem, str(answer)
'''
    assert "P1" in _codes(source)


def test_a_loop_bound_is_a_real_dependency_of_a_loop_carried_answer():
    """P1 must include control dependence through the iteration count."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    string_length = rng.randint(4, 11)
    previous, current = 1, 2
    for _ in range(string_length - 1):
        previous, current = current, previous + current
    answer = current
    check = 0
    for mask in range(1 << string_length):
        if all(((mask >> position) & 3) != 3 for position in range(string_length - 1)):
            check += 1
    assert answer == check
    problem = f"Count binary strings of length {string_length} with no adjacent ones."
    return problem, str(answer)
'''
    assert "P1" not in _codes(source)


def test_control_dependencies_do_not_hide_an_unrelated_stated_parameter():
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    irrelevant_radius = rng.randint(2, 9)
    item_count = rng.randint(3, 12)
    answer = 0
    for _ in range(item_count):
        answer += 1
    check = sum(1 for _ in range(item_count))
    assert answer == check
    problem = f"A circle has radius {irrelevant_radius}. Count {item_count} marked items."
    return problem, str(answer)
'''
    assert "P1" in _codes(source)


def test_repeating_a_constant_does_not_create_loop_bound_dependence():
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    iteration_count = rng.randint(2, 9)
    answer = 0
    for _ in range(iteration_count):
        answer = 0
    check = len([])
    assert answer == check
    problem = f"A process repeats {iteration_count} times. Compute its fixed output."
    return problem, str(answer)
'''
    assert "P1" in _codes(source)


def test_identical_constant_branches_do_not_create_condition_dependence():
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    irrelevant_condition = rng.randint(2, 9)
    if irrelevant_condition > 0:
        answer = 1
    else:
        answer = 1
    check = len([0])
    assert answer == check
    problem = f"The displayed parameter is {irrelevant_condition}. Compute one."
    return problem, str(answer)
'''
    assert "P1" in _codes(source)


def test_a_declared_distractor_opts_out_of_the_statement_rule():
    """A deliberate red herring is a legitimate device; it just has to be
    declared, so the default stays "a stated parameter can matter"."""
    source = '''
import random

DISTRACTORS = ("radius",)


def generate(seed):
    rng = random.Random(seed)
    radius = rng.randint(2, 9)
    n = rng.randint(3, 12)
    answer = n * (n - 3) // 2
    check = sum(1 for i in range(n) for j in range(i + 2, n) if (i, j) != (0, n - 1))
    assert answer == check, f"answer={answer} check={check}"
    problem = f"A circle of radius {radius} contains a convex {n}-gon. Count its diagonals."
    return problem, str(answer)
'''
    assert "P1" not in _codes(source)


def test_a_statement_that_names_the_technique_does_not_force_it():
    """P2. The rank-1 champion told the solver to use the transformation."""
    handed_over = (
        "Determine the exact number of such polynomials using the "
        "transformation that considers Q(x) = P(x) - x^2. State only the integer."
    )
    assert [f.code for f in check_problem_text(handed_over)] == ["P2"]
    assert check_problem_text("Count the divisors of 5040. State only the integer.") == []


def test_an_unparseable_source_is_not_the_checkers_problem():
    """``lint_generator_source`` owns syntax errors; reporting them twice would
    double every message the model sees in a fix-retry prompt."""
    assert check_generator_contract("def generate(seed) -> :\n    ???") == []
    assert check_generator_contract("x = 1") == []


# ---------------------------------------------------------------------------
# the six shapes that each cost a false positive during derivation
# ---------------------------------------------------------------------------


def test_the_answer_variable_does_not_have_to_be_called_answer():
    """133 of 456 archived programs bind something other than ``answer``."""
    source = SOUND_CROSS_CHECK.replace("answer", "max_product")
    assert check_generator_contract(source) == []


def test_a_nested_helpers_return_is_not_the_generators_answer():
    """Reading it left the answer name empty and made every assert look like
    an invariant -- four archived programs were misjudged this way."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)

    def transform(values):
        return list(reversed(values)), values[0]

    n = rng.randint(3, 9)
    values = [rng.randint(1, 9) for _ in range(n)]
    total = sum(values)
    _, head = transform(values)
    check = sum(v for v in values)
    assert total == check, f"total={total} check={check}"
    return f"Sum the list {values}.", str(total)
'''
    assert check_generator_contract(source) == []


def test_the_dual_route_shape_binds_the_answer_after_the_assert():
    """``answer_insight``/``answer_brute`` are compared before either is bound
    to ``answer``. Keyed on the returned name alone this reads as an invariant
    between two unrelated values -- 56 of 75 A2 firings, all of them sound."""
    source = '''
import random


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 9)
    k = rng.randint(2, 3)

    answer_insight = 1
    for i in range(k):
        answer_insight = answer_insight * (n - i) // (i + 1)
    answer_brute = len([1 for mask in range(1 << n) if bin(mask).count("1") == k])
    assert answer_insight == answer_brute, f"{answer_insight} {answer_brute}"
    answer = answer_insight
    return f"Choose {k} from {n}.", str(answer)
'''
    assert check_generator_contract(source) == []


def test_an_accumulator_initialised_from_a_parameter_is_not_an_alias():
    """``val = c`` then ``val = p * val + q``: reading the first as a rename
    drags ``c`` into the answer's identity and makes every route that samples
    ``c`` look derived from the answer. It rejected two seeds."""
    assert check_generator_contract(SOUND_CROSS_CHECK) == []


def test_transitive_dependence_on_the_answer_is_not_consumption():
    """A current number-theory seed may share sampled construction inputs
    across its two routes; only a direct read of ``answer`` is consumption."""
    source = ROOT.joinpath(
        "seed_programs_domain_type", "02_number_theory_counting.py"
    ).read_text()
    assert check_generator_contract(source) == []


def test_a_tolerance_check_counts_as_an_equality():
    """``abs(check - answer) < 1e-6`` is the float form of the contract; not
    recognising it reports the real check as a comparison against a constant."""
    source = '''
import math
import random


def generate(seed):
    rng = random.Random(seed)
    b = rng.randint(2, 9)
    c = rng.randint(1, 5)
    answer = 2 * b * b + 2 * c

    roots = [(-2 * b + math.sqrt(4 * b * b - 4 * (b * b - c))) / 2,
             (-2 * b - math.sqrt(4 * b * b - 4 * (b * b - c))) / 2]
    check = sum(x * x for x in roots)
    assert abs(check - answer) < 1e-6, f"answer={answer} check={check}"
    return f"Sum of squares of roots, b={b}, c={c}.", str(answer)
'''
    assert check_generator_contract(source) == []


def test_the_assert_may_live_inside_a_sampling_loop():
    """Scanning only the function's top level finds no assert in 8 of the 8
    verified fixtures."""
    source = '''
import random

MAX_ATTEMPTS = 200


def generate(seed):
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        n = rng.randint(4, 40)
        answer = n * (n + 1) // 2
        check = sum(range(1, n + 1))
        assert answer == check, f"answer={answer} check={check}"
        return f"Sum the first {n} positive integers.", str(answer)
    raise RuntimeError("no instance")
'''
    assert check_generator_contract(source) == []


# ---------------------------------------------------------------------------
# the clean corpus -- a single firing here blocks any rule
# ---------------------------------------------------------------------------


def _fixture_programs() -> list:
    out = []
    for path in sorted(FIXTURES.glob("*.txt")):
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        for index, block in enumerate(blocks):
            out.append(pytest.param(block, id=f"{path.stem}#{index}"))
    return out


SEEDS = [
    pytest.param(Path(p).read_text(), id=Path(p).name)
    for p in sorted(glob.glob(str(ROOT / "seed_programs" / "*.py")))
]


@pytest.mark.parametrize("source", SEEDS)
def test_every_seed_satisfies_the_contract(source):
    assert check_generator_contract(source) == []


@pytest.mark.parametrize("source", _fixture_programs())
def test_every_verified_fixture_satisfies_the_contract(source):
    """The direct regression against the incident at ``evolution.py:136-149``:
    a previous strict lint rejected 20 of 20 sound programs and every one of
    its flags was hard-wired off. These eight are the pin that stops a repeat.
    """
    assert check_generator_contract(source) == []


@pytest.mark.parametrize("source", SEEDS)
def test_no_seed_statement_hands_the_solver_its_technique(source):
    from rq_evolve.program import ProblemProgram

    program = ProblemProgram(source_code=source)
    for seed in range(3):
        instance = program.execute(seed=seed)
        if instance is not None:
            assert check_problem_text(instance.problem) == []


# ---------------------------------------------------------------------------
# tombstones -- rules that were measured and rejected
# ---------------------------------------------------------------------------


def test_tombstone_operation_multiset_overlap_cannot_separate_the_corpus():
    """REJECTED: Jaccard over the two routes' operation multisets.

    A verified fixture scores 1.00 -- both routes call a helper of the same
    arity, so the multisets coincide although the algorithms do not. The
    archive median is 0.55. No threshold separates them, so this stays
    telemetry and never gates.
    """
    assert check_generator_contract(SOUND_CROSS_CHECK) == []


def test_tombstone_the_answer_need_not_depend_on_every_stated_parameter():
    """REJECTED: directed ancestry for P1.

    It fired on four of the six seeds of the day: a statement may name a
    parameter that is a SIBLING of the answer (both descend from one draw) or
    its DESCENDANT, and neither is a defect. The surviving form is the
    undirected component, so the whole seed corpus must stay clean of P1.
    """
    for path in sorted(ROOT.joinpath("seed_programs_domain_type").glob("*.py")):
        assert "P1" not in _codes(path.read_text()), path.name


def test_tombstone_a_shared_helper_is_not_by_itself_a_defect():
    """REJECTED: "both routes call the same helper, so they are not
    independent." A shared helper can be a primitive used by otherwise
    independent computations, so helper sharing alone must not gate.
    """
    source = '''
import random


def identity(value):
    return value


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(3, 12)
    answer = identity(n * (n + 1) // 2)
    check = sum(identity(k) for k in range(1, n + 1))
    assert answer == check, f"answer={answer} check={check}"
    return f"Compute 1+...+{n}.", str(answer)
'''
    assert check_generator_contract(source) == []
