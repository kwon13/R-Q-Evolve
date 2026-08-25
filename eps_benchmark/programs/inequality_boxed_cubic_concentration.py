"""Held-out inequality family: maximizing a convex cubic on a box slice.

The intended extremal argument concentrates mass at upper endpoints.  An
exhaustive enumeration of the continuous box-slice vertices checks the value.
"""

import itertools
import random


def _polytope_vertex_check(n, upper, total):
    best = None
    # At a vertex of one equality intersected with a box, all but at most one
    # coordinate are endpoints.  Enumerate the free coordinate and every
    # endpoint assignment without using the quotient/remainder formula.
    for free_index in range(n):
        for endpoints in itertools.product((0, upper), repeat=n - 1):
            residual = total - sum(endpoints)
            if not 0 <= residual <= upper:
                continue
            values = list(endpoints)
            values.insert(free_index, residual)
            candidate = sum(value ** 3 for value in values)
            if best is None or candidate > best:
                best = candidate
    return best


def generate(seed):
    rng = random.Random(seed)

    options = []
    for upper in range(5, 19):
        for quotient in range(2, 8):
            # Canonicalize n for each (upper, total) configuration while
            # retaining at least one forced zero in the sharp construction.
            n = max(5, quotient + 2)
            for remainder in range(1, upper):
                total = quotient * upper + remainder
                value = quotient * upper ** 3 + remainder ** 3
                if 500 <= value < 1_000_000:
                    options.append((n, upper, total, value))

    n, upper, total, answer = rng.choice(options)
    check = _polytope_vertex_check(n, upper, total)
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Real numbers x_1, ..., x_{n} satisfy 0 <= x_i <= {upper} for "
        f"every i and x_1 + ... + x_{n} = {total}. What is the maximum "
        f"possible value of x_1^3 + ... + x_{n}^3?"
    )
    return problem, str(answer)


GROUP = "inequality"
SKILL = "extremal_principle"
