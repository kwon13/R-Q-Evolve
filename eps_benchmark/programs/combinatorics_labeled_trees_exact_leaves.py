"""Held-out combinatorics family: labeled trees with an exact leaf count.

The intended count uses the missing symbols in a Pruefer word together with
inclusion-exclusion.  A distinct-symbol dynamic program checks the result.
"""

import math
import random


def _inclusion_exclusion_count(n, leaves):
    used = n - leaves
    length = n - 2
    onto = sum(
        (-1) ** omitted
        * math.comb(used, omitted)
        * (used - omitted) ** length
        for omitted in range(used + 1)
    )
    return math.comb(n, leaves) * onto


def _distinct_symbol_dp(n, leaves):
    length = n - 2
    target_used = n - leaves
    counts = [0] * (n + 1)
    counts[0] = 1

    for _ in range(length):
        updated = [0] * (n + 1)
        for used in range(n + 1):
            updated[used] += counts[used] * used
            if used < n:
                updated[used + 1] += counts[used] * (n - used)
        counts = updated
    return counts[target_used]


def generate(seed):
    rng = random.Random(seed)

    options = []
    for n in range(7, 23):
        for leaves in range(2, n - 1):
            value = _inclusion_exclusion_count(n, leaves)
            if 1_000 <= value < 1_000_000_000:
                options.append((n, leaves, value))

    n, leaves, answer = rng.choice(options)
    check = _distinct_symbol_dp(n, leaves)
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"How many trees on the labeled vertex set {{1, 2, ..., {n}}} have "
        f"exactly {leaves} vertices of degree 1?"
    )
    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "counting"
