import math
import random


def _derangement(n):
    if n == 0:
        return 1
    if n == 1:
        return 0
    a, b = 1, 0
    for k in range(2, n + 1):
        a, b = b, (k - 1) * (a + b)
    return b


def generate(seed):
    rng = random.Random(seed)

    n = rng.randint(6, 15)
    k = rng.randint(1, min(5, n - 2))

    answer = sum(math.comb(n, j) * _derangement(n - j) for j in range(k, n + 1))

    # Two independent routes. First: every redistribution has some number of
    # fixed points, so the exact-j counts must sum to n! -- this is what pins
    # _derangement itself. Second: reach the same total from the complement,
    # counting the redistributions with fewer than k own cards and subtracting.
    assert sum(math.comb(n, j) * _derangement(n - j) for j in range(n + 1)) == math.factorial(n)
    check = math.factorial(n) - sum(
        math.comb(n, j) * _derangement(n - j) for j in range(k)
    )
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"{n} people each write their name on a card. The cards are shuffled "
        f"and redistributed. In how many redistributions do at least {k} people "
        f"receive their own card?"
    )

    return problem, str(answer)

GROUP = "combinatorics"
SKILL = "counting"
