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

    problem = (
        f"{n} people each write their name on a card. The cards are shuffled "
        f"and redistributed. In how many redistributions do at least {k} people "
        f"receive their own card?"
    )

    return problem, str(answer)

CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.derangement_fixed_points"
