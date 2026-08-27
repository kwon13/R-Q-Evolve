import itertools
import random

DOMAIN = "discrete_mathematics"


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(4, 6)

    answer = (n * n) // 4

    edges = list(itertools.combinations(range(n), 2))
    check = 0
    for mask in range(1 << len(edges)):
        chosen = {
            edge for index, edge in enumerate(edges) if (mask >> index) & 1
        }
        triangle_free = all(
            not ({(i, j), (i, k), (j, k)} <= chosen)
            for i, j, k in itertools.combinations(range(n), 3)
        )
        if triangle_free:
            check = max(check, len(chosen))
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"What is the maximum possible number of edges in a simple graph on {n} "
        "vertices that contains no triangle?"
    )
    return problem, str(answer), {"mode": "expression"}
