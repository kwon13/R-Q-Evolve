import random

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
    problem = (
        f"A sequence is defined by a(0) = {c} and "
        f"a(n) = {p} * a(n-1) + {q} for n >= 1. Find a({n})."
    )

    return problem, str(answer)

CONCEPT_GROUP = "sequence"
CONCEPT_TYPE = "sequence.linear_recurrence"
