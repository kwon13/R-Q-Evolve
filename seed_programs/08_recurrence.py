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

    # Independent route: the closed form of a(n) = p*a(n-1) + q, instead of
    # iterating the recurrence n times.
    check = p ** n * c + q * (p ** n - 1) // (p - 1)
    assert answer == check, f"answer={answer} check={check}"
    problem = (
        f"A sequence is defined by a(0) = {c} and "
        f"a(n) = {p} * a(n-1) + {q} for n >= 1. Find a({n})."
    )

    return problem, str(answer)

GROUP = "sequence"
SKILL = "induction"
