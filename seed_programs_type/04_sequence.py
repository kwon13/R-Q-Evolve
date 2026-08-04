import random

def generate(seed):
    rng = random.Random(seed)

    a = rng.randint(2, 5)
    r = rng.choice([2, 3])
    n = rng.randint(4, 8)
    geo_sum = a * (r ** n - 1) // (r - 1)
    d = rng.randint(5, 20)
    m = rng.randint(5, 10)
    arith_sum = m * (2 * geo_sum + (m - 1) * d) // 2
    answer = arith_sum
    problem = (
        f"A geometric sequence has first term {a} and common ratio {r}. "
        f"Let S be the sum of the first {n} terms. "
        f"An arithmetic sequence has first term S and common difference {d}. "
        f"What is the sum of the first {m} terms of this arithmetic sequence?"
    )

    return problem, str(answer)

CONCEPT_GROUP = "sequence"
CONCEPT_TYPE = "sequence.geometric_to_arithmetic_sum"
