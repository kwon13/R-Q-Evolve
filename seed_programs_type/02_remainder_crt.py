import math
import random


M1_POOL = [3, 5, 7, 11, 13, 17, 19]
M2_POOL = [4, 9, 11, 13, 17, 19, 23, 25, 29]


def generate(seed):
    rng = random.Random(seed)

    m1 = rng.choice(M1_POOL)
    m2 = rng.choice([m for m in M2_POOL if m != m1 and math.gcd(m, m1) == 1])
    r1 = rng.randint(1, m1 - 1)
    r2 = rng.randint(1, m2 - 1)
    product = m1 * m2
    base = next(n for n in range(1, product + 1) if n % m1 == r1 and n % m2 == r2)
    upper = rng.randint(500, 50000)
    answer = (upper - base) // product + 1

    problem = (
        f"A positive integer N leaves a remainder of {r1} when divided by {m1}, "
        f"and a remainder of {r2} when divided by {m2}. "
        f"How many such integers N exist in the range 1 to {upper}, inclusive?"
    )

    return problem, str(answer)

CONCEPT_GROUP = "number_theory"
CONCEPT_TYPE = "number_theory.crt_count"
