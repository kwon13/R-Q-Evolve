import random
from fractions import Fraction

def generate(seed):
    rng = random.Random(seed)

    r1 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)
    r2 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)
    while r2 == r1 or r1 * r2 == 0:
        r2 = rng.randint(-8, -1) if rng.random() < 0.5 else rng.randint(1, 8)

    s = r1 + r2
    p = r1 * r2

    b = -s
    c = p

    # New transformed roots:
    # u = 2 + 1/r1, v = 2 + 1/r2
    # The coefficient of x in the monic quadratic with roots u, v is -(u + v).
    u_plus_v = Fraction(4, 1) + Fraction(s, p)
    answer = -u_plus_v

    problem = (
        f"The quadratic equation x^2 + {b}x + {c} = 0 has two real roots r1 and r2. "
        f"Let u = 2 + 1/r1 and v = 2 + 1/r2. "
        f"Consider the monic quadratic equation whose roots are u and v. "
        f"Find the coefficient of x in that new quadratic equation, as a fraction."
    )

    return problem, str(answer)

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.quadratic_vieta_reciprocal"
