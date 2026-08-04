import random
from fractions import Fraction

def generate(seed):
    rng = random.Random(seed)

    external_1 = rng.randint(3, 14)
    internal_1 = rng.randint(8, 30)
    whole_1 = external_1 + internal_1
    product = external_1 * whole_1
    # Require PC != PA (avoid trivial symmetry where CD == AB is read off),
    # PC < PD so CD > 0, and PC divides product so PD is an integer >= 2.
    valid = [
        d for d in range(2, whole_1)
        if product % d == 0 and product // d > d and d != external_1
    ]
    # Fallback: the divisor d = external_1 is always in range and satisfies
    # product // d = whole_1 > external_1, so CD = internal_1 > 0.
    if not valid:
        valid = [external_1]
    external_2 = rng.choice(valid)
    whole_2 = Fraction(product, external_2)
    internal_2 = whole_2 - external_2
    answer = (
        str(internal_2.numerator)
        if internal_2.denominator == 1
        else f"{internal_2.numerator}/{internal_2.denominator}"
    )
    problem = (
        f"From an external point P, two secants intersect a circle at A,B "
        f"and C,D respectively, with A and C nearer to P. Given PA={external_1}, "
        f"AB={internal_1}, and PC={external_2}, find CD."
    )

    return problem, str(answer)

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.power_of_point_secants"
