"""Held-out sequence family: a nonlinear recurrence conjugate to squaring.

Writing a_n as z^(2^n) + z^(-2^n) gives the intended fast route.  Literal
recurrence iteration is an independent check at the modest hidden index.
"""

import random


PRIMES = (101, 103, 107, 109, 127, 131, 137, 139, 149, 151, 157, 163)


def generate(seed):
    rng = random.Random(seed)

    while True:
        modulus = rng.choice(PRIMES)
        z = rng.randint(3, modulus - 3)
        index = rng.randint(35, 120)
        inverse = pow(z, -1, modulus)
        exponent = pow(2, index, modulus - 1)
        answer = (
            pow(z, exponent, modulus)
            + pow(inverse, exponent, modulus)
        ) % modulus
        if answer > 2 and answer not in {z, index, modulus}:
            break

    initial = (z + inverse) % modulus
    check = initial
    for _ in range(index):
        check = (check * check - 2) % modulus

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Work modulo the prime {modulus}. Let z = {z}, let a_0 be the "
        f"least nonnegative residue of z + z^(-1), and define "
        f"a_(n+1) to be the least nonnegative residue of a_n^2 - 2. "
        f"Find a_{index}."
    )
    return problem, str(answer)


GROUP = "sequence"
SKILL = "transformation"
