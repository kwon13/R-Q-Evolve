import math
import random

K_MAX = 72


def _prime_factors(n):
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _primitive_root(p):
    factors = sorted(_prime_factors(p - 1))
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2


def generate(seed):
    rng = random.Random(seed)

    # Build a prime and exponent k.
    primes = [
        n for n in range(29, 294)
        if all(n % d for d in range(2, math.isqrt(n) + 1))
    ]
    p = rng.choice(primes)

    g = _primitive_root(p)

    k = rng.randint(2, min(K_MAX, p - 2))
    d = math.gcd(k, p - 1)

    # Create two residues requiring genuinely different treatment:
    #
    # Case 1: exponent e1 is divisible by d -> x^k = a1 has exactly d solutions.
    # Case 2: exponent e2 is not divisible by d -> x^k = a2 has no solutions.
    #
    # The solver must identify which regime each residue belongs to.

    multiples = [
        e for e in range(1, p - 1)
        if e % d == 0
    ]
    nonmultiples = [
        e for e in range(1, p - 1)
        if e % d != 0
    ]

    # Avoid degenerate d = 1, where there is no second case.
    if d == 1:
        k = (p - 1) // 2
        d = math.gcd(k, p - 1)

        multiples = [
            e for e in range(1, p - 1)
            if e % d == 0
        ]
        nonmultiples = [
            e for e in range(1, p - 1)
            if e % d != 0
        ]

    e1 = rng.choice(multiples)
    e2 = rng.choice(nonmultiples)

    a1 = pow(g, e1, p)
    a2 = pow(g, e2, p)

    # Asked for the total number of solutions across the two congruences.
    answer = d

    # Independent brute-force route.
    check1 = sum(
        1 for x in range(p)
        if pow(x, k, p) == a1
    )
    check2 = sum(
        1 for x in range(p)
        if pow(x, k, p) == a2
    )

    assert check1 == d
    assert check2 == 0
    assert answer == check1 + check2

    problem = (
        f"Let p = {p}, and let g = {g} be a primitive root modulo p. "
        f"Let a = {a1} and b = {a2}. "
        f"Determine the total number of residue classes x modulo {p} "
        f"that satisfy either x^{k} ≡ a (mod {p}) or "
        f"x^{k} ≡ b (mod {p})."
    )

    return problem, str(answer)


GROUP = "number_theory"
SKILL = "casework"