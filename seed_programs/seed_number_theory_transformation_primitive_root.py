import math
import random

# Cap on the exponent printed in the statement.
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

    return sorted(factors)


def _is_prime(n):
    if n < 2:
        return False

    for d in range(2, math.isqrt(n) + 1):
        if n % d == 0:
            return False

    return True


def _primitive_root(p):
    factors = _prime_factors(p - 1)

    for g in range(2, p):
        if all(
            pow(g, (p - 1) // q, p) != 1
            for q in factors
        ):
            return g

    raise ValueError("primitive root not found")


def generate(seed):
    rng = random.Random(seed)

    # Draw the answer first, then a prime that can carry it.
    #
    # Picking p first and a divisor of p-1 second collapses the answers onto
    # one value: p-1 is even for every odd prime, so 2 sits in the divisor list
    # every single time while the larger divisors are only sometimes there.
    # Cycling d over a fixed range and keeping only the primes with d | p-1
    # covers the answers evenly instead.
    d = 2 + (seed % 23)

    primes = [
        q
        for q in range(47, 600)
        if _is_prime(q) and (q - 1) % d == 0
    ]
    assert primes, f"no prime p in range with {d} | p-1: widen the prime range"
    p = rng.choice(primes)

    g = _primitive_root(p)

    # Choose the cofactor u coprime to (p-1)/d, so that
    # gcd(k, p-1) = d * gcd(u, (p-1)/d) = d.
    #
    # u > 1 is preferred: at u = 1 the exponent k printed in the statement IS
    # the answer, and the problem can be answered by copying it instead of
    # taking a gcd.
    v = (p - 1) // d
    u_options = [
        u
        for u in (2, 3, 5, 7, 11, 13)
        if math.gcd(u, v) == 1 and d * u <= K_MAX and d * u < p - 1
    ]
    if not u_options:
        u_options = [1]
    u = rng.choice(u_options)
    k = d * u

    # Make the transformed linear congruence solvable:
    #
    #     x = g^r
    #     x^k = g^e
    #
    # becomes
    #
    #     k r ≡ e (mod p-1).
    #
    # Since gcd(k, p-1) = d and d | e, this congruence
    # has exactly d solutions modulo p-1.
    e = d * rng.randint(1, (p - 2) // d)
    a = pow(g, e, p)

    assert math.gcd(k, p - 1) == d
    assert e % d == 0

    answer = d

    # Independent route: test every residue class directly in the original
    # multiplicative representation, without using exponent coordinates.
    check = sum(
        1
        for x in range(p)
        if pow(x, k, p) == a
    )

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let p = {p}, and let g = {g} be a primitive root modulo p. "
        f"Let a be the residue congruent to g^{e} modulo p. "
        f"How many residue classes x modulo p satisfy "
        f"x^{k} congruent to a modulo p?"
    )

    return problem, str(answer)


GROUP = "number_theory"
SKILL = "transformation"
