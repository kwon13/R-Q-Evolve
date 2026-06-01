import math
import random


PRIMES = [29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89,
          97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151,
          157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223,
          227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281,
          283, 293]
K_POOL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15, 16, 18, 20, 24]
DIVISOR_TARGETS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 15, 18, 20, 24]


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
    factors = _prime_factors(p - 1)
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2


def generate(seed):
    rng = random.Random(seed)

    target = DIVISOR_TARGETS[seed % len(DIVISOR_TARGETS)]
    candidates = [(p, k) for p in PRIMES for k in K_POOL
                  if math.gcd(k, p - 1) == target]
    p, k = rng.choice(candidates)
    g = _primitive_root(p)
    multiples = list(range(target, p - 1, target))
    exponent = rng.choice(multiples)
    a = pow(g, exponent, p)
    answer = target

    problem = (
        f"Let p = {p}, and let g = {g} be a primitive root modulo p. "
        f"For the residue a = {a}, how many residue classes x modulo {p} "
        f"satisfy x^{k} ≡ a (mod {p})?"
    )
    return problem, str(answer)

CONCEPT_GROUP = "number_theory"
CONCEPT_TYPE = "number_theory.kth_root_mod_prime"
