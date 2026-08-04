import math
import random

K_MAX = 72  # cap on the shown exponent k = t * u


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
    factors = sorted(_prime_factors(p - 1))  # sorted(): determinism rule
    for g in range(2, p):
        if all(pow(g, (p - 1) // q, p) != 1 for q in factors):
            return g
    return 2


def generate(seed):
    rng = random.Random(seed)

    # INVARIANT chain, carried by construction instead of curated lists:
    #   t (= the answer) cycles over [2, 24] so answers stay evenly covered;
    #   primes are DERIVED by a trial-division filter, never typed in;
    #   p is drawn from the derived primes with t | p - 1;
    #   k = t * u with gcd(u, (p-1)/t) = 1  ->  gcd(k, p-1) = t * gcd(u, v) = t;
    #   e = t * j                           ->  t | ind(a), so x^k = a is solvable.
    t = 2 + (seed % 23)

    primes = [n for n in range(29, 294)  # widen these bounds freely
              if all(n % d for d in range(2, math.isqrt(n) + 1))]
    eligible = [q for q in primes if (q - 1) % t == 0]
    assert eligible, f"no prime p with {t} | p-1 in range: widen the prime range"
    p = rng.choice(eligible)

    v = (p - 1) // t
    u_options = [u for u in (1, 2, 3)
                 if t * u <= K_MAX and math.gcd(u, v) == 1]
    u = rng.choice(u_options)              # u = 1 always valid: never empty
    k = t * u
    e = t * rng.randint(1, (p - 2) // t)   # multiple of t in [t, p-2]

    g = _primitive_root(p)
    a = pow(g, e, p)
    answer = t

    assert math.gcd(k, p - 1) == t and e % t == 0, (
        "invariant violated: gcd(k, p-1) == t and t | e must hold"
    )

    # Independent route: count the solutions of x^k = a by testing every
    # residue, instead of reading the count off gcd(k, p-1).
    check = sum(1 for x in range(p) if pow(x, k, p) == a)
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let p = {p}, and let g = {g} be a primitive root modulo p. "
        f"For the residue a = {a}, how many residue classes x modulo {p} "
        f"satisfy x^{k} ≡ a (mod {p})?"
    )
    return problem, str(answer)

GROUP = "number_theory"
SKILL = "transformation"