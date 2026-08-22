import random


def _is_prime(n):
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _smallest_quadratic_nonresidue(p):
    for g in range(2, p):
        if pow(g, (p - 1) // 2, p) == p - 1:
            return g
    raise ValueError("no quadratic nonresidue found")


def _crt_mod_3_p(r3, rp, p):
    # Solve x ≡ r3 (mod 3), x ≡ rp (mod p).
    # Since p is an odd prime different from 3, 3 is invertible modulo p.
    t = ((rp - r3) * pow(3, -1, p)) % p
    x = r3 + 3 * t
    return 3 * p if x == 0 else x


def generate(seed):
    rng = random.Random(seed)

    primes = [p for p in range(17, 60) if _is_prime(p)]
    p = rng.choice(primes)
    a = rng.randint(2, p - 2)
    g = _smallest_quadratic_nonresidue(p)

    # Case 1: x ≡ 0 (mod 3).
    # Then the additional condition is linear: x ≡ a (mod p),
    # giving exactly one solution modulo 3p.
    x0 = _crt_mod_3_p(0, a, p)

    # Case 2: x ≡ 1 (mod 3).
    # x^2 ≡ 4 (mod p) splits into x ≡ 2 or -2 (mod p),
    # so two separate CRT solutions occur.
    x1 = _crt_mod_3_p(1, 2, p)
    x2 = _crt_mod_3_p(1, p - 2, p)

    # Case 3: x ≡ 2 (mod 3).
    # Since g is a quadratic nonresidue modulo p,
    # x^2 ≡ g (mod p) has no solutions.
    assert pow(g, (p - 1) // 2, p) == p - 1

    answer = x0 + x1 + x2

    # Independent route: inspect every integer in the stated range and apply
    # exactly the condition belonging to its residue class modulo 3.
    check = 0
    for x in range(1, 3 * p + 1):
        if x % 3 == 0:
            valid = x % p == a
        elif x % 3 == 1:
            valid = pow(x, 2, p) == 4
        else:
            valid = pow(x, 2, p) == g

        if valid:
            check += x

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let p = {p}, let a = {a}, and let g = {g}, where g is a quadratic "
        f"nonresidue modulo p. For integers x with 1 <= x <= {3 * p}, call x "
        f"admissible as follows: if x is congruent to 0 modulo 3, require "
        f"x congruent to a modulo p; if x is congruent to 1 modulo 3, require "
        f"x^2 congruent to 4 modulo p; and if x is congruent to 2 modulo 3, "
        f"require x^2 congruent to g modulo p. Find the sum of all admissible "
        f"integers x. State only the integer."
    )

    return problem, str(answer)


GROUP = "number_theory"
SKILL = "casework"