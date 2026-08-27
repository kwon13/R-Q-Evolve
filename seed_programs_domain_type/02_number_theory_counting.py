import math
import random

DOMAIN = "number_theory"


def generate(seed):
    rng = random.Random(seed)
    n = rng.randint(18, 90)

    remaining = n
    answer = n
    prime = 2
    while prime * prime <= remaining:
        if remaining % prime == 0:
            answer -= answer // prime
            while remaining % prime == 0:
                remaining //= prime
        prime += 1
    if remaining > 1:
        answer -= answer // remaining

    check = sum(math.gcd(k, n) == 1 for k in range(1, n + 1))
    assert answer == check, f"answer={answer} check={check}"

    problem = f"How many integers k with 1 <= k <= {n} are relatively prime to {n}?"
    return problem, str(answer), {"mode": "expression"}
