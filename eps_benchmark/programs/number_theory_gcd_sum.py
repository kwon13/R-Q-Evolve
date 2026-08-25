"""Held-out transformation family: a gcd sum over a complete residue range.

The primary route groups terms by divisors and evaluates Euler totients.  The
check visits every summand directly with Euclid's gcd implementation.
"""

import math
import random


def _divisors(number):
    small = []
    large = []
    divisor = 1

    while divisor * divisor <= number:
        if number % divisor == 0:
            small.append(divisor)
            if divisor * divisor != number:
                large.append(number // divisor)
        divisor += 1

    return small + list(reversed(large))


def _totient(number):
    result = number
    remaining = number
    prime = 2

    while prime * prime <= remaining:
        if remaining % prime == 0:
            result -= result // prime
            while remaining % prime == 0:
                remaining //= prime
        prime += 1

    if remaining > 1:
        result -= result // remaining

    return result


def _divisor_totient_sum(number):
    return sum(
        divisor * _totient(number // divisor)
        for divisor in _divisors(number)
    )


def _direct_gcd_sum(number):
    return sum(math.gcd(value, number) for value in range(1, number + 1))


def generate(seed):
    rng = random.Random(seed)

    # Several proper divisors make the grouping nontrivial.  The range is
    # still small enough for a genuinely direct verification pass.
    options = [
        number
        for number in range(180, 2401)
        if len(_divisors(number)) >= 8
    ]
    number = rng.choice(options)

    answer = _divisor_totient_sum(number)

    # Independent route: compute every gcd in the sum as stated, without
    # factoring the index set into divisor classes.
    check = _direct_gcd_sum(number)

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"For n = {number}, compute the sum of gcd(k, n) over all integers k "
        f"with 1 <= k <= n."
    )
    return problem, str(answer)


GROUP = "number_theory"
SKILL = "transformation"
