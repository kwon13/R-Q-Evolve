"""Held-out number-theory family: a prime valuation of a binomial number.

The primary calculation subtracts factorial valuations.  Kummer's carry
count in base p supplies an independent generator-side route.
"""

import random


PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)


def _factorial_valuation(n, prime):
    value = 0
    power = prime
    while power <= n:
        value += n // power
        power *= prime
    return value


def _carry_count(left, right, base):
    carries = 0
    carry = 0
    while left or right or carry:
        column = left % base + right % base + carry
        if column >= base:
            carries += 1
            carry = 1
        else:
            carry = 0
        left //= base
        right //= base
    return carries


def _balanced_options():
    buckets = {
        target: {prime: [] for prime in PRIMES}
        for target in range(3, 16)
    }
    for prime in PRIMES:
        power = prime
        exponent = 1
        while power <= 1_000_000_000_000:
            if exponent >= 5:
                for target in buckets:
                    extra_power = exponent - target
                    if extra_power < 2 or prime == target:
                        continue
                    for multiplier in range(1, 80):
                        unit = 1 + multiplier * prime
                        if unit >= prime ** target:
                            break
                        k = prime ** extra_power * unit
                        if target not in {prime, power, k}:
                            buckets[target][prime].append(
                                (prime, power, k, target)
                            )
            exponent += 1
            power *= prime

    options = []
    for target in sorted(buckets):
        by_prime = {
            prime: values
            for prime, values in buckets[target].items()
            if values
        }
        assert len(by_prime) >= 3
        chosen = []
        offset = 0
        while len(chosen) < 40:
            for prime in sorted(by_prime):
                values = by_prime[prime]
                if offset < len(values):
                    chosen.append(values[offset])
                    if len(chosen) == 40:
                        break
            offset += 1
        assert len({item[0] for item in chosen}) >= 3
        options.extend(chosen)
    return options


def generate(seed):
    rng = random.Random(seed)

    # Every answer stratum contributes the same number of configurations, so
    # random selection does not collapse onto the most common carry count.
    prime, n, k, target = rng.choice(_balanced_options())
    answer = (
        _factorial_valuation(n, prime)
        - _factorial_valuation(k, prime)
        - _factorial_valuation(n - k, prime)
    )

    check = _carry_count(k, n - k, prime)
    assert answer == target
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"What is the largest integer e such that {prime}^e divides the "
        f"binomial coefficient C({n}, {k})?"
    )
    return problem, str(answer)


GROUP = "number_theory"
SKILL = "transformation"
