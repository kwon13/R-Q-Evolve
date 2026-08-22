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


# How far the witness search on the check route is allowed to look for a
# common difference that keeps one term prime. Every index below the answer
# is escapable, and in practice by a very small d; the cap only stops the
# search running forever at the index where no d works at all.
WITNESS_SEARCH = 400


def generate(seed):
    rng = random.Random(seed)

    # Small primes: the contradiction (a_{p+1} = p(d+1)) is identical at any
    # p, and a large p only lengthens the ruling-out of smaller indices. The
    # answer is p + 1, so p is the only thing that varies; the upper end is
    # what keeps a one-parameter family from collapsing onto a handful of
    # distinct statements while the arithmetic stays small.
    primes = [
        p for p in range(5, 40)
        if _is_prime(p)
    ]
    p = rng.choice(primes)

    # Intended route: a contradiction covering an unbounded family of d.
    #
    # Fix any d >= 1. The sequence is a_N = p + (N - 1) * d, so
    #
    #   a_{p+1} = p + p * d = p * (d + 1),
    #
    # a proper multiple of p and therefore composite -- for every d at once.
    # No finite search over d could ever establish that.
    #
    # No smaller index survives. For 2 <= N <= p the gap N - 1 lies in
    # [1, p - 1], so gcd(p, N - 1) = 1 and Dirichlet puts infinitely many
    # primes in {p + (N - 1) * d : d >= 1}; and a_1 = p is prime outright.
    # Every index up to p thus has some d that keeps its term prime, so p + 1
    # is the first index no choice of d can escape.
    answer = p + 1

    # Independent route: factor nothing. Walk the index upward and try to
    # exhibit a d that keeps that one term prime, stopping at the first index
    # for which the search turns up no witness at all.
    check = None

    for index in range(1, p + 2):
        witness = None

        for d in range(1, WITNESS_SEARCH + 1):
            if _is_prime(p + (index - 1) * d):
                witness = d
                break

        if witness is None:
            check = index
            break

    assert check is not None
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let p = {p}. For each positive integer d, define an infinite "
        f"sequence by a_1 = p and a_(n+1) = a_n + d for every n >= 1. Find "
        f"the smallest index N such that a_N is composite for every choice "
        f"of d. State only the integer."
    )

    return problem, str(answer)


GROUP = "sequence"
SKILL = "contradiction"
