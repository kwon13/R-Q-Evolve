import random
from itertools import product


def generate(seed):
    rng = random.Random(seed)

    # n is the only parameter. Below 6 the strings can be listed by hand and
    # the recursion is never forced. The upper end is set by the exhaustive
    # check, which walks all 2^n strings.
    n = rng.randint(6, 16)

    # Let a_m be the number of binary strings of length m with no three
    # consecutive 1s.
    #
    # Split by the run of 1s at the right end. Such a string ends in
    #
    #     ...0     -> the preceding m-1 characters are admissible
    #     ...01    -> the preceding m-2 characters are admissible
    #     ...011   -> the preceding m-3 characters are admissible
    #
    # and nothing else survives, since a longer run ends in 111. The three
    # cases are disjoint and exhaustive, so
    #
    #     a_m = a_{m-1} + a_{m-2} + a_{m-3}.
    #
    # Base cases: a_0 = 1 (the empty string), a_1 = 2, a_2 = 4. There is no
    # short closed form here, so the recursion is not just the fastest route
    # in -- it is the only one.
    counts = [1, 2, 4]

    for m in range(3, n + 1):
        counts.append(counts[-1] + counts[-2] + counts[-3])

    answer = counts[n]

    # Independent route: enumerate every string of length n and reject the
    # ones containing 111. This shares nothing with the case split above.
    check = 0

    for bits in product("01", repeat=n):
        if "111" not in "".join(bits):
            check += 1

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"A string of length {n} is formed using only the characters 0 and 1. "
        f"How many such strings contain no three consecutive 1s?"
    )

    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "induction"