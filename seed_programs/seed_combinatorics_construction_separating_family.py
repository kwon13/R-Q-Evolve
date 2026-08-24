import random


def _separates(n, family):
    for x in range(n):
        for y in range(x + 1, n):
            if not any((x in subset) != (y in subset) for subset in family):
                return False
    return True


def _bit_family(n, r):
    # For each of the r bit positions, the subset of elements whose index
    # carries a 1 there.
    return [
        {
            x
            for x in range(n)
            if ((x >> bit) & 1) == 1
        }
        for bit in range(r)
    ]


def generate(seed):
    rng = random.Random(seed)

    # Draw the size of the answer first, then a ground set inside that band.
    # Sampling n uniformly instead would pile almost every instance onto the
    # one or two values of ceil(log_2 n) that a wide interval of n maps to.
    width = rng.randint(3, 10)
    n = rng.randint((1 << (width - 1)) + 1, 1 << width)

    # Intended route: two counting bounds that meet.
    #
    # Lower bound:
    # With r subsets, each element of U carries an r-bit membership pattern.
    # Two elements that some set separates have different patterns, so all n
    # patterns are distinct and n <= 2^r.
    #
    # Construction:
    # Give the n elements distinct r-bit strings. For each bit position j,
    # take the subset of elements whose j-th bit is 1. Two distinct elements
    # differ in some bit, and that bit's subset separates them.
    #
    # Therefore r = ceil(log_2 n).
    answer = (n - 1).bit_length()

    # Independent route: assume nothing about where the threshold sits. Walk r
    # upward from zero and put each family through the pairwise separation
    # test the problem states, taking the first r that passes.
    #
    # The scan does not skip ahead to the counting bound. Below the threshold
    # the test genuinely fails -- two elements whose indices agree in the low
    # r bits collide -- so this route locates the threshold by testing pairs
    # rather than by reusing the inequality n <= 2^r.
    check = None

    for r in range(n + 1):
        if _separates(n, _bit_family(n, r)):
            check = r
            break

    assert check is not None
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let U = {{1, 2, ..., {n}}}. A family F of subsets of U is called "
        f"separating if for every two distinct elements x and y of U, there "
        f"is a set A in F that contains exactly one of x and y. Find the "
        f"minimum possible value of |F|."
    )

    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "construction"
