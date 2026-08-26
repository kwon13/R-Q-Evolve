import random


def generate(seed):
    rng = random.Random(seed)

    n = rng.randint(6, 15)

    # D_m = number of derangements of m objects.
    #
    # Look at where person 1's card goes, say to person j.
    #
    # Case 1: person j's card goes back to person 1.
    # Removing these two people leaves a derangement of m-2 people.
    #
    # Case 2: person j's card does not go to person 1.
    # Merge person 1 into the position involving j; this reduces to
    # a derangement of m-1 people.
    #
    # There are m-1 choices of j, hence
    #
    #   D_m = (m-1)(D_{m-1} + D_{m-2}).
    #
    # The size-m problem therefore essentially depends on smaller instances.

    D = [0] * (n + 1)
    D[0] = 1
    D[1] = 0

    for m in range(2, n + 1):
        D[m] = (m - 1) * (D[m - 1] + D[m - 2])

    answer = D[n]

    # Independent check via inclusion-exclusion.
    import math
    check = round(
        math.factorial(n)
        * sum(((-1) ** j) / math.factorial(j) for j in range(n + 1))
    )

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"{n} people each write their name on a card. The cards are shuffled "
        f"and redistributed, one card to each person. In how many redistributions "
        f"does no person receive their own card?"
    )

    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "induction"