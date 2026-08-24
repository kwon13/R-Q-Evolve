import random


def generate(seed):
    rng = random.Random(seed)

    # n is the only parameter, so its range is also the instance space. The
    # upper end is kept where 2^(n-3) is still read straight off the closed
    # form; much larger and the reduction mod 1000 becomes a cycle-finding
    # puzzle of its own, stacked on top of the induction.
    n = rng.randint(10, 30)

    # Let F_1 = {(1)}. For m >= 2, every permutation in F_{m-1}
    # produces two members of F_m by inserting m at the left end
    # or at the right end.
    #
    # If T_m is the total inversion count over all permutations in F_m,
    # then |F_{m-1}| = 2^(m-2), and:
    #
    #   T_m = 2 T_{m-1} + (m-1) 2^(m-2).
    #
    # The first term duplicates all old inversions into both children.
    # A left insertion of m creates exactly m-1 new inversions,
    # while a right insertion creates none.
    #
    # Inductively this gives:
    #
    #   T_m = m(m-1) 2^(m-3).
    #
    # T_n is asked modulo 1000 so the answer stays a short integer. The
    # recursion is still the only way in -- the residue is read off the
    # closed form.
    answer = n * (n - 1) * (2 ** (n - 3)) % 1000

    # Independent route: compute T_n directly from the recursive construction,
    # without using the closed form.
    family_size = 1
    check = 0

    for m in range(2, n + 1):
        check = 2 * check + (m - 1) * family_size
        family_size *= 2

    check %= 1000

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let F_1 consist only of the permutation (1). For each integer m from "
        f"2 through {n}, construct F_m from F_(m-1) as follows: for every "
        f"permutation in F_(m-1), create two new permutations by inserting m "
        f"either at the left end or at the right end. Let T be the sum of the "
        f"numbers of inversions over all permutations in F_{n}. Find the "
        f"remainder when T is divided by 1000."
    )

    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "induction"