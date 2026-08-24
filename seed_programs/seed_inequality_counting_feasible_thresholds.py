import random


def generate(seed):
    rng = random.Random(seed)

    # n is the only parameter, so its range is also the whole instance space:
    # 31 distinct statements. The reasoning is AM-GM then a parity split,
    # which a wider range would not change -- it would only make the count
    # longer to write down.
    n = rng.randint(10, 40)

    # For a fixed integer s >= 2, consider positive real x, y with x + y = s.
    # By AM-GM,
    #
    #     xy <= (s/2)^2,
    #
    # with equality at x = y = s/2.
    #
    # Therefore an integer t >= 1 is feasible for this s iff
    #
    #     t <= floor(s^2 / 4).
    #
    # We must count
    #
    #     sum_{s=2}^n floor(s^2 / 4).
    #
    # Split s by parity:
    #
    #   s = 2r     -> floor(s^2/4) = r^2
    #   s = 2r + 1 -> floor(s^2/4) = r(r+1)
    #
    # and evaluate the two sums in closed form.
    m = n // 2

    if n % 2 == 0:
        answer = m * (m + 1) * (4 * m - 1) // 6
    else:
        answer = m * (m + 1) * (4 * m + 5) // 6

    # Independent route: enumerate the integer pairs (s, t), and test
    # feasibility through the quadratic inequality
    #
    #     x(s-x) >= t.
    #
    # Such a real x exists exactly when the quadratic
    # x^2 - sx + t <= 0 has nonnegative discriminant.
    check = 0

    for s in range(2, n + 1):
        for t in range(1, s * s // 4 + 1):
            if s * s - 4 * t >= 0:
                check += 1

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"For each ordered pair of positive integers (s, t) with 2 <= s <= {n}, "
        f"call (s, t) feasible if there exist positive real numbers x and y "
        f"such that x + y = s and xy >= t. How many feasible ordered pairs "
        f"(s, t) are there?"
    )

    return problem, str(answer)


GROUP = "inequality"
SKILL = "counting"