import random


def generate(seed):
    rng = random.Random(seed)

    roots = rng.sample([value for value in range(-7, 9) if value != 0], 3)
    exponent = rng.randint(6, 10)
    first, second, third = roots
    e1 = first + second + third
    e2 = first * second + first * third + second * third
    e3 = first * second * third

    # Hidden-root route: the roots are known here but never printed.
    answer = sum(value ** exponent for value in roots)

    # Independent route from the three printed symmetric quantities only, via
    # Newton's identity p_n = e1*p_{n-1} - e2*p_{n-2} + e3*p_{n-3}. Two
    # variables need one previous term; three need two, and the middle sign
    # flips, so the two-variable recurrence does not extend by analogy.
    p0, p1, p2 = 3, e1, e1 * e1 - 2 * e2
    for _ in range(3, exponent + 1):
        p0, p1, p2 = p1, p2, e1 * p2 - e2 * p1 + e3 * p0
    check = p2 if exponent >= 2 else (p1 if exponent == 1 else p0)
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Real numbers x, y, z satisfy x + y + z = {e1}, "
        f"xy + xz + yz = {e2}, and xyz = {e3}. "
        f"Find x^{exponent} + y^{exponent} + z^{exponent}. State only the integer."
    )
    return problem, str(answer)


GROUP = "algebra"
SKILL = "transformation"
