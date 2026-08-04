import math
import random


def generate(seed):
    rng = random.Random(seed)

    # b is even so that every case yields an integer: the solutions come in
    # pairs -beta +/- sqrt(t), whose squares add to 2*beta**2 + 2*t.
    # |beta| >= 2 keeps the single-solution case away from the guessable
    # answers 0 and 1.
    beta = rng.choice([-9, -8, -7, -6, -5, -4, -3, -2, 2, 3, 4, 5, 6, 7, 8, 9])
    k = rng.randint(1, 12)
    b = 2 * beta

    # |f(x)| = k splits into f(x) = k and f(x) = -k. Writing f(x) as
    # (x + beta)**2 + (c - beta**2), the two branches have shifted radicands
    #     upper = beta**2 - c + k     (from f(x) = k)
    #     lower = beta**2 - c - k     (from f(x) = -k)
    # which differ by 2k, so how many solutions exist is decided by where
    # beta**2 - c sits relative to +/- k. c is chosen from the interval that
    # realises the wanted case, rotating with the seed so every branch occurs.
    case = seed % 4
    if case == 0:
        c = beta * beta - k - rng.randint(1, 25)   # lower > 0  -> four solutions
    elif case == 1:
        c = beta * beta - k                        # lower == 0 -> three solutions
    elif case == 2:
        c = beta * beta + rng.randint(-k + 1, k - 1)   # lower < 0 < upper -> two
    else:
        c = beta * beta + k                        # upper == 0 -> one solution

    upper = beta * beta - c + k
    lower = beta * beta - c - k

    # Each branch contributes a different expression, and no single one of them
    # is valid across the four cases -- which is the whole solve.
    answer = 0
    for radicand in (upper, lower):
        if radicand > 0:
            answer += 2 * beta * beta + 2 * radicand   # two solutions
        elif radicand == 0:
            answer += beta * beta                      # one repeated solution

    # Independent route: solve both quadratics with the ordinary formula in
    # floating point and add up the squares of the distinct real solutions,
    # instead of deciding the case from upper and lower.
    solutions = []
    for constant in (c - k, c + k):
        disc = b * b - 4 * constant
        if disc < 0:
            continue
        root = math.sqrt(disc)
        solutions.append((-b + root) / 2)
        if root > 0:
            solutions.append((-b - root) / 2)
    check = sum(x * x for x in solutions)
    assert abs(check - answer) < 1e-6, f"answer={answer} check={check}"

    linear = f"+ {b}x" if b > 0 else f"- {abs(b)}x"
    constant = f"+ {c}" if c > 0 else f"- {abs(c)}"
    problem = (
        f"Let f(x) = x^2 {linear} {constant}. Find the sum of the squares of "
        f"all real solutions of |f(x)| = {k}. State only the integer."
    )

    return problem, str(answer)


GROUP = "algebra"
SKILL = "casework"
