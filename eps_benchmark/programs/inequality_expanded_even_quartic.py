"""Held-out inequality family: exposing an even quartic after recentering.

The primary route recenters the expanded polynomial and substitutes the
square of the shifted variable.  Direct evaluation at every derivative
critical point supplies an independent check.
"""

import random


def _format_polynomial(coefficients):
    pieces = []
    degree = len(coefficients) - 1
    for index, coefficient in enumerate(coefficients):
        power = degree - index
        if coefficient == 0:
            continue
        magnitude = abs(coefficient)
        if power == 0:
            term = str(magnitude)
        elif power == 1:
            term = "x" if magnitude == 1 else f"{magnitude}x"
        else:
            term = f"x^{power}" if magnitude == 1 else f"{magnitude}x^{power}"
        if not pieces:
            pieces.append(term if coefficient > 0 else f"-{term}")
        else:
            sign = "+" if coefficient > 0 else "-"
            pieces.append(f" {sign} {term}")
    return "".join(pieces)


def _evaluate(coefficients, x):
    value = 0
    for coefficient in coefficients:
        value = value * x + coefficient
    return value


def generate(seed):
    rng = random.Random(seed)

    while True:
        center = rng.randint(-5, 5)
        if center == 0:
            continue
        root_offset = rng.randint(2, 7)
        e = rng.randint(2, 20)
        h = root_offset * root_offset
        d = e + h
        coefficients = [
            1,
            -4 * center,
            6 * center * center - 2 * h,
            -4 * center ** 3 + 4 * h * center,
            center ** 4 - 2 * h * center * center + d * d,
        ]
        answer = e * (2 * d - e)
        if answer not in {abs(value) for value in coefficients}:
            break

    derivative = [
        4 * coefficients[0],
        3 * coefficients[1],
        2 * coefficients[2],
        coefficients[3],
    ]
    # Derive all stationary points from the rendered coefficients alone.
    # The construction makes them integral; the fixed search interval strictly
    # contains every possible root for the parameter ranges above.
    critical_points = [
        x for x in range(-20, 21) if _evaluate(derivative, x) == 0
    ]
    assert len(critical_points) == 3
    check = min(_evaluate(coefficients, x) for x in critical_points)
    assert answer == check, f"answer={answer} check={check}"

    polynomial = _format_polynomial(coefficients)
    problem = (
        f"For real x, let F(x) = {polynomial}. Determine the minimum "
        f"possible value of F(x)."
    )
    return problem, str(answer)


GROUP = "inequality"
SKILL = "transformation"
