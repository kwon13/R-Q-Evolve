"""Held-out algebra family: a root-value product computed as a resultant.

The intended route reduces the cubic modulo the quadratic and then uses
Vieta's relations.  An exact Sylvester determinant supplies an independent
generator-side check.
"""

import math
import random


def _bareiss_determinant(matrix):
    values = [row[:] for row in matrix]
    size = len(values)
    previous_pivot = 1
    sign = 1

    for column in range(size - 1):
        if values[column][column] == 0:
            swap_row = next(
                row
                for row in range(column + 1, size)
                if values[row][column] != 0
            )
            values[column], values[swap_row] = values[swap_row], values[column]
            sign *= -1

        pivot = values[column][column]
        for row in range(column + 1, size):
            for col in range(column + 1, size):
                numerator = (
                    values[row][col] * pivot
                    - values[row][column] * values[column][col]
                )
                assert numerator % previous_pivot == 0
                values[row][col] = numerator // previous_pivot
            values[row][column] = 0
        previous_pivot = pivot

    return sign * values[-1][-1]


def _sylvester_resultant(a, b, c, d, e):
    matrix = [
        [1, a, b, 0, 0],
        [0, 1, a, b, 0],
        [0, 0, 1, a, b],
        [1, c, d, e, 0],
        [0, 1, c, d, e],
    ]
    return _bareiss_determinant(matrix)


def _format_monic(coefficients):
    degree = len(coefficients)
    pieces = [f"x^{degree}"]
    for index, coefficient in enumerate(coefficients):
        power = degree - index - 1
        if coefficient == 0:
            continue
        sign = "+" if coefficient > 0 else "-"
        magnitude = abs(coefficient)
        if power == 0:
            term = str(magnitude)
        elif power == 1:
            term = "x" if magnitude == 1 else f"{magnitude}x"
        else:
            term = f"x^{power}" if magnitude == 1 else f"{magnitude}x^{power}"
        pieces.append(f" {sign} {term}")
    return "".join(pieces)


def generate(seed):
    rng = random.Random(seed)

    while True:
        a = rng.randint(-10, 10)
        b = rng.randint(-14, 14)
        c = rng.randint(-8, 8)
        d = rng.randint(-12, 12)
        e = rng.randint(-15, 15)

        discriminant = a * a - 4 * b
        if discriminant == 0:
            continue
        if discriminant > 0 and math.isqrt(discriminant) ** 2 == discriminant:
            continue

        # Modulo x^2 + ax + b, the cubic is ux + v.
        u = a * a - b - a * c + d
        v = a * b - b * c + e
        answer = b * u * u - a * u * v + v * v
        if 1_000 <= abs(answer) < 100_000_000:
            break

    check = _sylvester_resultant(a, b, c, d, e)
    assert answer == check, f"answer={answer} check={check}"

    quadratic = _format_monic([a, b])
    cubic = _format_monic([c, d, e])
    problem = (
        f"Let alpha and beta be the two roots of P(x) = {quadratic}. "
        f"For Q(x) = {cubic}, determine the integer "
        f"Q(alpha)Q(beta)."
    )
    return problem, str(answer)


GROUP = "algebra"
SKILL = "transformation"
