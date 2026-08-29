import random
import sympy

DOMAIN = "calculus"


def generate(seed):
    rng = random.Random(seed)

    h = rng.randint(-5, 5)
    shift = rng.randint(1, 5)

    left = rng.randint(-8, -1)
    right = rng.randint(1, 8)

    # f'(x) = 2(x-h), so the unique critical point is x = h.
    answer = "Yes" if left < h < right else "No"

    # Independent route: differentiate and solve f'(x)=0.
    x = sympy.symbols("x", real=True)
    f = (x - h) ** 2 + shift

    derivative = sympy.diff(f, x)
    critical_points = sympy.solve(derivative, x)

    check = "Yes" if any(
        point.is_real is True and left < point < right
        for point in critical_points
    ) else "No"

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let f(x) = (x - ({h}))^2 + {shift}. "
        f"Does f have a critical point strictly inside the interval "
        f"({left}, {right})? Answer Yes or No."
    )

    return problem, answer, {"mode": "boolean"}