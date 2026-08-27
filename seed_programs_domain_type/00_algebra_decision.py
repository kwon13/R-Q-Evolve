import random
import sympy

DOMAIN = "algebra"


def generate(seed):
    rng = random.Random(seed)
    b = rng.randint(-9, 9)
    c = rng.randint(-12, 12)

    discriminant = b * b - 4 * c
    answer = "Yes" if discriminant > 0 else "No"

    x = sympy.symbols("x", real=True)
    roots = sympy.solve(x**2 + b * x + c, x)
    real_distinct_roots = {root for root in roots if root.is_real is True}
    check = "Yes" if len(real_distinct_roots) == 2 else "No"
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Does the quadratic equation x^2 + ({b})x + ({c}) = 0 have "
        "two distinct real solutions? Answer Yes or No."
    )
    return problem, answer, {"mode": "boolean"}
