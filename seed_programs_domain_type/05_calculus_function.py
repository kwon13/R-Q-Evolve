import random
import sympy

DOMAIN = "calculus"


def generate(seed):
    rng = random.Random(seed)
    exponent = rng.randint(1, 5)
    multiplier = rng.randint(1, 6)
    coefficient = multiplier * (exponent + 1)
    upper = rng.randint(2, 6)

    answer = multiplier * upper ** (exponent + 1)

    x = sympy.symbols("x")
    check = int(sympy.integrate(coefficient * x**exponent, (x, 0, upper)))
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        rf"Evaluate the definite integral $\int_0^{{{upper}}} "
        rf"{coefficient}x^{{{exponent}}}\,dx$."
    )
    return problem, str(answer), {"mode": "expression"}
