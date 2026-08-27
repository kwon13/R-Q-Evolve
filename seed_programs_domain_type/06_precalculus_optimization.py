import math
import random

DOMAIN = "precalculus"


def generate(seed):
    rng = random.Random(seed)
    scale = rng.randint(1, 9)
    a = 3 * scale
    b = 4 * scale

    answer = 5 * scale
    check = math.isqrt(a * a + b * b)
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Find the maximum value of {a} sin(theta) + {b} cos(theta) over all "
        "real numbers theta."
    )
    return problem, str(answer), {"mode": "expression"}
