import random

DOMAIN = "geometry"


def generate(seed):
    rng = random.Random(seed)
    a = rng.randint(3, 10)
    b = rng.randint(a + 1, 2 * a + 2)

    candidates = [b]
    if 2 * a > b:
        candidates.insert(0, a)
    answer = r"\{" + ",".join(str(x) for x in candidates) + r"\}"

    check_values = []
    for x in range(1, a + b):
        is_triangle = a + b > x and a + x > b and b + x > a
        is_isosceles = a == b or a == x or b == x
        if is_triangle and is_isosceles:
            check_values.append(x)
    check = r"\{" + ",".join(str(x) for x in check_values) + r"\}"
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Find all positive integer lengths x for which side lengths {a}, {b}, "
        "and x form a nondegenerate isosceles triangle."
    )
    verifier = {"mode": "set", "elements": [str(x) for x in candidates]}
    return problem, answer, verifier
