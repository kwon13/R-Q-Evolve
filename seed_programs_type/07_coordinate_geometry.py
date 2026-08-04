import random
import sympy

def generate(seed):
    rng = random.Random(seed)

    a1 = rng.randint(1, 5)
    b1 = rng.randint(-5, 5)
    c1 = rng.randint(-20, 20)
    a2 = rng.randint(-5, 5)
    b2 = rng.randint(1, 5)
    while a1 * b2 == a2 * b1:
        b2 = rng.randint(1, 5)
    c2 = rng.randint(-20, 20)
    det = a1 * b2 - a2 * b1
    x_num = c1 * b2 - c2 * b1
    y_num = a1 * c2 - a2 * c1
    x = sympy.Rational(x_num, det)
    y = sympy.Rational(y_num, det)
    answer = sympy.sqrt(x ** 2 + y ** 2)
    problem = (
        f"Two lines are defined by {a1}x + {b1}y = {c1} and "
        f"{a2}x + {b2}y = {c2}. Find the exact distance from their "
        f"intersection point to the origin. Express the answer as a "
        f"simplified radical or rational number."
    )

    return problem, str(answer)

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.line_intersection_distance"
