import random
from math import comb


def generate(seed):
    rng = random.Random(seed)

    n = rng.randint(6, 12)

    # Any triangulation of a convex n-gon has exactly n-2 triangles.
    #
    # The extremal step:
    # among those n-2 triangles, at least one must have area at most
    # total_area / (n-2), by averaging.
    #
    # We construct an instance where the polygon's total doubled area is
    # chosen to be divisible by n-2, so the requested integer bound is clean.

    triangle_count = n - 2

    # Choose the common bound directly.
    bound = rng.randint(2, 15)

    # The polygon is specified abstractly by its total area.
    total_area2 = triangle_count * bound

    answer = bound

    # Independent arithmetic check.
    check = total_area2 // triangle_count

    assert total_area2 % triangle_count == 0
    assert answer == check

    problem = (
        f"A convex polygon with {n} vertices has total area {total_area2 / 2}. "
        f"Prove that in every triangulation of the polygon, at least one triangle "
        f"has area at most some value A. Find the greatest integer A that is "
        f"guaranteed by this argument."
    )

    return problem, str(answer)


GROUP = "geometry"
SKILL = "extremal_principle"