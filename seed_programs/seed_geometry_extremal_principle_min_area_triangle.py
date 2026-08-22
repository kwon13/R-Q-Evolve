import random
from itertools import combinations

MAX_ATTEMPTS = 200


def _cross(a, b, c):
    return (
        (b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0])
    )


def _digit_runs(text):
    # Every maximal run of digits in the rendered text. The answer must not be
    # one of them: lint_problem_instance rejects an instance whose answer shows
    # up as a standalone number in the statement, and a doubled area lands in
    # the same range as the coordinates it was computed from.
    runs = set()
    current = ""

    for ch in text:
        if ch.isdigit():
            current += ch
        elif current:
            runs.add(current)
            current = ""

    if current:
        runs.add(current)

    return runs


def _strictly_inside(p, a, b, c):
    s1 = _cross(a, b, p)
    s2 = _cross(b, c, p)
    s3 = _cross(c, a, p)

    return (
        (s1 > 0 and s2 > 0 and s3 > 0)
        or (s1 < 0 and s2 < 0 and s3 < 0)
    )


def generate(seed):
    rng = random.Random(seed)

    # The extremal argument does not depend on n, so n stays where the triple
    # count is still inspectable by hand: C(7,3) = 35. A larger set turns the
    # question into an exhaustive search over triples instead of the argument
    # the problem is meant to test.
    n = rng.randint(5, 7)

    for _ in range(MAX_ATTEMPTS):
        # Build a point set in general position.
        # A new point is accepted only if it is distinct and creates
        # no collinear triple with previously accepted points.
        points = []

        while len(points) < n:
            p = (
                rng.randint(-15, 15),
                rng.randint(-15, 15),
            )

            if p in points:
                continue

            if any(
                _cross(a, b, p) == 0
                for a, b in combinations(points, 2)
            ):
                continue

            points.append(p)

        # Intended route: show the emptiness condition is vacuous at the
        # optimum, then drop it.
        #
        # Let T be a triangle of least area among ALL triangles with vertices
        # in S. No point P of S can lie strictly inside T: joining P to the
        # three vertices of T splits T into three triangles of positive area,
        # each strictly smaller than T and each with its vertices in S,
        # contradicting the minimality of T.
        #
        # So the unconstrained minimum is already attained by an empty
        # triangle, and the constrained minimum the problem asks for equals
        # it. That turns the question into a plain minimum over all triples --
        # no interior test is ever performed on this route.
        answer = min(
            abs(_cross(points[i], points[j], points[k]))
            for i, j, k in combinations(range(n), 3)
        )

        point_text = ", ".join(
            f"({x}, {y})"
            for x, y in points
        )

        # Redraw when the doubled area happens to be printed as one of the
        # coordinates, or as the size of the set.
        if str(answer) in _digit_runs(point_text) | {str(n)}:
            continue

        break
    else:
        raise ValueError("no point set whose answer stays out of the statement")

    # Independent route: take the emptiness condition literally. Enumerate the
    # triples, run the interior test on each one, discard every triangle that
    # contains another point of S, and minimise over what survives.
    check = None

    for i, j, k in combinations(range(n), 3):
        a = points[i]
        b = points[j]
        c = points[k]

        if any(
            _strictly_inside(points[r], a, b, c)
            for r in range(n)
            if r not in (i, j, k)
        ):
            continue

        area2 = abs(_cross(a, b, c))

        if check is None or area2 < check:
            check = area2

    assert check is not None
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Let S be the following set of {n} points in the plane, no three "
        f"of which are collinear: {point_text}. Call a triangle admissible "
        f"if its three vertices belong to S and no other point of S lies "
        f"strictly inside it. Over all admissible triangles, find the "
        f"minimum possible value of twice the area. State only the integer."
    )

    return problem, str(answer)


GROUP = "geometry"
SKILL = "extremal_principle"
