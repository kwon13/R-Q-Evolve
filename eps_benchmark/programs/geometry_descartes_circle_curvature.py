"""Held-out geometry family: the second circle in a Descartes configuration.

The intended route solves the curvature relation for its larger root.  A
reflection in an integral Descartes quadruple independently checks that root.
"""

import math
import random


def _configuration_options():
    initial = (2, 2, 3, 15)
    seen = {initial}
    frontier = [initial]

    while frontier and len(seen) < 600:
        current = frontier.pop(0)
        total = sum(current)
        for index in range(4):
            replacement = 2 * (total - current[index]) - current[index]
            candidate = list(current)
            candidate[index] = replacement
            candidate = tuple(sorted(candidate))
            if (
                min(candidate) > 0
                and max(candidate) <= 200_000
                and candidate not in seen
            ):
                seen.add(candidate)
                frontier.append(candidate)

    by_given = {}
    for quadruple in sorted(seen):
        for index, hidden in enumerate(quadruple):
            given = tuple(
                value
                for position, value in enumerate(quadruple)
                if position != index
            )
            total = sum(given)
            alternate = 2 * total - hidden
            larger = max(hidden, alternate)
            if (
                min(given) > 0
                and min(hidden, alternate) > 0
                and 100 <= larger <= 5_000
                and larger not in given
            ):
                record = (given, hidden, larger)
                previous = by_given.get(given)
                if previous is None or hidden < previous[1]:
                    by_given[given] = record
    return [by_given[given] for given in sorted(by_given)]


def generate(seed):
    rng = random.Random(seed)

    given, hidden, check = rng.choice(_configuration_options())
    first, second, third = given
    radicand = first * second + second * third + third * first
    root = math.isqrt(radicand)
    assert root * root == radicand
    answer = first + second + third + 2 * root

    alternate = 2 * sum(given) - hidden
    assert check == max(hidden, alternate)
    all_curvatures = list(given) + [answer]
    assert sum(all_curvatures) ** 2 == 2 * sum(
        value * value for value in all_curvatures
    )
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Three mutually tangent oriented circles have positive integer "
        f"curvatures {first}, {second}, and {third}. There are two possible "
        f"integer curvatures for a fourth oriented circle tangent to all "
        f"three. What is the larger curvature?"
    )
    return problem, str(answer)


GROUP = "geometry"
SKILL = "transformation"
