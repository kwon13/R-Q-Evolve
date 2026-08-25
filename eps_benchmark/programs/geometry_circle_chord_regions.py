import random


def _choose(total, selected):
    if selected < 0 or selected > total:
        return 0
    selected = min(selected, total - selected)
    value = 1
    for index in range(1, selected + 1):
        value = value * (total - selected + index) // index
    return value


def generate(seed):
    rng = random.Random(seed)
    number_of_points = rng.randint(10, 90)

    # Every pair of boundary points supplies a chord.  Every choice of four
    # points supplies one interior crossing, and the general-position condition
    # makes all of those crossings distinct.
    answer = (
        1
        + _choose(number_of_points, 2)
        + _choose(number_of_points, 4)
    )

    # Independent route: add the marked points one at a time.  The new point's
    # chords create (m-1) base pieces plus one further piece for every triple of
    # old points whose chords meet a new chord.
    check = 1
    for current_size in range(2, number_of_points + 1):
        check += current_size - 1 + _choose(current_size - 1, 3)

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"There are {number_of_points} marked points on a circle. Every pair "
        "of marked points is joined by a straight chord. No three chords have "
        "a common point in the interior of the circle. Into how many regions "
        "do the chords divide the disk?"
    )

    return problem, str(answer)


GROUP = "geometry"
SKILL = "counting"
