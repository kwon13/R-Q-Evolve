import random


def generate(seed):
    rng = random.Random(seed)

    while True:
        ax = rng.randint(-20, 20)
        ay = rng.randint(-20, 20)
        bx = rng.randint(-20, 20)
        by = rng.randint(-20, 20)
        cx = rng.randint(-20, 20)
        cy = rng.randint(-20, 20)

        twice_area = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        if twice_area > 0:
            break

    if twice_area % 2 == 0:
        answer = str(twice_area // 2)
    else:
        answer = f"{twice_area}/2"

    problem = (
        f"Triangle ABC has A = ({ax}, {ay}), B = ({bx}, {by}), "
        f"and C = ({cx}, {cy}). Find its area."
    )

    return problem, answer

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.trig_area"
