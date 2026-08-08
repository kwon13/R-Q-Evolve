import random


def generate(seed):
    rng = random.Random(seed)

    # Five weighted centers. Distinct weights make the six regions carry
    # different slopes. Their total is forced odd, so no prefix can have the
    # same weight as its complement and no interval can be flat (which would
    # otherwise allow infinitely many integer solutions at one target value).
    centers = sorted(rng.sample(range(-30, 31), 5))
    omitted_weight = rng.choice([2, 4, 6])
    weights = [weight for weight in range(1, 7) if weight != omitted_weight]
    rng.shuffle(weights)
    assert len(set(weights)) == 5 and sum(weights) % 2 == 1
    witness = rng.randint(centers[0] - 12, centers[-1] + 12)
    target = sum(
        weight * abs(witness - center)
        for weight, center in zip(weights, centers)
    )

    # Region-by-region route: on each interval the sum is linear, so solve the
    # linear equation there and keep the root only if it lands in the region.
    roots = set()
    bounds = [None] + centers + [None]
    for i in range(len(centers) + 1):
        low, high = bounds[i], bounds[i + 1]
        slope = 0
        offset = 0
        for weight, center in zip(weights, centers):
            if low is not None and center <= low:
                slope += weight
                offset -= weight * center
            else:
                slope -= weight
                offset += weight * center
        if slope == 0:
            continue
        numerator = target - offset
        if numerator % slope:
            continue
        candidate = numerator // slope
        if (low is None or candidate >= low) and (high is None or candidate <= high):
            roots.add(candidate)
    answer = sum(value * value for value in roots)

    # Independent route: scan every integer that could possibly satisfy the
    # equation. Outside the outermost centers the sum grows by sum(weights)
    # per unit, so this radius is safely beyond every root.
    radius = target + abs(centers[0]) + abs(centers[-1]) + 2
    brute = [
        value
        for value in range(-radius, radius + 1)
        if sum(w * abs(value - c) for w, c in zip(weights, centers)) == target
    ]
    check = sum(value * value for value in brute)
    assert brute, "no integer solution"
    assert answer == check, f"answer={answer} check={check}"

    terms = " + ".join(
        f"{w}|x - ({c})|" for w, c in zip(weights, centers)
    )
    problem = (
        f"Find the sum of the squares of all integer solutions x to "
        f"{terms} = {target}. State only the integer."
    )
    return problem, str(answer)


GROUP = "algebra"
SKILL = "casework"
