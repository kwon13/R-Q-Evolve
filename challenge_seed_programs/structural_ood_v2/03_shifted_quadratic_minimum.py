import random


def generate(seed):
    rng = random.Random(seed)

    weights = rng.sample(range(2, 14), 3)
    shifts = [rng.randint(-9, 9) for _ in range(3)]
    scale = rng.randint(5, 15)
    weight_sum = sum(weights)
    # The stated total is the shifted total plus the shift sum, so the
    # translated problem has a clean weighted-Cauchy optimum.
    inner_total = scale * weight_sum
    total = inner_total + sum(shifts)

    # Translating by the shifts turns this into the plain weighted Cauchy
    # bound; the shifts are what a solver has to remove first.
    answer = inner_total * inner_total // weight_sum

    # Independent route: evaluate the objective at the attainment point
    # u_i = scale * a_i, i.e. x_i = scale * a_i + s_i.
    attain = [scale * weight + shift for weight, shift in zip(weights, shifts)]
    check = sum(
        (value - shift) * (value - shift) // weight
        for value, shift, weight in zip(attain, shifts, weights)
    )
    assert sum(attain) == total, f"attain={sum(attain)} total={total}"
    assert answer == check, f"answer={answer} check={check}"

    a, b, c = weights
    s, t, u = shifts
    problem = (
        f"Let x, y, z be real numbers with x + y + z = {total}, "
        f"x > {s}, y > {t}, and z > {u}. What is the minimum value of "
        f"(x - ({s}))^2/{a} + (y - ({t}))^2/{b} + (z - ({u}))^2/{c}? "
        "State only the integer."
    )
    return problem, str(answer)


GROUP = "inequality"
SKILL = "transformation"
