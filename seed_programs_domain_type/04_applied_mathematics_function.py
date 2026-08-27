import random

DOMAIN = "applied_mathematics"


def generate(seed):
    rng = random.Random(seed)
    sample_size = rng.randint(4, 12)
    reported_mean = rng.randint(15, 60)
    correction_steps = rng.randint(-3, 5)
    if correction_steps == 0:
        correction_steps = 2
    recorded_value = reported_mean
    corrected_value = recorded_value + sample_size * correction_steps

    answer = reported_mean + correction_steps

    measurements = [reported_mean] * sample_size
    measurements[-1] = corrected_value
    check = sum(measurements) // sample_size
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"A data set of {sample_size} measurements was reported to have mean "
        f"{reported_mean}. One measurement was recorded as {recorded_value}, but "
        f"its correct value is {corrected_value}. What is the corrected mean?"
    )
    return problem, str(answer), {"mode": "expression"}
