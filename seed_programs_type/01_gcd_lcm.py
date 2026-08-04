import random
import math

def lcm(x, y):
    return x * y // math.gcd(x, y)


def lcm_list(nums):
    out = 1
    for n in nums:
        out = lcm(out, n)
    return out


def generate(seed):
    rng = random.Random(seed)

    a = rng.choice([6, 8, 10, 12, 15])
    b = rng.choice([9, 12, 14, 16, 18, 20])
    c = rng.choice([10, 15, 20, 21, 24, 25, 30])
    d = rng.choice([7, 11, 13, 17, 19])

    hours = rng.choice([8, 10, 12])
    total_minutes = hours * 60

    lcm_abc = lcm_list([a, b, c])
    lcm_abcd = lcm_list([a, b, c, d])

    together_abc = total_minutes // lcm_abc
    together_abcd = total_minutes // lcm_abcd

    answer = together_abc - together_abcd

    problem = (
        f"Four bus routes depart from the same station. Route A departs every "
        f"{a} minutes, Route B every {b} minutes, Route C every {c} minutes, "
        f"and Route D every {d} minutes. All four depart simultaneously at 6:00 AM. "
        f"During the next {hours} hours, excluding the initial 6:00 AM departure "
        f"but including the endpoint, how many times do Routes A, B, and C depart "
        f"together while Route D does not?"
    )

    return problem, str(answer)

CONCEPT_GROUP = "number_theory"
CONCEPT_TYPE = "number_theory.gcd_lcm_sync"
