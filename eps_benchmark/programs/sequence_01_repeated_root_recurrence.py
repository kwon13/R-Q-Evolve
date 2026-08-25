import random


MAX_ATTEMPTS = 200
PRIME_MODULI = [1009, 1013, 1019, 1021, 1031]


def generate(seed):
    rng = random.Random(seed)

    for _ in range(MAX_ATTEMPTS):
        root = rng.randint(2, 12)
        constant = rng.randint(2, 25)
        slope = rng.choice(
            [value for value in range(-12, 13) if value != 0]
        )
        first = (constant + slope) * root
        if first == 0:
            continue

        index = rng.randint(45, 180)
        modulus = rng.choice(PRIME_MODULI)

        # The characteristic polynomial is (x-root)^2, and the initial values
        # select a_n = (constant + slope*n) * root^n.
        answer = (
            (constant + slope * index)
            * pow(root, index, modulus)
        ) % modulus

        # Independent route: advance the stated recurrence one term at a time,
        # reducing modulo the requested prime after every operation.
        previous = constant % modulus
        current = first % modulus

        for _ in range(index - 1):
            previous, current = (
                current,
                (
                    2 * root * current
                    - root * root * previous
                ) % modulus,
            )

        check = current
        assert answer == check, f"answer={answer} check={check}"

        visible_magnitudes = {
            abs(constant),
            abs(first),
            2 * root,
            root * root,
            index,
            modulus,
        }
        if answer < 100 or answer in visible_magnitudes:
            continue

        problem = (
            f"An integer sequence is defined by a_0 = {constant}, "
            f"a_1 = {first}, and a_(n+2) = {2 * root} a_(n+1) - "
            f"{root * root} a_n for every n >= 0. Find the least "
            f"nonnegative residue of a_{index} modulo {modulus}."
        )

        return problem, str(answer)

    raise ValueError("failed to sample a nontrivial recurrence instance")


GROUP = "sequence"
SKILL = "transformation"
