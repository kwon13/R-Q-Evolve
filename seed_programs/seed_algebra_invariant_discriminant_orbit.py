import random


def _shift_quadratic(b, c, t):
    # If P(x) = x^2 + bx + c, then P(x+t) has coefficients
    # x^2 + (b+2t)x + (c+bt+t^2).
    return b + 2 * t, c + b * t + t * t


def generate(seed):
    rng = random.Random(seed)

    b = rng.randint(-12, 12)
    c = rng.randint(-20, 30)

    # Every move P(x) -> P(x+1) or P(x-1) preserves the discriminant
    # b^2 - 4c. Conversely, two monic integer quadratics with the same
    # discriminant have linear coefficients of the same parity, so one is
    # an integer translate of the other.
    invariant = b * b - 4 * c

    shifts = rng.sample(
        [t for t in range(-8, 9) if t != 0],
        5,
    )
    target_slot = rng.randrange(5)

    candidates = []

    for i, t in enumerate(shifts):
        qb, qc = _shift_quadratic(b, c, t)

        if i != target_slot:
            # Perturb only the constant term, which changes the discriminant
            # by a nonzero multiple of 4 and therefore leaves the orbit.
            delta = rng.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
            qc += delta

        candidates.append((qb, qc))

    # Intended route: the reachable candidate is exactly the one preserving
    # the discriminant.
    matching = [
        i + 1
        for i, (qb, qc) in enumerate(candidates)
        if qb * qb - 4 * qc == invariant
    ]

    assert len(matching) == 1
    answer = matching[0]

    # Independent route: directly perform the allowed moves in coefficient
    # space, without using the discriminant.
    reachable = {(b, c)}

    for _ in range(8):
        next_states = set(reachable)

        for rb, rc in reachable:
            # P(x+1)
            next_states.add(
                (rb + 2, rc + rb + 1)
            )

            # P(x-1)
            next_states.add(
                (rb - 2, rc - rb + 1)
            )

        reachable = next_states

    reached_indices = [
        i + 1
        for i, candidate in enumerate(candidates)
        if candidate in reachable
    ]

    assert len(reached_indices) == 1
    check = reached_indices[0]

    assert answer == check, f"answer={answer} check={check}"

    def poly_text(qb, qc):
        linear = f"+ {qb}x" if qb >= 0 else f"- {abs(qb)}x"
        constant = f"+ {qc}" if qc >= 0 else f"- {abs(qc)}"
        return f"x^2 {linear} {constant}"

    start = poly_text(b, c)

    candidate_text = "; ".join(
        f"Q_{i + 1}(x) = {poly_text(qb, qc)}"
        for i, (qb, qc) in enumerate(candidates)
    )

    problem = (
        f"Start with P(x) = {start}. In one move, P(x) may be replaced by "
        f"P(x+1) or by P(x-1), and any number of moves may be performed. "
        f"Exactly one of the following polynomials is reachable: "
        f"{candidate_text}. Find its index. State only the integer."
    )

    return problem, str(answer)


GROUP = "algebra"
SKILL = "invariant"