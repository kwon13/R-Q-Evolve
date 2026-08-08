import random


def _pair(index, alpha, beta):
    """Return (a(index), a(index + 1)) by halving the index.

    The two branches are not symmetric, so the recursion cannot be collapsed
    into a single closed form; each bit of the index applies a different map.
    """
    if index == 0:
        return 0, 1
    if index == 1:
        # A base case, not a recursive one: the stated rules hold for m >= 1,
        # so a(1) is given. Descending into a(2*0+1) = a(0) + beta*a(1) would
        # instead force a(1) = beta*a(1). Stern's sequence hides this -- at
        # alpha = beta = 1 that equation happens to hold for the true value.
        return 1, alpha
    left, right = _pair(index // 2, alpha, beta)
    if index % 2 == 0:
        return alpha * left, left + beta * right
    return left + beta * right, alpha * right


def generate(seed):
    rng = random.Random(seed)

    # alpha = beta = 1 is Stern's diatomic sequence, which a solver may simply
    # recall. Weighting the two branches removes that route: the value has to
    # come from the recursion itself.
    alpha = rng.choice([2, 3])
    beta = rng.choice([2, 3, 4])
    index = rng.randint(4_000, 9_000)
    modulus = 1_000_003

    answer = _pair(index, alpha, beta)[0] % modulus

    # Independent route: build the sequence bottom-up from the two stated
    # rules, instead of descending on the binary expansion of the index.
    values = [0] * (index + 2)
    values[1] = 1
    for position in range(2, index + 2):
        half = position // 2
        if position % 2 == 0:
            values[position] = (alpha * values[half]) % modulus
        else:
            values[position] = (values[half] + beta * values[half + 1]) % modulus
    check = values[index]
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        "The sequence a(0), a(1), ... is defined by a(0) = 0, a(1) = 1, "
        f"a(2m) = {alpha} * a(m), and a(2m+1) = a(m) + {beta} * a(m+1) for "
        f"every positive integer m. Find a({index}) modulo {modulus}. "
        "State only the integer."
    )
    return problem, str(answer)


GROUP = "sequence"
SKILL = "induction"
