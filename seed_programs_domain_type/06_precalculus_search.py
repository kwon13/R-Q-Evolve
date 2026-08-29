import random

DOMAIN = "precalculus"


def generate(seed):
    rng = random.Random(seed)

    # Choose two distinct integer solutions.
    a = rng.randint(1, 5)
    b = rng.randint(a + 1, 8)

    # Let y = 2^x. Construct
    #
    #   (y - 2^a)(y - 2^b) = 0
    #
    # so the only solutions are x = a and x = b.
    A = 2**a
    B = 2**b

    coefficient = A + B
    constant = A * B

    candidates = [a, b]

    # Intended route:
    #   2^(2x) - (A+B)2^x + AB = 0
    #   (2^x-A)(2^x-B) = 0
    # hence x = a or x = b.
    answer = r"\{" + ",".join(str(x) for x in candidates) + r"\}"

    # Independent route:
    # Exhaustively test every integer in the stated bounded interval.
    check_values = []

    for x in range(0, 10):
        if 2 ** (2 * x) - coefficient * 2**x + constant == 0:
            check_values.append(x)

    check = r"\{" + ",".join(str(x) for x in check_values) + r"\}"

    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Find all integers x with 0 <= x <= 9 satisfying "
        f"2^(2x) - {coefficient}*2^x + {constant} = 0."
    )

    verifier = {
        "mode": "set",
        "elements": [str(x) for x in candidates],
    }

    return problem, answer, verifier