import random


S_POOL = [6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42]
K_POOL = [1, 2, 3, 4, 5, 6, 7]


def generate(seed):
    rng = random.Random(seed)

    s = rng.choice(S_POOL)
    k = rng.choice(K_POOL)

    max_product = ((s + 3 * k) // 3) ** 3

    problem = (
        f"Let a, b, c be positive real numbers with a + b + c = {s}. "
        f"What is the maximum value of (a + {k})(b + {k})(c + {k})?"
    )

    return problem, str(max_product)

CONCEPT_GROUP = "inequality"
CONCEPT_TYPE = "inequality.am_gm_product"
