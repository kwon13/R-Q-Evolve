import random

def generate(seed):
    rng = random.Random(seed)

    # INVARIANT: s must stay a multiple of 3. The real optimum is a = b = c
    # = s/3, so the answer ((s + 3k)/3)^3 is an integer exactly when 3 | s.
    # Keep the invariant in the construction (3 * m), never in a value list.
    m = rng.randint(2, 14)          # free parameter; widen freely
    s = 3 * m                       # s in {6, 9, ..., 42}
    k = rng.randint(1, 7)           # k >= 1 keeps every factor positive

    assert s % 3 == 0 and k >= 1, (
        "invariant violated: s must stay a multiple of 3 and k >= 1"
    )

    max_product = (m + k) ** 3      # = ((s + 3k)/3)^3, exact since s = 3m

    # Independent route: search every positive integer triple summing to s.
    # Since 3 | s, the real optimum a = b = c = s/3 is a lattice point, so
    # the searched maximum must equal the formula.
    check = max(
        (a + k) * (b + k) * (s - a - b + k)
        for a in range(1, s - 1)
        for b in range(1, s - a)
    )
    assert max_product == check, f"max_product={max_product} check={check}"

    problem = (
        f"Let a, b, c be positive real numbers with a + b + c = {s}. "
        f"What is the maximum value of (a + {k})(b + {k})(c + {k})?"
    )

    return problem, str(max_product)

GROUP = "inequality"
SKILL = "transformation"