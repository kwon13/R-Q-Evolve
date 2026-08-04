import random

def generate(seed):
    rng = random.Random(seed)

    x = rng.randint(-6, 6)
    y = rng.randint(-6, 6)
    z = rng.randint(-6, 6)

    # Construct equations where adding all three gives a multiple of x+y+z.
    rows = [
        [rng.randint(1, 4), rng.randint(-3, 3), rng.randint(-3, 3)],
        [rng.randint(-3, 3), rng.randint(1, 4), rng.randint(-3, 3)],
        [0, 0, 0],
    ]

    # Make row sums by column equal to k, so E1+E2+E3 = k(x+y+z).
    k = rng.choice([2, 3, 4, 5])
    rows[2] = [
        k - rows[0][0] - rows[1][0],
        k - rows[0][1] - rows[1][1],
        k - rows[0][2] - rows[1][2],
    ]

    sol = [x, y, z]
    rhs = [sum(row[j] * sol[j] for j in range(3)) for row in rows]
    answer = (sum(rhs)) // k

    def fmt(row, rhs_val):
        parts = []
        for coef, var in zip(row, ["x", "y", "z"]):
            if coef == 0:
                continue
            sign = "+" if coef > 0 else "-"
            abscoef = abs(coef)
            term = var if abscoef == 1 else f"{abscoef}{var}"
            if not parts:
                parts.append(term if coef > 0 else f"-{term}")
            else:
                parts.append(f" {sign} {term}")
        return "".join(parts) + f" = {rhs_val}"

    problem = (
        f"Suppose x, y, and z satisfy the following system:\n"
        f"{fmt(rows[0], rhs[0])}\n"
        f"{fmt(rows[1], rhs[1])}\n"
        f"{fmt(rows[2], rhs[2])}\n"
        f"Find x + y + z."
    )

    return problem, str(answer)

CONCEPT_GROUP = "algebra"
CONCEPT_TYPE = "algebra.linear_system_sum"
