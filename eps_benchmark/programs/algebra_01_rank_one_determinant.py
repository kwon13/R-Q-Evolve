import random
from itertools import permutations


MAX_ATTEMPTS = 200


def _determinant_by_permutations(matrix):
    """Compute a small determinant directly from its defining expansion."""
    size = len(matrix)
    total = 0

    for permutation in permutations(range(size)):
        inversions = sum(
            1
            for i in range(size)
            for j in range(i + 1, size)
            if permutation[i] > permutation[j]
        )
        term = -1 if inversions % 2 else 1

        for row in range(size):
            term *= matrix[row][permutation[row]]

        total += term

    return total


def generate(seed):
    rng = random.Random(seed)
    diagonal_choices = list(range(-12, -2)) + list(range(3, 13))

    for _ in range(MAX_ATTEMPTS):
        size = rng.choice([4, 5])
        diagonal = [rng.choice(diagonal_choices) for _ in range(size)]
        u = [rng.randint(-6, 6) for _ in range(size)]
        v = [rng.randint(-6, 6) for _ in range(size)]

        if not any(u) or not any(v):
            continue

        # For diagonal D, the rank-one determinant identity gives
        #
        #   det(D + u v^T)
        #     = det(D) + sum_i u_i v_i product_{j != i} D_jj.
        #
        # This form stays integral and does not require any rational arithmetic.
        determinant_d = 1
        for entry in diagonal:
            determinant_d *= entry

        answer = determinant_d + sum(
            u[i] * v[i] * (determinant_d // diagonal[i])
            for i in range(size)
        )

        # Independent route: materialize the full matrix and use the Leibniz
        # expansion, without the diagonal-plus-rank-one identity.
        matrix = [
            [
                (diagonal[i] if i == j else 0) + u[i] * v[j]
                for j in range(size)
            ]
            for i in range(size)
        ]
        check = _determinant_by_permutations(matrix)

        assert answer == check, f"answer={answer} check={check}"

        # Keep the requested value from being one of the small data values
        # printed in the statement, and avoid near-trivial determinants.
        visible_magnitudes = {
            size,
            *(abs(value) for value in diagonal),
            *(abs(value) for value in u),
            *(abs(value) for value in v),
        }
        if abs(answer) < 100 or abs(answer) in visible_magnitudes:
            continue

        diagonal_text = ", ".join(str(value) for value in diagonal)
        u_text = ", ".join(str(value) for value in u)
        v_text = ", ".join(str(value) for value in v)
        problem = (
            f"Let D be the {size} by {size} diagonal matrix with diagonal "
            f"entries ({diagonal_text}). Let u = ({u_text})^T and "
            f"v = ({v_text})^T, and define A = D + u v^T. Find det(A)."
        )

        return problem, str(answer)

    raise ValueError("failed to sample a nontrivial determinant instance")


GROUP = "algebra"
SKILL = "transformation"
