"""Held-out counting family: spanning trees of complete tripartite graphs.

The primary route uses the complete-multipartite closed form.  The independent
check builds a Laplacian cofactor and evaluates its exact determinant with the
fraction-free Bareiss algorithm.
"""

import random


def _closed_form(part_sizes):
    a, b, c = part_sizes
    total = a + b + c
    return (
        total
        * (b + c) ** (a - 1)
        * (a + c) ** (b - 1)
        * (a + b) ** (c - 1)
    )


def _bareiss_determinant(matrix):
    values = [row[:] for row in matrix]
    size = len(values)

    if size == 0:
        return 1
    if size == 1:
        return values[0][0]

    sign = 1
    previous_pivot = 1

    for column in range(size - 1):
        if values[column][column] == 0:
            swap_row = next(
                row
                for row in range(column + 1, size)
                if values[row][column] != 0
            )
            values[column], values[swap_row] = values[swap_row], values[column]
            sign *= -1

        pivot = values[column][column]

        for row in range(column + 1, size):
            for col in range(column + 1, size):
                numerator = (
                    values[row][col] * pivot
                    - values[row][column] * values[column][col]
                )
                assert numerator % previous_pivot == 0
                values[row][col] = numerator // previous_pivot
            values[row][column] = 0

        previous_pivot = pivot

    return sign * values[-1][-1]


def _matrix_tree_count(part_sizes):
    total = sum(part_sizes)
    part_of_vertex = []
    for part_index, part_size in enumerate(part_sizes):
        part_of_vertex.extend([part_index] * part_size)

    laplacian = [[0] * total for _ in range(total)]

    for row in range(total):
        row_part = part_of_vertex[row]
        laplacian[row][row] = total - part_sizes[row_part]

        for col in range(total):
            if row != col and row_part != part_of_vertex[col]:
                laplacian[row][col] = -1

    cofactor = [row[:-1] for row in laplacian[:-1]]
    return _bareiss_determinant(cofactor)


def generate(seed):
    rng = random.Random(seed)

    # Sorted sizes avoid counting isomorphic permutations as different
    # rendered instances.  Singleton parts are valid, and the answer cap keeps
    # recognition of the tree formula -- rather than long multiplication --
    # as the load-bearing difficulty.
    options = [
        (a, b, c)
        for a in range(1, 15)
        for b in range(a, 15)
        for c in range(b, 15)
        if a + b + c <= 18
        and _closed_form((a, b, c)) >= 100
        and _closed_form((a, b, c)) < 100_000_000
    ]
    part_sizes = rng.choice(options)

    answer = _closed_form(part_sizes)

    # Independent route: the cofactor is constructed from adjacency alone;
    # no factor from the closed expression is used by this computation.
    check = _matrix_tree_count(part_sizes)

    assert answer == check, f"answer={answer} check={check}"

    a, b, c = part_sizes
    problem = (
        f"Let G be the complete tripartite graph whose three independent "
        f"parts have sizes {a}, {b}, and {c}. Every two vertices in different "
        f"parts are adjacent, and no two vertices in the same part are "
        f"adjacent. How many spanning trees does G have?"
    )
    return problem, str(answer)


GROUP = "combinatorics"
SKILL = "counting"
