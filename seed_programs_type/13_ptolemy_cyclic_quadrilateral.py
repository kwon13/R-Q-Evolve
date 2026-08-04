import random
from fractions import Fraction

TRIPLES_BY_HYPOTENUSE = {
    65: [(16, 63), (25, 60), (33, 56), (39, 52)],
    85: [(13, 84), (36, 77), (40, 75), (51, 68)],
}


def generate(seed):
    rng = random.Random(seed)
    diagonal = rng.choice(sorted(TRIPLES_BY_HYPOTENUSE))
    pair1, pair2 = rng.sample(TRIPLES_BY_HYPOTENUSE[diagonal], 2)

    ab, bc = pair1
    cd, da = pair2
    if rng.random() < 0.5:
        ab, bc = bc, ab
    if rng.random() < 0.5:
        cd, da = da, cd

    other = Fraction(ab * cd + bc * da, diagonal)
    answer = str(other.numerator) if other.denominator == 1 else f"{other.numerator}/{other.denominator}"
    problem = (
        f"ABCD is a cyclic quadrilateral. Its side lengths are AB={ab}, "
        f"BC={bc}, CD={cd}, and DA={da}. The diagonal AC has length {diagonal}. "
        f"Using Ptolemy's theorem, find the length of the other diagonal BD."
    )
    return problem, answer

CONCEPT_GROUP = "geometry"
CONCEPT_TYPE = "geometry.ptolemy_cyclic_quadrilateral"
