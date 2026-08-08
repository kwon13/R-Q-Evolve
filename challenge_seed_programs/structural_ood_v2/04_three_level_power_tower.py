import math
import random


def generate(seed):
    rng = random.Random(seed)

    base = rng.choice([v for v in range(11, 100) if math.gcd(v, 1000) == 1])
    mid = rng.randint(3, 9)
    top = rng.randint(3, 7)
    height = rng.randint(4, 9)
    # Never materialise mid ** (top ** height). The tower's exponent has on the
    # order of 10**7 digits, and building it takes about a minute against a
    # five-second sandbox. Reducing first is also the whole point of the
    # problem, so the generator does what the solver has to do.
    ladder = top ** height

    # Route 1: one reduction against the Carmichael function of the whole
    # modulus. gcd(base, 1000) = 1, so lambda(1000) = 100 applies directly and
    # no lifting correction is needed.
    answer = pow(base, pow(mid, ladder, 100), 1000)

    # Route 2 (independent): split 1000 = 8 * 125, reduce each part against its
    # own order -- lambda(8) = 2 and lambda(125) = 100 -- and glue the two
    # residues back with the Chinese remainder theorem. The mod-8 half is a
    # parity question the single-modulus route never asks.
    residue_8 = pow(base, pow(mid, ladder, 2), 8)
    residue_125 = pow(base, pow(mid, ladder, 100), 125)
    check = next(
        candidate
        for candidate in range(residue_125, 1000, 125)
        if candidate % 8 == residue_8
    )
    assert answer == check, f"answer={answer} check={check}"

    problem = (
        f"Find the least nonnegative residue of {base}^({mid}^({top}^{height})) "
        "modulo 1000. State only the integer."
    )
    return problem, str(answer)


GROUP = "number_theory"
SKILL = "transformation"
