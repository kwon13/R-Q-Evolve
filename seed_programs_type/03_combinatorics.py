import random
import math

def generate(seed):
    rng = random.Random(seed)

    n_men = rng.randint(5, 9)
    n_women = rng.randint(4, 8)
    committee = rng.randint(3, 5)

    # Choose a female chair, then choose the remaining committee members.
    # Since the chair is already a woman, we only need at least one man among
    # the remaining committee-1 members.
    remaining_slots = committee - 1

    total_remaining = math.comb(n_men + n_women - 1, remaining_slots)
    all_remaining_women = math.comb(n_women - 1, remaining_slots) if n_women - 1 >= remaining_slots else 0

    answer = n_women * (total_remaining - all_remaining_women)

    problem = (
        f"A club has {n_men} men and {n_women} women. A committee of "
        f"{committee} people is to be formed with at least 1 man and at least "
        f"1 woman. One member of the committee will be chosen as chair, and the "
        f"chair must be a woman. How many ways can the committee and chair be chosen?"
    )

    return problem, str(answer)

CONCEPT_GROUP = "combinatorics"
CONCEPT_TYPE = "combinatorics.committee_count"
