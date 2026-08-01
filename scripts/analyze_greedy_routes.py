"""Score error routes on greedy records, confirmatory arm kept separate.

The routes in ``PRESPECIFIED`` were fixed by mining the temperature-1.0
rollouts, so testing them on greedy instances is a genuine out-of-sample check.
``EXPLORATORY`` routes are additional guesses scored on the same data; they can
only ever generate a hypothesis for a future run, never confirm one, and are
reported under their own heading so the distinction survives into the writeup.

Significance comes from a permutation null that re-pairs each observed answer
with a different instance. That preserves both the answer distribution and the
route-value distribution while destroying the correspondence between them, which
is exactly the null "the solver's errors have nothing to do with this route".
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

Route = Callable[[dict], int | None]


def _linear_routes() -> dict[str, Route]:
    def guard(fn: Route) -> Route:
        def wrapped(d: dict) -> int | None:
            try:
                return fn(d)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return None

        return wrapped

    def div(total: int, k: int) -> int | None:
        return total // k if k and total % k == 0 else None

    return {
        name: guard(fn)
        for name, fn in {
            # Pre-specified: mined from the sampled rollouts.
            "unweighted_sum_then_divide": lambda d: div(
                d["rhs"][0] + d["rhs"][1], d["aggregate_multiplier"]
            ),
            "plain_sum": lambda d: d["rhs"][0] + d["rhs"][1],
            "undivided_weighted": lambda d: d["combination_weight"] * d["rhs"][0]
            + d["rhs"][1],
            # Exploratory.
            "difference": lambda d: d["rhs"][0] - d["rhs"][1],
            "reverse_difference": lambda d: d["rhs"][1] - d["rhs"][0],
            "rhs0": lambda d: d["rhs"][0],
            "rhs1": lambda d: d["rhs"][1],
            "weighted_diff_divided": lambda d: div(
                d["combination_weight"] * d["rhs"][0] - d["rhs"][1],
                d["aggregate_multiplier"],
            ),
        }.items()
    }


def _modular_routes() -> dict[str, Route]:
    def guard(fn: Route) -> Route:
        def wrapped(d: dict) -> int | None:
            try:
                return fn(d)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                return None

        return wrapped

    return {
        name: guard(fn)
        for name, fn in {
            # Pre-specified.
            "congruences_subtracted": lambda d: (d["rhs"][0] - d["rhs"][1])
            % d["modulus"],
            "plain_sum": lambda d: (d["rhs"][0] + d["rhs"][1]) % d["modulus"],
            "integer_division": lambda d: (
                (d["rhs"][0] + d["rhs"][1]) // d["multiplier"]
            )
            % d["modulus"],
            # Exploratory.
            "multiplied_not_inverted": lambda d: (
                (d["rhs"][0] + d["rhs"][1]) * d["multiplier"]
            )
            % d["modulus"],
            "subtracted_then_inverted": lambda d: (
                (d["rhs"][0] - d["rhs"][1]) * pow(d["multiplier"], -1, d["modulus"])
            )
            % d["modulus"],
            "rhs0": lambda d: d["rhs"][0] % d["modulus"],
            "rhs1": lambda d: d["rhs"][1] % d["modulus"],
        }.items()
    }


PRESPECIFIED = {
    "linear_system_aggregate": (
        "unweighted_sum_then_divide",
        "plain_sum",
        "undivided_weighted",
    ),
    "modular_linear_system_aggregate": (
        "congruences_subtracted",
        "plain_sum",
        "integer_division",
    ),
}

_ONTOPIC = re.compile(
    r"\b(x\s*\+\s*y\s*\+\s*z|equation|system|congruen|modul|coefficient"
    r"|substitut|eliminat)\b",
    re.I,
)
_OFFTOPIC = re.compile(r"feet|images|tuple|profit|ice_cream|dollar|\$\d|percent|%", re.I)


def _as_int(value: Any) -> int | None:
    text = str(value).strip() if value is not None else ""
    return int(text) if re.fullmatch(r"-?\d+", text) else None


def classify(record: dict) -> str:
    if record.get("correct"):
        return "correct"
    answer = _as_int(record.get("predicted_answer"))
    text = str(record.get("response") or "")
    if record.get("predicted_answer") in (None, ""):
        return "no_answer"
    if answer is None:
        return "non_integer"
    if _OFFTOPIC.search(text) and not _ONTOPIC.search(text):
        return "off_topic"
    if not _ONTOPIC.search(text):
        return "unclear"
    return "coherent_wrong"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    random.seed(args.seed)

    rows = [json.loads(line) for line in args.records.open(encoding="utf-8")]
    by_variant: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_variant[(row["generator_family"], row["family_variant"])].append(row)

    for (family, variant), group in sorted(by_variant.items()):
        routes = (
            _linear_routes()
            if family == "linear_system_aggregate"
            else _modular_routes()
        )
        counts = collections.Counter(classify(r) for r in group)
        coherent = [r for r in group if classify(r) == "coherent_wrong"]
        print("=" * 78)
        print(f"{family} / {variant}   n={len(group)}  greedy")
        print(
            f"  correct={counts['correct']}  coherent_wrong={counts['coherent_wrong']}"
            f"  no_answer={counts['no_answer']}  non_integer={counts['non_integer']}"
            f"  off_topic={counts['off_topic']}  unclear={counts['unclear']}"
        )
        if len(coherent) < 5:
            print("  too few coherent errors to test")
            continue

        instances = [r["instance_data"] for r in group]
        observed = [_as_int(r["predicted_answer"]) for r in coherent]
        paired = [(a, r["instance_data"]) for a, r in zip(observed, coherent)]

        def tally(pairs: list[tuple[int, dict]]) -> collections.Counter:
            hits: collections.Counter = collections.Counter()
            for answer, data in pairs:
                for name, route in routes.items():
                    value = route(data)
                    if value is not None and answer == value:
                        hits[name] += 1
            return hits

        real = tally(paired)
        null: dict[str, list[int]] = collections.defaultdict(list)
        for _ in range(args.permutations):
            shuffled = [(a, random.choice(instances)) for a, _ in paired]
            counted = tally(shuffled)
            for name in routes:
                null[name].append(counted.get(name, 0))

        pre = set(PRESPECIFIED[family])
        for heading, names in (
            ("confirmatory (pre-specified from the sampled run)", sorted(pre)),
            ("exploratory (hypothesis-generating only)", sorted(set(routes) - pre)),
        ):
            print(f"  -- {heading}")
            scored = []
            for name in names:
                observed_hits = real.get(name, 0)
                draws = null[name]
                chance = sum(draws) / len(draws)
                p = (sum(1 for v in draws if v >= observed_hits) + 1) / (
                    len(draws) + 1
                )
                scored.append((observed_hits, name, chance, p))
            for observed_hits, name, chance, p in sorted(scored, reverse=True):
                bonf = min(1.0, p * len(names))
                mark = "  <-- survives Bonferroni" if bonf < 0.05 else ""
                print(
                    f"     {name:<26} obs={observed_hits:<4} chance={chance:5.1f}"
                    f"  p={p:.4f}  bonf={bonf:.3f}{mark}"
                )

        distinct = collections.Counter(observed)
        explained = sum(
            1
            for answer, data in paired
            if any(
                route(data) is not None and answer == route(data)
                for route in routes.values()
            )
        )
        print(
            f"  distinct wrong values={len(distinct)}  "
            f"most common={distinct.most_common(3)}"
        )
        print(
            f"  errors matching ANY route: {explained}/{len(paired)} "
            f"({explained / len(paired):.0%})"
        )


if __name__ == "__main__":
    main()
