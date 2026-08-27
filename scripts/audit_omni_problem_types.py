#!/usr/bin/env python3
"""Pilot-audit Omni-MATH for a Domain x Computational-Problem-Type MAP.

This is a conservative surface annotator, not a gold-label generator.  Rows
whose requested output is not explicit enough remain in ``needs_review``.

Example:
    python scripts/audit_omni_problem_types.py \
      --input /path/to/Omni-Math.jsonl \
      --output-dir analysis/omni_problem_types/full
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rq_evolve.problem_type import (  # noqa: E402
    PROBLEM_TYPE_RULESET,
    PROBLEM_TYPES,
    annotate_problem_type,
    integer_answer,
    problem_type_ruleset_sha256,
    top_level_domains,
)


def _entropy(counts: list[int]) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log(count / total)
        for count in counts
        if count
    )


def _association(matrix: list[list[int]]) -> tuple[float, float]:
    """Return (NMI, bias-corrected Cramer's V) for a contingency table."""

    if not matrix or not matrix[0]:
        return 0.0, 0.0
    rows = [sum(row) for row in matrix]
    cols = [sum(matrix[i][j] for i in range(len(matrix))) for j in range(len(matrix[0]))]
    total = sum(rows)
    if total == 0:
        return 0.0, 0.0

    mutual_information = 0.0
    chi2 = 0.0
    for i, row in enumerate(matrix):
        for j, observed in enumerate(row):
            expected = rows[i] * cols[j] / total
            if observed:
                mutual_information += (observed / total) * math.log(
                    observed * total / (rows[i] * cols[j])
                )
            if expected:
                chi2 += (observed - expected) ** 2 / expected

    denominator = math.sqrt(_entropy(rows) * _entropy(cols))
    nmi = mutual_information / denominator if denominator else 0.0

    r, k = len(rows), len(cols)
    phi2 = chi2 / total
    correction = ((k - 1) * (r - 1)) / max(total - 1, 1)
    phi2_corrected = max(0.0, phi2 - correction)
    r_corrected = r - ((r - 1) ** 2) / max(total - 1, 1)
    k_corrected = k - ((k - 1) ** 2) / max(total - 1, 1)
    v_denominator = min(k_corrected - 1, r_corrected - 1)
    cramers_v = math.sqrt(phi2_corrected / v_denominator) if v_denominator > 0 else 0.0
    return nmi, cramers_v


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected object at {path}:{line_no}")
            rows.append(row)
    return rows


def audit(rows: list[dict]) -> tuple[list[dict], dict, list[dict]]:
    annotations: list[dict] = []
    type_counts: Counter[str] = Counter()
    review_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    cell_counts: Counter[tuple[str, str]] = Counter()
    single_domain_cell_counts: Counter[tuple[str, str]] = Counter()
    integer_counts: Counter[str] = Counter()
    label_cardinality: Counter[int] = Counter()

    for index, row in enumerate(rows):
        annotation = annotate_problem_type(str(row.get("problem", "")))
        domains = top_level_domains(row.get("domain"))
        label_cardinality[len(domains)] += 1
        for domain in domains:
            domain_counts[domain] += 1
        if annotation.problem_type is None:
            review_counts[annotation.review_reason or "unspecified"] += 1
        else:
            type_counts[annotation.problem_type] += 1
            if integer_answer(row.get("answer")):
                integer_counts[annotation.problem_type] += 1
            for domain in domains:
                cell_counts[(domain, annotation.problem_type)] += 1
            if len(domains) == 1:
                single_domain_cell_counts[(domains[0], annotation.problem_type)] += 1

        annotations.append(
            {
                "index": index,
                "problem_type": annotation.problem_type,
                "confidence": annotation.confidence,
                "needs_review": annotation.needs_review,
                "review_reason": annotation.review_reason,
                "evidence": annotation.evidence,
                "top_level_domains": list(domains),
                "integer_answer": integer_answer(row.get("answer")),
                "problem": row.get("problem"),
                "answer": row.get("answer"),
                "source": row.get("source"),
            }
        )

    domains = sorted(domain_counts)
    expanded_matrix = [
        [cell_counts[(domain, problem_type)] for problem_type in PROBLEM_TYPES]
        for domain in domains
    ]
    expanded_nmi, expanded_cramers_v = _association(expanded_matrix)
    single_domain_matrix = [
        [
            single_domain_cell_counts[(domain, problem_type)]
            for problem_type in PROBLEM_TYPES
        ]
        for domain in domains
    ]
    single_domain_nmi, single_domain_cramers_v = _association(single_domain_matrix)
    classified = sum(type_counts.values())
    expanded_total = sum(sum(row) for row in expanded_matrix)
    single_domain_total = sum(sum(row) for row in single_domain_matrix)

    cells: list[dict] = []
    for domain in domains:
        for problem_type in PROBLEM_TYPES:
            count = cell_counts[(domain, problem_type)]
            single_domain_count = single_domain_cell_counts[(domain, problem_type)]
            cells.append(
                {
                    "domain": domain,
                    "problem_type": problem_type,
                    "expanded_count": count,
                    "single_domain_count": single_domain_count,
                }
            )

    summary = {
        "problem_type_ruleset": PROBLEM_TYPE_RULESET,
        "problem_type_ruleset_sha256": problem_type_ruleset_sha256(),
        "rows": len(rows),
        "classified_rows": classified,
        "classified_fraction": classified / len(rows) if rows else 0.0,
        "needs_review_rows": len(rows) - classified,
        "problem_type_counts": {key: type_counts[key] for key in PROBLEM_TYPES},
        "problem_type_integer_answer_counts": {
            key: integer_counts[key] for key in PROBLEM_TYPES
        },
        "review_reason_counts": dict(sorted(review_counts.items())),
        "top_level_domain_counts": dict(sorted(domain_counts.items())),
        "domain_label_cardinality": {
            str(key): value for key, value in sorted(label_cardinality.items())
        },
        "expanded_domain_memberships": expanded_total,
        "expanded_domain_type_nmi": expanded_nmi,
        "expanded_domain_type_bias_corrected_cramers_v": expanded_cramers_v,
        "association_warning": (
            "Descriptive only: Omni-MATH domains are multi-label, so expanded "
            "memberships are not independent observations."
        ),
        "single_domain_classified_rows": single_domain_total,
        "single_domain_type_nmi": single_domain_nmi,
        "single_domain_type_bias_corrected_cramers_v": single_domain_cramers_v,
        "possible_cells": len(domains) * len(PROBLEM_TYPES),
    }
    return annotations, summary, cells


def _write_outputs(
    output_dir: Path,
    annotations: list[dict],
    summary: dict,
    cells: list[dict],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "annotations.jsonl").open("w", encoding="utf-8") as handle:
        for row in annotations:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with (output_dir / "domain_type_counts.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "domain",
                "problem_type",
                "expanded_count",
                "single_domain_count",
            ),
        )
        writer.writeheader()
        writer.writerows(cells)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    rows = _read_jsonl(args.input)
    annotations, summary, cells = audit(rows)
    summary["input"] = str(args.input.resolve())
    summary["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    _write_outputs(args.output_dir, annotations, summary, cells)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
