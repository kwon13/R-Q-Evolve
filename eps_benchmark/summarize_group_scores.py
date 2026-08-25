#!/usr/bin/env python3
"""Aggregate checkpoint EPS summaries into six equally weighted GROUP scores."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from artifact_integrity import artifact_sha256


HERE = Path(__file__).resolve().parent
EXPECTED_BENCHMARK = "evolved_performance_group_balanced_ood_v2"
GROUP_ORDER = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
)

def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=0.0, abs_tol=1e-12)


def _read_details(path: Path, expected_artifact_hash: str) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path}; details are required to verify the label-sensitive "
            "artifact hash"
        )
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(rows) != 480:
        raise ValueError(f"{path} contains {len(rows)} rows, not 480")
    for index, row in enumerate(rows):
        if type(row.get("correct")) is not bool:
            raise ValueError(f"{path} row {index} has non-boolean correct value")
    actual_hash = artifact_sha256(rows)
    if actual_hash != expected_artifact_hash:
        raise ValueError(
            f"{path} has artifact hash {actual_hash}, not the locked "
            f"label-sensitive hash {expected_artifact_hash}"
        )
    return rows


def checkpoint_rows(results_dir: Path) -> tuple[list[dict], str, str]:
    paths = sorted(
        results_dir.glob("global_step_*/summary.json"),
        key=lambda path: int(path.parent.name.removeprefix("global_step_")),
    )
    if not paths:
        raise FileNotFoundError(f"no global_step_*/summary.json under {results_dir}")

    local_manifest = json.loads(
        (HERE / "manifest.json").read_text(encoding="utf-8")
    )
    expected_hash = str(local_manifest["benchmark_sha256"])
    expected_artifact_hash = str(local_manifest["artifact_sha256"])
    expected_programs = {
        str(item["program_name"]): item
        for item in local_manifest["programs"]
    }
    benchmark_hash = ""
    rows = []
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("benchmark") != EXPECTED_BENCHMARK:
            raise ValueError(
                f"{path} uses {summary.get('benchmark')!r}, not {EXPECTED_BENCHMARK!r}"
            )
        current_hash = str(summary.get("benchmark_sha256", ""))
        if current_hash != expected_hash:
            raise ValueError(
                f"{path} uses benchmark hash {current_hash!r}, not the locked "
                f"local artifact hash {expected_hash!r}"
            )
        if not benchmark_hash:
            benchmark_hash = current_hash
        elif current_hash != benchmark_hash:
            raise ValueError("checkpoint summaries mix different benchmark hashes")
        if type(summary.get("num_examples")) is not int or summary["num_examples"] != 480:
            raise ValueError(f"{path} was not evaluated on 480 examples")

        step_from_path = int(path.parent.name.removeprefix("global_step_"))
        if type(summary.get("global_step")) is not int or summary["global_step"] != step_from_path:
            raise ValueError(f"global_step does not match directory name in {path}")
        details = _read_details(
            path.parent / "details.jsonl",
            expected_artifact_hash,
        )
        detail_by_program: dict[str, list[dict]] = defaultdict(list)
        for row in details:
            detail_by_program[str(row["program_name"])].append(row)

        grouped: dict[str, list[float]] = defaultdict(list)
        per_program = summary.get("per_program", {})
        if set(per_program) != set(expected_programs):
            raise ValueError(f"{path} has the wrong 12-program inventory")
        for name, item in per_program.items():
            expected = expected_programs[name]
            if item.get("group") != expected["group"]:
                raise ValueError(f"GROUP mismatch for {name} in {path}")
            if item.get("skill") != expected["skill"]:
                raise ValueError(f"SKILL mismatch for {name} in {path}")
            if type(item.get("num_examples")) is not int or item["num_examples"] != 40:
                raise ValueError(f"{name} does not have 40 examples in {path}")
            if type(item.get("correct")) is not int:
                raise ValueError(f"invalid correct count type for {name} in {path}")
            correct = item["correct"]
            if not 0 <= correct <= 40:
                raise ValueError(f"invalid correct count for {name} in {path}")
            if type(item.get("accuracy")) not in {int, float}:
                raise ValueError(f"invalid accuracy type for {name} in {path}")
            accuracy = float(item["accuracy"])
            if not math.isfinite(accuracy) or not _close(accuracy, correct / 40):
                raise ValueError(f"invalid accuracy for {name} in {path}")
            detail_rows = detail_by_program.get(name, [])
            detail_correct = sum(row["correct"] for row in detail_rows)
            if len(detail_rows) != 40 or detail_correct != correct:
                raise ValueError(f"details disagree with summary for {name} in {path}")
            if any(
                row.get("group") != expected["group"]
                or row.get("skill") != expected["skill"]
                for row in detail_rows
            ):
                raise ValueError(f"details labels disagree for {name} in {path}")
            grouped[str(item["group"])].append(float(item["accuracy"]))
        if set(grouped) != set(GROUP_ORDER):
            raise ValueError(f"{path} has unexpected GROUPs: {sorted(grouped)}")
        if any(len(values) != 2 for values in grouped.values()):
            raise ValueError(f"{path} does not have two programs per GROUP")

        group_percent = {
            group: 100.0 * sum(grouped[group]) / len(grouped[group])
            for group in GROUP_ORDER
        }
        group_macro = sum(group_percent.values()) / len(GROUP_ORDER)
        if type(summary.get("score_percent")) not in {int, float}:
            raise ValueError(f"invalid EPS type in {path}")
        eps = float(summary["score_percent"])
        if not math.isfinite(eps) or not 0.0 <= eps <= 100.0:
            raise ValueError(f"invalid EPS in {path}")
        if not _close(group_macro, eps):
            raise ValueError(
                f"GROUP macro {group_macro} disagrees with EPS {eps} in {path}"
            )
        if type(summary.get("micro_accuracy")) not in {int, float}:
            raise ValueError(f"invalid micro accuracy type in {path}")
        micro = 100.0 * float(summary["micro_accuracy"])
        if not _close(micro, eps):
            raise ValueError(f"micro score {micro} disagrees with EPS {eps} in {path}")
        total_correct = sum(int(item["correct"]) for item in per_program.values())
        if type(summary.get("correct")) is not int or summary["correct"] != total_correct:
            raise ValueError(f"total correct count disagrees in {path}")
        rows.append(
            {
                "global_step": step_from_path,
                "eps_percent": eps,
                "micro_percent": micro,
                "group_percent": group_percent,
            }
        )
    return rows, benchmark_hash, expected_artifact_hash


def write_outputs(
    results_dir: Path,
    rows: list[dict],
    benchmark_hash: str,
    label_sensitive_hash: str,
) -> None:
    payload = {
        "benchmark": EXPECTED_BENCHMARK,
        "benchmark_sha256": benchmark_hash,
        "artifact_sha256": label_sensitive_hash,
        "score_definition": (
            "equal macro over six GROUPs; equal to program macro and micro "
            "because the fixed artifact has 2 programs x 40 rows per GROUP"
        ),
        "checkpoints": rows,
    }
    (results_dir / "group_scores.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    header = ["Step", "EPS", *GROUP_ORDER]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---:"] * len(header)) + " |",
    ]
    for row in rows:
        values = [
            str(row["global_step"]),
            f"{row['eps_percent']:.2f}",
            *(f"{row['group_percent'][group]:.2f}" for group in GROUP_ORDER),
        ]
        lines.append("| " + " | ".join(values) + " |")
    (results_dir / "group_scores.md").write_text(
        "# Group-balanced EPS\n\n" + "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", type=Path)
    args = parser.parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    rows, benchmark_hash, label_sensitive_hash = checkpoint_rows(results_dir)
    write_outputs(results_dir, rows, benchmark_hash, label_sensitive_hash)
    print(f"wrote {results_dir / 'group_scores.json'}")
    print(f"wrote {results_dir / 'group_scores.md'}")


if __name__ == "__main__":
    main()
