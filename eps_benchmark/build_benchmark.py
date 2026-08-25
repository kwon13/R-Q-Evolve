#!/usr/bin/env python3
"""Build the fixed 6-GROUP, 480-instance EPS benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from artifact_integrity import (
    EXPECTED_ARTIFACT_SHA256,
    EXPECTED_BENCHMARK_SHA256,
    LOCKED_ROW_FIELDS,
    artifact_sha256,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
PROGRAM_DIR = HERE / "programs"
BENCHMARK_NAME = "evolved_performance_group_balanced_ood_v2"
GROUPS = (
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
)
PROGRAMS_PER_GROUP = 2
EXAMPLES_PER_PROGRAM = 40
SEED_START = 5_000_000
CREATED_AT = "2026-08-24T00:00:00+09:00"


def find_rq_root() -> Path:
    """Locate the nested R-Q-Evolve checkout without relying on Unicode form."""

    candidates = [WORKSPACE / "수식 증명" / "R-Q-Evolve"]
    candidates.extend(path / "R-Q-Evolve" for path in WORKSPACE.iterdir())
    for candidate in candidates:
        if (candidate / "src" / "rq_evolve" / "evolved_performance.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the nested R-Q-Evolve repository")


RQ_ROOT = find_rq_root()
sys.path.insert(0, str(RQ_ROOT / "src"))

from rq_evolve.ast_contract import check_generator_contract  # noqa: E402
from rq_evolve.code_utils import lint_generator_source  # noqa: E402
from rq_evolve.concepts import validate_label_decl  # noqa: E402
from rq_evolve.evolved_performance import (  # noqa: E402
    build_seed_id_rows,
    load_benchmark,
    write_benchmark,
)
from rq_evolve.program import ProblemProgram  # noqa: E402


def source_inventory(directory: Path) -> list[dict]:
    inventory = []
    for path in sorted(directory.glob("*.py")):
        source = path.read_bytes()
        program = ProblemProgram.from_file(path)
        inventory.append(
            {
                "program_name": path.stem,
                "program_id": program.program_id,
                "program_sha256": hashlib.sha256(source).hexdigest(),
                "group": program.declared_group(),
                "skill": program.declared_skill(),
            }
        )
    return inventory


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_provenance() -> dict:
    """Record the exact local machinery used to materialize the rows."""

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=RQ_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    git_head = completed.stdout.strip() if completed.returncode == 0 else "unavailable"
    module_paths = (
        "src/rq_evolve/ast_contract.py",
        "src/rq_evolve/code_utils.py",
        "src/rq_evolve/concepts.py",
        "src/rq_evolve/evolved_performance.py",
        "src/rq_evolve/program.py",
    )
    return {
        "python_version": platform.python_version(),
        "rq_evolve_git_head": git_head,
        "artifact_integrity_sha256": _file_sha256(
            HERE / "artifact_integrity.py"
        ),
        "builder_sha256": _file_sha256(HERE / "build_benchmark.py"),
        "validator_sha256": _file_sha256(HERE / "validate_benchmark.py"),
        "rq_evolve_module_sha256": {
            relative: _file_sha256(RQ_ROOT / relative)
            for relative in module_paths
        },
    }


def validate_program_sources() -> list[dict]:
    paths = sorted(PROGRAM_DIR.glob("*.py"))
    expected = len(GROUPS) * PROGRAMS_PER_GROUP
    if len(paths) != expected:
        raise ValueError(
            f"expected exactly {expected} generator files, found {len(paths)}"
        )

    inventory = []
    group_counts: Counter[str] = Counter()
    for path in paths:
        source = path.read_text(encoding="utf-8")
        program = ProblemProgram.from_file(path)
        errors = list(lint_generator_source(source))
        errors.extend(str(item) for item in check_generator_contract(source))
        errors.extend(
            validate_label_decl(
                program.declared_group(),
                program.declared_skill(),
            )
        )
        if errors:
            joined = "; ".join(errors)
            raise ValueError(f"invalid generator {path.name}: {joined}")
        group = str(program.declared_group())
        group_counts[group] += 1
        inventory.append(
            {
                "program_name": path.stem,
                "program_id": program.program_id,
                "program_sha256": hashlib.sha256(
                    path.read_bytes()
                ).hexdigest(),
                "group": group,
                "skill": program.declared_skill(),
            }
        )

    expected_counts = {group: PROGRAMS_PER_GROUP for group in GROUPS}
    if dict(sorted(group_counts.items())) != dict(sorted(expected_counts.items())):
        raise ValueError(
            f"generator GROUP imbalance: {dict(group_counts)}; "
            f"expected {expected_counts}"
        )
    return inventory


def build(*, force: bool) -> None:
    benchmark_path = HERE / "benchmark.jsonl"
    manifest_path = HERE / "manifest.json"
    if (benchmark_path.exists() or manifest_path.exists()) and not force:
        raise FileExistsError(
            "fixed benchmark already exists; pass --force only for an "
            "intentional deterministic rebuild"
        )

    generator_inventory = validate_program_sources()
    rows, programs = build_seed_id_rows(
        PROGRAM_DIR,
        examples_per_program=EXAMPLES_PER_PROGRAM,
        seed_start=SEED_START,
        max_seed_scan=100_000,
        benchmark_name=BENCHMARK_NAME,
    )

    group_counts = Counter(str(row["group"]) for row in rows)
    program_counts = Counter(str(row["program_name"]) for row in rows)
    expected_group_counts = {group: 80 for group in GROUPS}
    if dict(sorted(group_counts.items())) != dict(sorted(expected_group_counts.items())):
        raise AssertionError(f"unexpected GROUP counts: {dict(group_counts)}")
    if set(program_counts.values()) != {EXAMPLES_PER_PROGRAM}:
        raise AssertionError(f"unexpected per-program counts: {dict(program_counts)}")

    current_seed_inventory = source_inventory(RQ_ROOT / "seed_programs")
    old_challenge_inventory = source_inventory(
        RQ_ROOT / "challenge_seed_programs" / "structural_ood_v2"
    )
    with tempfile.TemporaryDirectory(prefix=".eps-build-", dir=HERE) as temp_name:
        staging_dir = Path(temp_name)
        _, _, manifest = write_benchmark(
            staging_dir,
            rows,
            programs,
            seed_start=SEED_START,
            examples_per_program=EXAMPLES_PER_PROGRAM,
            created_at=CREATED_AT,
            benchmark_name=BENCHMARK_NAME,
        )
        manifest.update({
            "artifact_schema": "group_balanced_ood_v1",
            "artifact_sha256": artifact_sha256(rows),
            "locked_row_fields": list(LOCKED_ROW_FIELDS),
            "balance": {
                "groups": list(GROUPS),
                "programs_per_group": PROGRAMS_PER_GROUP,
                "examples_per_program": EXAMPLES_PER_PROGRAM,
                "examples_per_group": 80,
                "group_counts": dict(sorted(group_counts.items())),
            },
            "score_equivalence": (
                "Because all 12 programs have 40 rows and every GROUP has two "
                "programs, macro-program accuracy = macro-GROUP accuracy = "
                "micro accuracy."
            ),
            "distribution": (
                "Evaluation-only generator sources distinct from the eight "
                "current seed sources; no Seed-ID half."
            ),
            "generator_inventory": generator_inventory,
            "reference_seed_program_count": len(current_seed_inventory),
            "reference_seed_inventory": current_seed_inventory,
            "legacy_challenge_program_count": len(old_challenge_inventory),
            "legacy_challenge_inventory": old_challenge_inventory,
            "provenance": build_provenance(),
            "selection": (
                "For each of 12 held-out programs, take the first 40 globally "
                "unique (normalized problem, integer answer) instances from "
                "sequential seeds starting at 5000000."
            ),
        })
        staged_manifest_path = staging_dir / "manifest.json"
        staged_benchmark_path = staging_dir / "benchmark.jsonl"
        staged_manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        staged_rows, staged_manifest = load_benchmark(
            staged_benchmark_path,
            staged_manifest_path,
        )
        if staged_rows != rows or staged_manifest != manifest:
            raise AssertionError("staged benchmark failed exact reload validation")
        if manifest["benchmark_sha256"] != EXPECTED_BENCHMARK_SHA256:
            raise ValueError(
                "pinned legacy digest changed; create a new benchmark version "
                "instead of overwriting this artifact"
            )
        if manifest["artifact_sha256"] != EXPECTED_ARTIFACT_SHA256:
            raise ValueError(
                "pinned label-sensitive digest changed; create a new benchmark "
                "version instead of overwriting this artifact"
            )

        # Both complete files are staged before either destination is touched.
        # Each replacement is atomic; the manifest is installed last so a
        # crash between replacements is detected by every existing loader.
        os.replace(staged_benchmark_path, benchmark_path)
        os.replace(staged_manifest_path, manifest_path)
    print(
        f"wrote {len(rows)} rows; legacy_sha256={manifest['benchmark_sha256']}; "
        f"artifact_sha256={manifest['artifact_sha256']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace the fixed JSONL and manifest with a deterministic rebuild",
    )
    args = parser.parse_args()
    build(force=args.force)


if __name__ == "__main__":
    main()
