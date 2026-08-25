#!/usr/bin/env python3
"""Strictly validate the fixed group-balanced EPS artifact and its sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import Counter
from datetime import datetime, timezone
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
EXPECTED_BENCHMARK = "evolved_performance_group_balanced_ood_v2"
EXPECTED_GROUPS = {
    "number_theory",
    "combinatorics",
    "sequence",
    "algebra",
    "geometry",
    "inequality",
}
INTEGER_RE = re.compile(r"-?(?:0|[1-9]\d*)\Z")


def find_rq_root() -> Path:
    candidates = [WORKSPACE / "수식 증명" / "R-Q-Evolve"]
    candidates.extend(path / "R-Q-Evolve" for path in WORKSPACE.iterdir())
    for candidate in candidates:
        if (candidate / "src" / "rq_evolve" / "evolved_performance.py").is_file():
            return candidate.resolve()
    raise FileNotFoundError("could not locate the nested R-Q-Evolve repository")


RQ_ROOT = find_rq_root()
sys.path.insert(0, str(RQ_ROOT / "src"))

from rq_evolve.ast_contract import (  # noqa: E402
    check_generator_contract,
    check_problem_text,
)
from rq_evolve.code_utils import (  # noqa: E402
    lint_generator_source,
    lint_problem_instance,
)
from rq_evolve.concepts import validate_label_decl  # noqa: E402
from rq_evolve.evolved_performance import (  # noqa: E402
    build_seed_id_rows,
    load_benchmark,
    normalize_problem,
)
from rq_evolve.program import ProblemProgram  # noqa: E402

from build_benchmark import (  # noqa: E402
    BENCHMARK_NAME,
    EXAMPLES_PER_PROGRAM,
    SEED_START,
    build_provenance,
    source_inventory,
)


def fail(message: str) -> None:
    raise AssertionError(message)


def normalized_numeric_template(problem: str) -> str:
    """Normalize wording while masking every numerical parameter."""

    return re.sub(r"\d+", "#", normalize_problem(problem).casefold())


def validate_sources(rows: list[dict], manifest: dict) -> dict[str, ProblemProgram]:
    paths = sorted(PROGRAM_DIR.glob("*.py"))
    if len(paths) != 12:
        fail(f"expected 12 generator sources, found {len(paths)}")

    row_by_program: dict[str, list[dict]] = {}
    for row in rows:
        row_by_program.setdefault(str(row["program_name"]), []).append(row)

    programs: dict[str, ProblemProgram] = {}
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
            fail(f"{path.name} source contract failed: {'; '.join(errors)}")
        if path.stem not in row_by_program:
            fail(f"source {path.name} has no benchmark rows")

        source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in row_by_program[path.stem]:
            if row["program_id"] != program.program_id:
                fail(f"program_id mismatch for {row['sample_id']}")
            if row["program_sha256"] != source_hash:
                fail(f"program_sha256 mismatch for {row['sample_id']}")
            if row["group"] != program.declared_group():
                fail(f"GROUP mismatch for {row['sample_id']}")
            if row["skill"] != program.declared_skill():
                fail(f"SKILL mismatch for {row['sample_id']}")
        programs[path.stem] = program

    declared = {
        item["program_name"]: item
        for item in manifest.get("generator_inventory", [])
    }
    if set(declared) != set(programs):
        fail("manifest generator_inventory does not match source directory")

    actual_inventory = {
        item["program_name"]: item
        for item in source_inventory(PROGRAM_DIR)
    }
    if declared != actual_inventory:
        fail("manifest generator_inventory values do not match generator sources")

    row_program_records = []
    for name in sorted(row_by_program):
        program_rows = row_by_program[name]
        first = program_rows[0]
        row_program_records.append(
            {
                "program_name": name,
                "program_id": first["program_id"],
                "program_sha256": first["program_sha256"],
                "group": first["group"],
                "skill": first["skill"],
                "num_examples": len(program_rows),
                "seeds_scanned": (
                    max(int(row["seed"]) for row in program_rows)
                    - int(manifest["seed_start"])
                    + 1
                ),
            }
        )
    if manifest.get("programs") != row_program_records:
        fail("manifest programs records do not match benchmark rows/sources")
    return programs


def validate_rows(
    rows: list[dict],
    manifest: dict,
    programs: dict[str, ProblemProgram],
) -> dict:
    if manifest.get("benchmark") != EXPECTED_BENCHMARK:
        fail(f"unexpected benchmark name: {manifest.get('benchmark')!r}")
    if manifest.get("benchmark_sha256") != EXPECTED_BENCHMARK_SHA256:
        fail("legacy benchmark digest differs from the immutable v1 lock")
    if manifest.get("artifact_sha256") != EXPECTED_ARTIFACT_SHA256:
        fail("label-sensitive artifact digest differs from the immutable v1 lock")
    if len(rows) != 480:
        fail(f"expected 480 rows, found {len(rows)}")
    if int(manifest.get("num_programs", -1)) != 12:
        fail("manifest num_programs must be 12")
    if int(manifest.get("examples_per_program", -1)) != 40:
        fail("manifest examples_per_program must be 40")
    if manifest.get("artifact_sha256") != artifact_sha256(rows):
        fail("artifact_sha256 mismatch (labels/indices/content are not locked)")
    if manifest.get("locked_row_fields") != list(LOCKED_ROW_FIELDS):
        fail("manifest locked_row_fields mismatch")

    group_counts = Counter(str(row.get("group")) for row in rows)
    program_counts = Counter(str(row.get("program_name")) for row in rows)
    if set(group_counts) != EXPECTED_GROUPS:
        fail(f"unexpected GROUP set: {set(group_counts)}")
    if set(group_counts.values()) != {80}:
        fail(f"GROUP counts are not all 80: {dict(group_counts)}")
    if len(program_counts) != 12 or set(program_counts.values()) != {40}:
        fail(f"program counts are not 12 x 40: {dict(program_counts)}")
    programs_by_group = Counter(
        next(iter({str(row["group"]) for row in rows if row["program_name"] == name}))
        for name in program_counts
    )
    if set(programs_by_group.values()) != {2}:
        fail(f"each GROUP must have two programs: {dict(programs_by_group)}")

    expected_balance = {
        "groups": [
            "number_theory",
            "combinatorics",
            "sequence",
            "algebra",
            "geometry",
            "inequality",
        ],
        "programs_per_group": 2,
        "examples_per_program": 40,
        "examples_per_group": 80,
        "group_counts": dict(sorted(group_counts.items())),
    }
    if manifest.get("balance") != expected_balance:
        fail("manifest balance metadata does not match benchmark rows")

    normalized_problems: set[str] = set()
    deterministic_checks = 0
    answer_token_leaks = 0
    for expected_index, row in enumerate(rows):
        missing = [field for field in LOCKED_ROW_FIELDS if field not in row]
        if missing:
            fail(f"row {expected_index} is missing fields: {missing}")
        if row["benchmark"] != EXPECTED_BENCHMARK:
            fail(f"row benchmark mismatch at index {expected_index}")
        if row["index"] != expected_index:
            fail(f"non-contiguous index at row {expected_index}: {row['index']}")
        if not INTEGER_RE.fullmatch(str(row["answer"])):
            fail(f"non-canonical integer answer at {row['sample_id']}")
        answer_pattern = (
            r"(?<![\d.])"
            + re.escape(str(row["answer"]))
            + r"(?![\d.])"
        )
        if re.search(answer_pattern, str(row["problem"])):
            answer_token_leaks += 1
            fail(f"answer token leaked into problem text at {row['sample_id']}")
        normalized = normalize_problem(str(row["problem"]))
        if normalized in normalized_problems:
            fail(f"duplicate normalized problem text at {row['sample_id']}")
        normalized_problems.add(normalized)

        program = programs[str(row["program_name"])]
        instance = program.execute(int(row["seed"]))
        if instance is None:
            fail(
                f"source re-execution failed for {row['sample_id']}: "
                f"{program.last_execution_error}"
            )
        if instance.problem != row["problem"] or instance.answer != row["answer"]:
            fail(f"source re-execution changed {row['sample_id']}")
        instance_errors = list(lint_problem_instance(instance))
        instance_errors.extend(str(item) for item in check_problem_text(instance.problem))
        if instance_errors:
            fail(
                f"problem contract failed for {row['sample_id']}: "
                f"{'; '.join(instance_errors)}"
            )
        deterministic_checks += 1

    return {
        "group_counts": dict(sorted(group_counts.items())),
        "program_counts": dict(sorted(program_counts.items())),
        "unique_normalized_problems": len(normalized_problems),
        "integer_answers": len(rows),
        "answer_token_leaks": answer_token_leaks,
        "source_reexecutions": deterministic_checks,
    }


def validate_deterministic_selection(rows: list[dict], manifest: dict) -> int:
    """Rebuild and enforce the documented first-valid sequential selection."""

    rebuilt_rows, rebuilt_programs = build_seed_id_rows(
        PROGRAM_DIR,
        examples_per_program=EXAMPLES_PER_PROGRAM,
        seed_start=SEED_START,
        max_seed_scan=100_000,
        benchmark_name=BENCHMARK_NAME,
    )
    if rebuilt_rows != rows:
        fail("independent deterministic rebuild does not match benchmark rows")
    if rebuilt_programs != manifest.get("programs"):
        fail("independent deterministic rebuild does not match manifest programs")
    return len(rebuilt_rows)


def validate_provenance(manifest: dict) -> dict:
    """Ensure the recorded source machinery still matches this workspace."""

    recorded = manifest.get("provenance")
    if not isinstance(recorded, dict):
        fail("manifest is missing build provenance")
    current = build_provenance()
    locked_keys = (
        "rq_evolve_git_head",
        "artifact_integrity_sha256",
        "builder_sha256",
        "validator_sha256",
        "rq_evolve_module_sha256",
    )
    for key in locked_keys:
        if recorded.get(key) != current.get(key):
            fail(f"manifest provenance is stale for {key}")
    return recorded


def audit_reference_overlap(
    rows: list[dict],
    *,
    scan_per_range: int,
) -> dict:
    """Audit exact text/source reuse against current seeds and old OOD v2.

    This is deliberately described as an exact audit. It cannot prove semantic
    disjointness; the README supplies the manual structural rationale.
    """

    benchmark_problems = {normalize_problem(str(row["problem"])) for row in rows}
    benchmark_templates = {
        normalized_numeric_template(str(row["problem"])) for row in rows
    }
    benchmark_source_hashes = {str(row["program_sha256"]) for row in rows}
    benchmark_program_ids = {str(row["program_id"]) for row in rows}
    directories = {
        "current_seed_programs": RQ_ROOT / "seed_programs",
        "legacy_structural_ood_v2": (
            RQ_ROOT / "challenge_seed_programs" / "structural_ood_v2"
        ),
    }
    result = {}
    exact_overlaps: set[str] = set()
    source_reuse = []
    for label, directory in directories.items():
        generated: set[str] = set()
        generated_templates: set[str] = set()
        failures = 0
        paths = sorted(directory.glob("*.py"))
        for path in paths:
            source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            program = ProblemProgram.from_file(path)
            if source_hash in benchmark_source_hashes or program.program_id in benchmark_program_ids:
                source_reuse.append(path.name)
            seeds = list(range(scan_per_range)) + list(
                range(5_000_000, 5_000_000 + scan_per_range)
            )
            for seed in seeds:
                instance = program.execute(seed)
                if instance is None:
                    failures += 1
                    continue
                generated.add(normalize_problem(instance.problem))
                generated_templates.add(
                    normalized_numeric_template(instance.problem)
                )
        overlap = generated & benchmark_problems
        template_overlap = generated_templates & benchmark_templates
        exact_overlaps.update(overlap)
        result[label] = {
            "source_programs": len(paths),
            "seed_ranges": [
                [0, max(scan_per_range - 1, 0)],
                [5_000_000, 5_000_000 + max(scan_per_range - 1, 0)],
            ],
            "executed_instances": len(paths) * scan_per_range * 2 - failures,
            "unique_normalized_problems": len(generated),
            "execution_failures": failures,
            "exact_problem_text_overlaps": len(overlap),
            "numeric_masked_template_overlaps": len(template_overlap),
        }
        if template_overlap:
            fail(
                f"reference numeric-masked template overlap detected for "
                f"{label}: {len(template_overlap)}"
            )

    current_inventory = source_inventory(directories["current_seed_programs"])
    legacy_inventory = source_inventory(directories["legacy_structural_ood_v2"])
    if len(current_inventory) != 8:
        fail(
            "the live current-seed inventory changed: expected 8 sources, "
            f"found {len(current_inventory)}; rebuild/re-audit the benchmark"
        )
    # The manifest is read in main; inventories are returned for an exact check
    # there without weakening this function's overlap-only return structure.
    if source_reuse:
        fail(f"reference generator source/program reuse detected: {source_reuse}")
    if exact_overlaps:
        fail(f"reference exact problem-text overlap detected: {len(exact_overlaps)}")
    return {
        "source_or_program_id_reuse": 0,
        "exact_problem_text_overlaps": 0,
        "corpora": result,
        "current_seed_inventory": current_inventory,
        "legacy_challenge_inventory": legacy_inventory,
        "limitation": (
            "A zero exact-overlap count is not proof of semantic independence; "
            "the family-level structural audit is documented in README.md."
        ),
    }


def audit_committed_benchmark_overlap(rows: list[dict]) -> dict:
    """Reject exact or numeric-template reuse from committed benchmark files."""

    benchmark_problems = {normalize_problem(str(row["problem"])) for row in rows}
    benchmark_templates = {
        normalized_numeric_template(str(row["problem"])) for row in rows
    }
    exact_overlaps: set[str] = set()
    template_overlaps: set[str] = set()
    files_scanned = 0
    rows_scanned = 0

    for path in sorted((RQ_ROOT / "benchmarks").rglob("benchmark.jsonl")):
        files_scanned += 1
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if "problem" not in item:
                continue
            rows_scanned += 1
            normalized = normalize_problem(str(item["problem"]))
            template = normalized_numeric_template(str(item["problem"]))
            if normalized in benchmark_problems:
                exact_overlaps.add(normalized)
            if template in benchmark_templates:
                template_overlaps.add(template)

    if exact_overlaps:
        fail(
            f"committed benchmark exact problem overlap detected: "
            f"{len(exact_overlaps)}"
        )
    if template_overlaps:
        fail(
            f"committed benchmark numeric-masked template overlap detected: "
            f"{len(template_overlaps)}"
        )
    return {
        "benchmark_files_scanned": files_scanned,
        "rows_scanned": rows_scanned,
        "exact_problem_text_overlaps": 0,
        "numeric_masked_template_overlaps": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--overlap-seeds-per-range",
        type=int,
        default=80,
        help="seeds scanned in both a low and benchmark-adjacent range per reference source",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="validate without rewriting validation_report.json",
    )
    args = parser.parse_args()
    if args.overlap_seeds_per_range < 1:
        parser.error("--overlap-seeds-per-range must be positive")

    rows, manifest = load_benchmark(HERE / "benchmark.jsonl", HERE / "manifest.json")
    programs = validate_sources(rows, manifest)
    row_report = validate_rows(rows, manifest, programs)
    rebuilt_rows = validate_deterministic_selection(rows, manifest)
    build_environment = validate_provenance(manifest)
    overlap_report = audit_reference_overlap(
        rows,
        scan_per_range=args.overlap_seeds_per_range,
    )
    committed_benchmark_overlap = audit_committed_benchmark_overlap(rows)
    if manifest.get("reference_seed_program_count") != len(
        overlap_report["current_seed_inventory"]
    ):
        fail("manifest reference_seed_program_count is stale")
    if manifest.get("reference_seed_inventory") != overlap_report[
        "current_seed_inventory"
    ]:
        fail("manifest reference_seed_inventory is stale")
    if manifest.get("legacy_challenge_program_count") != len(
        overlap_report["legacy_challenge_inventory"]
    ):
        fail("manifest legacy_challenge_program_count is stale")
    if manifest.get("legacy_challenge_inventory") != overlap_report[
        "legacy_challenge_inventory"
    ]:
        fail("manifest legacy_challenge_inventory is stale")

    # Inventory details are already locked in the manifest; keep the report
    # compact and focused on the executed overlap audit.
    overlap_report.pop("current_seed_inventory")
    overlap_report.pop("legacy_challenge_inventory")
    report = {
        "status": "pass",
        "validated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "validator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation_python_version": platform.python_version(),
        "build_environment": build_environment,
        "benchmark": manifest["benchmark"],
        "benchmark_sha256": manifest["benchmark_sha256"],
        "artifact_sha256": manifest["artifact_sha256"],
        "num_examples": len(rows),
        "num_programs": len(programs),
        "deterministic_rebuild_rows": rebuilt_rows,
        **row_report,
        "reference_overlap_audit": overlap_report,
        "committed_benchmark_overlap_audit": committed_benchmark_overlap,
    }
    if not args.no_report:
        (HERE / "validation_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
