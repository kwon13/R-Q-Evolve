#!/usr/bin/env python3
"""Merge fixed Evolved Performance benchmarks into one immutable benchmark."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import (  # noqa: E402
    SCHEMA_VERSION,
    benchmark_sha256,
    load_benchmark,
)

_LABEL_RE = re.compile(r"^[A-Za-z0-9_-]+$")


def _parse_component(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("component must be LABEL=BENCHMARK_DIR")
    label, raw_path = value.split("=", 1)
    if not _LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "component LABEL may contain only letters, digits, '_' and '-'"
        )
    return label, Path(raw_path)


def _load_components(
    specs: list[tuple[str, Path]],
) -> list[tuple[str, Path, list[dict[str, Any]], dict[str, Any]]]:
    labels: set[str] = set()
    loaded = []
    for label, directory in specs:
        if label in labels:
            raise ValueError(f"duplicate component label: {label}")
        labels.add(label)
        directory = directory.expanduser().resolve()
        rows, manifest = load_benchmark(
            directory / "benchmark.jsonl", directory / "manifest.json"
        )
        loaded.append((label, directory, rows, manifest))
    return loaded


def _merge_rows(
    components: list[tuple[str, Path, list[dict[str, Any]], dict[str, Any]]],
    benchmark_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged: list[dict[str, Any]] = []
    component_records: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for label, directory, rows, manifest in components:
        component_records.append(
            {
                "label": label,
                "benchmark": manifest.get("benchmark"),
                "benchmark_sha256": manifest["benchmark_sha256"],
                "num_examples": len(rows),
                "num_programs": manifest.get("num_programs"),
                "path": str(directory),
            }
        )
        for row in rows:
            signature = str(row["instance_sha256"])
            if signature in signatures:
                raise ValueError(
                    "duplicate problem/answer instance across components: "
                    f"{label}={row['sample_id']} ({signature})"
                )
            signatures.add(signature)
            source_sample_id = str(row["sample_id"])
            source_program_name = str(row["program_name"])
            source_program_id = str(row["program_id"])
            merged.append(
                {
                    **row,
                    "benchmark": benchmark_name,
                    "index": len(merged),
                    "sample_id": f"{label}:{source_sample_id}",
                    "program_name": f"{label}__{source_program_name}",
                    "program_id": f"{label}:{source_program_id}",
                    "source_label": label,
                    "source_benchmark": str(manifest.get("benchmark") or ""),
                    "source_sample_id": source_sample_id,
                    "source_program_name": source_program_name,
                    "source_program_id": source_program_id,
                }
            )
    return merged, component_records


def _write_merged(
    output_dir: Path,
    benchmark_name: str,
    rows: list[dict[str, Any]],
    component_records: list[dict[str, Any]],
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    digest = benchmark_sha256(rows)
    (output_dir / "benchmark.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    program_names = {str(row["program_name"]) for row in rows}
    program_sizes: dict[str, int] = {}
    for row in rows:
        name = str(row["program_name"])
        program_sizes[name] = program_sizes.get(name, 0) + 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "benchmark": benchmark_name,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "benchmark_sha256": digest,
        "num_examples": len(rows),
        "num_programs": len(program_names),
        "examples_per_program": (
            next(iter(set(program_sizes.values())))
            if len(set(program_sizes.values())) == 1
            else None
        ),
        "selection": (
            "immutable union of component benchmark rows; component labels "
            "namespace sample IDs and program identities"
        ),
        "components": component_records,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component",
        action="append",
        type=_parse_component,
        required=True,
        help="component in LABEL=BENCHMARK_DIR form; repeat for every set",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--benchmark-name", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    components = _load_components(args.component)
    rows, component_records = _merge_rows(components, args.benchmark_name)
    output_dir = args.output_dir.expanduser().resolve()
    existing_manifest_path = output_dir / "manifest.json"
    existing_rows_path = output_dir / "benchmark.jsonl"
    if existing_manifest_path.is_file() and existing_rows_path.is_file() and not args.force:
        existing_rows, existing = load_benchmark(
            existing_rows_path, existing_manifest_path
        )
        expected_hash = benchmark_sha256(rows)
        if (
            existing.get("benchmark") != args.benchmark_name
            or existing.get("benchmark_sha256") != expected_hash
            or existing.get("components") != component_records
        ):
            raise ValueError(
                f"stale or different merged benchmark at {output_dir}; use --force"
            )
        print(
            f"[EPB] reusing {len(existing_rows)} merged examples; "
            f"sha256={existing['benchmark_sha256']}"
        )
        return

    manifest = _write_merged(
        output_dir, args.benchmark_name, rows, component_records
    )
    # Validate the final artifact through the same path used by evaluation.
    load_benchmark(output_dir / "benchmark.jsonl", output_dir / "manifest.json")
    print(
        f"[EPB] wrote {len(rows)} examples across "
        f"{manifest['num_programs']} programs; "
        f"sha256={manifest['benchmark_sha256']}"
    )


if __name__ == "__main__":
    main()
