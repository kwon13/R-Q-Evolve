#!/usr/bin/env python3
"""Build/reuse a fixed generator-based Evolved Performance benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import (  # noqa: E402
    audit_known_seed_overlap,
    build_seed_id_rows,
    load_benchmark,
    write_benchmark,
)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-dir", type=Path, default=ROOT / "seed_programs")
    parser.add_argument(
        "--benchmark-name",
        default="evolved_performance_seed_id_v1",
        help="logical benchmark name stored in every row and the manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "benchmarks" / "evolved_performance_seed_id_v1",
    )
    parser.add_argument("--examples-per-program", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=1_000_000)
    parser.add_argument("--max-seed-scan", type=int, default=100_000)
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace an existing benchmark (changes require explicit intent)",
    )
    parser.add_argument(
        "--used-seeds-json",
        type=Path,
        help="optional rq_used_seeds.json for a retrospective exact-overlap audit",
    )
    parser.add_argument(
        "--overlap-audit-output",
        type=Path,
        help="where to write the optional overlap audit JSON",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    jsonl_path = args.output_dir / "benchmark.jsonl"
    manifest_path = args.output_dir / "manifest.json"
    if jsonl_path.is_file() and manifest_path.is_file() and not args.force:
        rows, manifest = load_benchmark(jsonl_path, manifest_path)
        if manifest.get("benchmark") != args.benchmark_name:
            raise ValueError(
                f"existing benchmark is {manifest.get('benchmark')!r}, not "
                f"{args.benchmark_name!r}; use another --output-dir or --force"
            )
        print(
            f"[EPB] reusing {len(rows)} examples; "
            f"sha256={manifest['benchmark_sha256']}"
        )
    else:
        rows, programs = build_seed_id_rows(
            args.seed_dir,
            examples_per_program=args.examples_per_program,
            seed_start=args.seed_start,
            max_seed_scan=args.max_seed_scan,
            benchmark_name=args.benchmark_name,
        )
        jsonl_path, manifest_path, manifest = write_benchmark(
            args.output_dir,
            rows,
            programs,
            seed_start=args.seed_start,
            examples_per_program=args.examples_per_program,
            created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            benchmark_name=args.benchmark_name,
        )
        print(f"[EPB] wrote {jsonl_path}")
        print(f"[EPB] wrote {manifest_path}")
        print(
            f"[EPB] {len(rows)} examples across {len(programs)} programs; "
            f"sha256={manifest['benchmark_sha256']}"
        )

    if args.used_seeds_json:
        audit = audit_known_seed_overlap(rows, args.seed_dir, args.used_seeds_json)
        audit_path = args.overlap_audit_output
        if audit_path is None:
            audit_path = args.output_dir / "overlap_audit.json"
        audit_path = audit_path.expanduser().resolve()
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(
            f"[EPB] known training overlap: {audit['exact_overlap_count']}/"
            f"{audit['benchmark_examples']} "
            f"({100.0 * audit['exact_overlap_rate']:.2f}%)"
        )
        print(f"[EPB] wrote {audit_path}")


if __name__ == "__main__":
    main()
