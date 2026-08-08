#!/usr/bin/env python3
"""Show prominent high-R_Q concepts across the evolution timeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.evolved_performance import (  # noqa: E402
    build_prominent_high_rq_rows,
    write_prominent_high_rq_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--rq-quantile", type=float, default=0.90)
    parser.add_argument("--max-rq-events", type=int, default=8)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "evolved_performance"
    )
    rows = build_prominent_high_rq_rows(
        run_dir,
        quantile=args.rq_quantile,
        max_count=args.max_rq_events,
    )
    json_path, markdown_path = write_prominent_high_rq_report(
        output_dir,
        rows,
        quantile=args.rq_quantile,
    )
    for row in rows:
        print(
            f"global {row['emerged_global_step']:>4} "
            f"O{row['outer_iteration']:<3} R_Q={row['rq_score']:8.2f} "
            f"{row['group']}/{row['skill']}"
        )
    print(f"[R_Q] wrote {markdown_path}")
    print(f"[R_Q] wrote {json_path}")


if __name__ == "__main__":
    main()
