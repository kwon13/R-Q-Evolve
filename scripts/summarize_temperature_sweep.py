#!/usr/bin/env python
"""Collect the temperature-sweep arms into one comparison table.

Reads each arm's sample_summary.json plus its evolution log and reports the two
things the sweep was run to decide: how much of the GROUP x SKILL grid each
temperature opened, and how often the judge agreed with the label the child
gave itself.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_arm(directory: Path) -> dict | None:
    summary = directory / "sample_summary.json"
    if not summary.exists():
        return None
    payload = json.loads(summary.read_text())
    payload["arm"] = directory.name
    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="rq_output/temp_sweep")
    args = ap.parse_args()

    root = Path(args.root)
    arms = [a for a in (load_arm(d) for d in sorted(root.glob("t*"))) if a]
    if not arms:
        print(f"no finished arms under {root}")
        return 1

    head = (
        f"{'arm':<7}{'temp':>6}{'cells':>7}{'opened':>8}{'reached':>9}"
        f"{'agreed':>8}{'mism':>6}{'closed':>8}{'agree%':>8}{'inserted':>10}{'min':>6}"
    )
    print(head)
    print("-" * len(head))
    for a in arms:
        judge = a.get("judge") or {}
        status = a.get("status_counts") or {}
        rate = judge.get("agreement_rate")
        print(
            f"{a['arm']:<7}{a.get('code_temperature', float('nan')):>6}"
            f"{a.get('cells_after', 0):>7}{len(a.get('cells_opened', [])):>8}"
            f"{judge.get('reached_judge', 0):>9}{judge.get('agreed', 0):>8}"
            f"{judge.get('label_mismatch', 0):>6}{judge.get('failed_closed', 0):>8}"
            f"{(f'{rate:.0%}' if rate is not None else '-'):>8}"
            f"{status.get('inserted', 0):>10}{a.get('elapsed_s', 0) / 60:>6.0f}"
        )

    print("\nper-arm candidate outcomes")
    for a in arms:
        counts = a.get("status_counts") or {}
        ordered = sorted(counts.items(), key=lambda kv: -kv[1])
        print(f"  {a['arm']:<6} " + "  ".join(f"{k}={v}" for k, v in ordered))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
