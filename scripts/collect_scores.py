"""Collect per-step benchmark pass@1 into a markdown table.

Scans <BASE>/global_step_*/eval/<benchmark>/summary.json and writes
<BASE>/scores.md with two tables:
  1. pass@1 (final, including the GPT-4o re-check)
  2. pass@1 (pre-GPT, math_verify only)  + total GPT flips per step

Usage:
  python analysis/collect_scores.py [BASE_DIR]
  (BASE_DIR default: rq_output/rq_evolve_base)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

DEFAULT_BASE = Path(__file__).resolve().parent.parent / "rq_output" / "rq_evolve_base"

# R-Zero/evaluation order (incl. gsm8k).
BENCHES = [
    "math500",
    "gsm8k",
    "amc23",
    "aime24",
    "aime25",
    "minerva_math",
    "olympiadbench",
]


def _steps(base: Path) -> list[int]:
    steps = []
    for d in base.glob("global_step_*"):
        m = re.fullmatch(r"global_step_(\d+)", d.name)
        if m and d.is_dir():
            steps.append(int(m.group(1)))
    return sorted(steps)


def _read(base: Path, step: int, bench: str) -> dict | None:
    p = base / f"global_step_{step}" / "eval" / bench / "summary.json"
    if not p.is_file():
        return None
    try:
        data = json.load(p.open())
    except (json.JSONDecodeError, OSError):
        return None
    return data.get("benchmarks", {}).get(bench)


def _fmt(x: float | None) -> str:
    return f"{x * 100:.2f}" if isinstance(x, (int, float)) else "-"


def _avg(vals: list[float]) -> str:
    return f"{sum(vals) / len(vals) * 100:.2f}" if vals else "-"


def _table(rows: list[list[str]], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def build_markdown(base: Path) -> str:
    steps = _steps(base)
    header = ["step", *BENCHES, "AVG"]

    final_rows: list[list[str]] = []
    pre_rows: list[list[str]] = []
    for step in steps:
        final_cells = [str(step)]
        pre_cells = [str(step)]
        final_vals: list[float] = []
        pre_vals: list[float] = []
        total_flips = 0
        for bench in BENCHES:
            b = _read(base, step, bench)
            if b is None:
                final_cells.append("-")
                pre_cells.append("-")
                continue
            fin = b.get("pass_at_1")
            pre = b.get("pass_at_1_pre_gpt", fin)
            total_flips += int(b.get("gpt_flips", 0) or 0)
            final_cells.append(_fmt(fin))
            pre_cells.append(_fmt(pre))
            if isinstance(fin, (int, float)):
                final_vals.append(fin)
            if isinstance(pre, (int, float)):
                pre_vals.append(pre)
        final_cells.append(_avg(final_vals))
        pre_cells.append(_avg(pre_vals) + (f" (+{total_flips})" if total_flips else ""))
        final_rows.append(final_cells)
        pre_rows.append(pre_cells)

    out = []
    out.append("# R-Q-Evolve eval — pass@1 (%) , R-Zero-aligned grading")
    out.append("")
    out.append(f"Base: `{base}`  |  steps: {', '.join(map(str, steps)) or 'none'}")
    out.append("")
    out.append("## pass@1 (final — includes GPT-4o re-check)")
    out.append("")
    out.append(_table(final_rows, header))
    out.append("")
    out.append("## pass@1 (pre-GPT — math_verify only; AVG cell shows total GPT flips)")
    out.append("")
    out.append(_table(pre_rows, header))
    out.append("")
    return "\n".join(out)


def main() -> None:
    base = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_BASE
    md = build_markdown(base)
    out_path = base / "scores.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\n[collect_scores] wrote {out_path}")


if __name__ == "__main__":
    main()
