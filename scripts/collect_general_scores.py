#!/usr/bin/env python
"""Per-step x benchmark table for the general-domain eval.

Mirrors scripts/collect_scores.py, with two extra columns the math table does
not need. On multiple choice an unparsed answer still scores 1/10 of the time
through R-Zero's random-letter fallback, so accuracy alone cannot be told apart
from guessing; and a response that hit the token cap never emitted an answer at
all. Both rates are reported so a rise in accuracy can be checked against them.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ORDER = ["mmlupro", "supergpqa", "bbeh"]
LABEL = {"mmlupro": "MMLU-Pro", "supergpqa": "SuperGPQA", "bbeh": "BBEH"}


def steps(base: Path) -> list[int]:
    out = []
    for d in base.glob("global_step_*"):
        m = re.fullmatch(r"global_step_(\d+)", d.name)
        if m and d.is_dir():
            out.append(int(m.group(1)))
    return sorted(out)


def main() -> int:
    base = Path(os.environ.get("BASE") or (sys.argv[1] if len(sys.argv) > 1 else "."))
    sub = os.environ.get("OUTDIR_NAME", "eval_general")

    rows: dict[int, dict[str, dict]] = {}
    for step in steps(base):
        for name in ORDER:
            path = base / f"global_step_{step}" / sub / name / "summary.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text())
            entry = payload.get("benchmarks", {}).get(name)
            if entry:
                rows.setdefault(step, {})[name] = entry
    if not rows:
        print(f"no summaries under {base}/global_step_*/{sub}", file=sys.stderr)
        return 1

    present = [n for n in ORDER if any(n in v for v in rows.values())]
    lines = [
        "# R-Q-Evolve general-domain reasoning eval",
        "",
        f"Base: `{base}`  |  steps: {', '.join(str(s) for s in sorted(rows))}",
        "",
        "## accuracy (%) — micro average over sampled questions",
        "",
        "| step | " + " | ".join(LABEL[n] for n in present) + " | AVG |",
        "|---" * (len(present) + 2) + "|",
    ]
    for step in sorted(rows):
        cells, accs = [], []
        for name in present:
            entry = rows[step].get(name)
            if not entry:
                cells.append("—")
                continue
            acc = entry["pass_at_1"] * 100
            accs.append(acc)
            cells.append(f"{acc:.2f}")
        avg = f"{sum(accs)/len(accs):.2f}" if accs else "—"
        lines.append(f"| {step} | " + " | ".join(cells) + f" | {avg} |")

    for title, key in (
        ("unparsed answer rate (%) — random-letter fallback fired", "unparsed_rate"),
        ("truncated rate (%) — hit the token cap", "truncated_rate"),
        ("macro average over categories (%)", "macro_avg"),
    ):
        lines += ["", f"## {title}", "",
                  "| step | " + " | ".join(LABEL[n] for n in present) + " |",
                  "|---" * (len(present) + 1) + "|"]
        for step in sorted(rows):
            cells = []
            for name in present:
                entry = rows[step].get(name)
                cells.append(f"{entry[key]*100:.2f}" if entry and key in entry else "—")
            lines.append(f"| {step} | " + " | ".join(cells) + " |")

    n_note = {
        name: next(
            (v[name]["num_examples"] for v in rows.values() if name in v), None
        )
        for name in present
    }
    lines += [
        "",
        "n per benchmark: "
        + ", ".join(f"{LABEL[n]}={n_note[n]}" for n in present)
        + ".",
    ]

    text = "\n".join(lines) + "\n"
    (base / "scores_general.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
