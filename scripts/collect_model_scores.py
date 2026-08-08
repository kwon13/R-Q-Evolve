#!/usr/bin/env python
"""Model x benchmark tables for scripts/eval_models_fanout.sh.

The step-wise aggregators walk global_step_* under one run; this walks one
directory per model. Math and general are reported separately because they are
graded differently -- math carries a GPT-4o re-check, general is exact-match
with a random-letter fallback -- and averaging across them would hide both.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

MATH = ["math500", "gsm8k", "amc23", "aime24", "aime25", "minerva_math",
        "olympiadbench"]
GENERAL = ["mmlupro", "supergpqa", "bbeh"]
GEN_LABEL = {"mmlupro": "MMLU-Pro", "supergpqa": "SuperGPQA", "bbeh": "BBEH"}


def read(path: Path, name: str) -> dict | None:
    p = path / name / "summary.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())["benchmarks"][name]
    except (KeyError, json.JSONDecodeError):
        return None


def table(models: list[Path], sub: str, names: list[str], label: dict | None,
          key: str = "pass_at_1") -> list[str]:
    head = [label[n] if label else n for n in names]
    lines = ["| model | " + " | ".join(head) + " | AVG |",
             "|---" * (len(names) + 2) + "|"]
    rows = []
    for m in models:
        cells, got = [], {}
        for n in names:
            entry = read(m / sub, n)
            if entry is None:
                cells.append("—")
                continue
            got[n] = entry[key] * 100
            cells.append(f"{got[n]:.2f}")
        if got:
            rows.append((m.name, cells, got))
    if not rows:
        return []
    # Average over the benchmarks every listed model has. Averaging each row
    # over whatever it happens to have makes a model with a missing hard
    # benchmark look better than one that ran it: dropping minerva_math alone
    # moved a row by several points.
    common = set(names)
    for _, _, got in rows:
        common &= set(got)
    common_list = [n for n in names if n in common]
    for name, cells, got in rows:
        avg = (f"{sum(got[n] for n in common_list) / len(common_list):.2f}"
               if common_list else "—")
        lines.append(f"| `{name}` | " + " | ".join(cells) + f" | {avg} |")
    if common_list and len(common_list) < len(names):
        missing = [n for n in names if n not in common]
        lines.append("")
        lines.append(f"AVG is over the {len(common_list)} benchmarks all models "
                     f"have; {', '.join(missing)} incomplete.")
    return lines


def main() -> int:
    root = Path(os.environ.get("OUT_ROOT")
                or (sys.argv[1] if len(sys.argv) > 1
                    else "rq_output/model_bench"))
    models = sorted(d for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    if not models:
        print(f"no model directories under {root}", file=sys.stderr)
        return 1

    out = ["# Standalone model benchmarks", "", f"Root: `{root}`", ""]
    math_rows = table(models, "eval", MATH, None)
    if math_rows:
        out += ["## math — pass@1 (%), R-Zero grading with GPT-4o re-check", ""]
        out += math_rows + [""]
        pre = table(models, "eval", MATH, None, key="pass_at_1_pre_gpt")
        if pre:
            out += ["## math — pass@1 (%), math_verify only", ""] + pre + [""]

    gen_rows = table(models, "eval_general", GENERAL, GEN_LABEL)
    if gen_rows:
        out += ["## general-domain — accuracy (%)", ""] + gen_rows + [""]
        # An unparsed multiple-choice answer still scores 1/10 through the
        # random fallback, so accuracy alone cannot be told apart from guessing.
        for title, key in (("unparsed answer rate (%)", "unparsed_rate"),
                           ("truncated rate (%)", "truncated_rate")):
            rows = table(models, "eval_general", GENERAL, GEN_LABEL, key=key)
            if rows:
                out += [f"## general-domain — {title}", ""] + rows + [""]

    text = "\n".join(out) + "\n"
    (root / "scores_models.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
