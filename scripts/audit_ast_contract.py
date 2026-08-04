"""Run the AST contract over every generator on disk.

Two populations, and the difference between them is the whole point:

  * the CLEAN set -- the hand-written seeds plus the verified mutation-pair
    fixtures -- must produce zero findings. A previous strict lint was reverted
    after it rejected 20 of 20 mathematically sound programs on shape alone
    (see the comment at ``evolution.py:136-149``); this population is the guard
    against repeating that, and a single firing here blocks the rule.

  * the ARCHIVE -- every champion across every run's snapshots -- is a defect
    corpus, not a positive one. At least a third of it is known broken, so a
    high firing rate here is expected and a zero rate would mean the rule does
    nothing.

Usage:
    python scripts/audit_ast_contract.py [--show CODE] [--limit N]
"""

from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rq_evolve.ast_contract import (  # noqa: E402
    check_generator_contract,
    check_problem_text,
)
from rq_evolve.program import ProblemProgram  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def clean_corpus() -> list[tuple[str, str]]:
    out = [(p.name, p.read_text()) for p in sorted((ROOT / "seed_programs").glob("*.py"))]
    for path in sorted((ROOT / "tests" / "fixtures" / "mutation_pairs").glob("*.txt")):
        blocks = re.findall(r"```python\n(.*?)```", path.read_text(), re.S)
        for index, block in enumerate(blocks):
            out.append((f"{path.stem}#{index}", block))
    return out


def archive_corpus() -> dict[str, dict]:
    """Every distinct champion source across every snapshot, newest wins."""
    programs: dict[str, dict] = {}
    for name in sorted(glob.glob(str(ROOT / "rq_output/**/archive*.json"), recursive=True)):
        try:
            payload = json.loads(Path(name).read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for champion in payload.get("champions", []):
            source = champion.get("source_code")
            if source:
                programs[source] = champion
    return programs


def rejected_corpus() -> list[tuple[str, str]]:
    """The few rejected generators whose source survived anywhere on disk."""
    out: list[tuple[str, str]] = []
    for name in sorted(glob.glob(str(ROOT / "rq_output/**/probe*.jsonl"), recursive=True)):
        for line in Path(name).read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source"):
                out.append((f"{Path(name).parent.name}:{row.get('i')}", row["source"]))
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show", help="print sources flagged with this rule code")
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()

    clean = clean_corpus()
    failures = [(n, check_generator_contract(s)) for n, s in clean]
    false_positives = [(n, f) for n, f in failures if f]
    print(f"CLEAN  {len(clean)} programs")
    for name, findings in false_positives:
        print(f"   FALSE POSITIVE {name}: {'; '.join(str(f) for f in findings)}")
    print(f"   findings: {len(false_positives)}  (must be 0)\n")

    archive = archive_corpus()
    per_rule: Counter[str] = Counter()
    flagged: dict[str, list[str]] = {}
    any_flagged = 0
    for source in archive:
        findings = check_generator_contract(source)
        if findings:
            any_flagged += 1
        for finding in findings:
            per_rule[finding.code] += 1
            flagged.setdefault(finding.code, []).append(source)

    print(f"ARCHIVE  {len(archive)} distinct champion sources")
    for code, count in per_rule.most_common():
        print(f"   {code:<5} {count:>4}  {count / len(archive):>5.1%}")
    print(f"   {'ANY':<5} {any_flagged:>4}  {any_flagged / len(archive):>5.1%}\n")

    # P2 needs an executed statement, so it is measured separately.
    text_hits = 0
    executed = 0
    for source in archive:
        instance = ProblemProgram(source_code=source).execute(seed=0)
        if instance is None:
            continue
        executed += 1
        if check_problem_text(instance.problem):
            text_hits += 1
    print(f"P2 (rendered statement)  {text_hits}/{executed} executable  "
          f"{text_hits / max(executed, 1):.1%}\n")

    rejected = rejected_corpus()
    if rejected:
        hits = sum(1 for _, s in rejected if check_generator_contract(s))
        print(f"REJECTED-WITH-SOURCE  {hits}/{len(rejected)} flagged\n")

    if args.show:
        for source in flagged.get(args.show, [])[: args.limit]:
            print("=" * 70)
            print(source)

    return 1 if false_positives else 0


if __name__ == "__main__":
    raise SystemExit(main())
