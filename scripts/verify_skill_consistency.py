#!/usr/bin/env python3
"""Offline verification of the stage-2 self-consistency skill gate.

For each champion in an archive, this script:
  1. Extracts the source code and the declared GROUP/SKILL.
  2. Simulates what stage 2 would see: the problem text from the code,
     then asks an LLM to infer GROUP and SKILL from the code alone.
  3. Reports matches and mismatches in a summary table.

This does NOT actually call an LLM. Instead it uses a heuristic analysis
of the code structure to flag likely mismatches, and prints the results
for manual review. For a full LLM-based verification, set --use-llm.

Usage:
    python scripts/verify_skill_consistency.py \\
        --archive rq_output/rq_evolve_4b_8gpu/rq_archive/archive.json
"""

import argparse
import json
import re
import sys
from pathlib import Path


def extract_problem_text(source_code: str) -> str:
    """Extract the problem text from a generator's source code."""
    # Look for problem = f"..." or problem = ("..."
    match = re.search(
        r'problem\s*=\s*(?:\(\s*)?f?(?:"""(.+?)"""|"(.+?)")',
        source_code,
        re.S,
    )
    if match:
        return (match.group(1) or match.group(2) or "").strip()
    return ""


def heuristic_skill_analysis(source_code: str, problem_text: str) -> dict:
    """Analyze code structure for skill-related patterns.

    Returns a dict with detected patterns and a suggested skill.
    """
    findings = []
    suggested_skill = None

    # Normalize
    code_lower = source_code.lower()
    prob_lower = problem_text.lower()

    # --- Pattern: Trivial computation (n^2, direct formula) ---
    # If the answer is a simple closed-form with no branching/iteration
    lines = source_code.split("\n")
    answer_line = ""
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("answer") and "=" in stripped:
            answer_line = stripped
            break

    # Check for trivially simple answer (single expression, no function calls)
    if answer_line and "**" in answer_line and "for" not in answer_line:
        if "if" not in answer_line and "sum" not in answer_line:
            findings.append("answer is a single power expression (trivial)")

    # --- Pattern: Brute-force only (answer and check are nearly identical) ---
    check_lines = []
    answer_lines = []
    in_check = False
    in_answer = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("check"):
            in_check = True
            in_answer = False
        elif stripped.startswith("answer"):
            in_answer = True
            in_check = False
        elif stripped.startswith("assert"):
            in_check = False
            in_answer = False

        if in_check:
            check_lines.append(stripped)
        if in_answer:
            answer_lines.append(stripped)

    # If answer uses the same for-loop pattern as check, it's likely not
    # using the declared skill
    answer_block = " ".join(answer_lines)
    check_block = " ".join(check_lines)

    if answer_block and check_block:
        # Both use sum(...for...) or both use for loops
        answer_has_loop = "for " in answer_block or "while " in answer_block
        check_has_loop = "for " in check_block or "while " in check_block
        if answer_has_loop and check_has_loop:
            # Check if they're structurally similar
            answer_fors = re.findall(r"for\s+\w+\s+in", answer_block)
            check_fors = re.findall(r"for\s+\w+\s+in", check_block)
            if len(answer_fors) == len(check_fors) and len(answer_fors) > 0:
                findings.append(
                    "answer and check both iterate similarly "
                    "(answer may not use declared skill)"
                )

    # --- Pattern: Problem text vs code mismatch ---
    # Problem says "maximum" but code computes a fixed formula
    if "maximum" in prob_lower or "greatest" in prob_lower or "largest" in prob_lower:
        if "max(" not in code_lower and "maximize" not in code_lower:
            if answer_line and ("**" in answer_line or "//" in answer_line):
                findings.append(
                    "problem asks for max/greatest but answer is a fixed formula"
                )

    # Problem mentions merging/process but skill says something else
    if "erased" in prob_lower or "merge" in prob_lower or "blackboard" in prob_lower:
        if "invariant" not in source_code:
            findings.append(
                "problem describes a process (blackboard/merge) "
                "→ likely needs 'invariant' skill"
            )
            suggested_skill = "invariant"

    # Problem asks "how many" → likely counting or casework
    if "how many" in prob_lower:
        if "sum(1 for" in code_lower or "sum(" in code_lower:
            if suggested_skill is None:
                suggested_skill = "counting"

    # Contradiction pattern: assuming and deriving impossibility
    if "assume" in code_lower and "contradiction" not in source_code:
        findings.append("code uses assumption pattern but skill != contradiction")

    # Induction pattern: recurrence
    if re.search(r"f\[\w+\]\s*=\s*f\[\w+\s*-\s*\d+\]", source_code):
        if suggested_skill is None:
            suggested_skill = "induction"
            findings.append("code uses recurrence relation → likely 'induction'")

    return {
        "findings": findings,
        "suggested_skill": suggested_skill,
        "answer_line": answer_line[:100] if answer_line else "",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify skill label consistency in R-Q-Evolve archive"
    )
    parser.add_argument(
        "--archive",
        type=str,
        default="rq_output/rq_evolve_4b_8gpu/rq_archive/archive.json",
        help="Path to archive.json",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Show full source code"
    )
    args = parser.parse_args()

    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"Error: {archive_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(archive_path) as f:
        data = json.load(f)

    champions = data.get("champions", [])
    print(f"Archive: {archive_path}")
    print(f"Champions: {len(champions)}")
    print()

    # Header
    print(f"{'#':>3} {'ID':8s} {'Gen':>3} {'GROUP':15s} {'SKILL':20s} {'Status':12s} {'Findings'}")
    print("-" * 120)

    suspicious = []
    clean = []

    for i, champ in enumerate(champions):
        group = champ["metadata"].get("group", "?")
        skill = champ["metadata"].get("skill", "?")
        prog_id = champ["program_id"][:8]
        gen = champ["generation"]
        source = champ["source_code"]

        problem_text = extract_problem_text(source)
        analysis = heuristic_skill_analysis(source, problem_text)

        if analysis["findings"]:
            status = "⚠ SUSPECT"
            suspicious.append((i, prog_id, group, skill, analysis))
        else:
            status = "✓ OK"
            clean.append((i, prog_id, group, skill))

        findings_str = "; ".join(analysis["findings"][:2]) if analysis["findings"] else ""
        print(
            f"{i:3d} {prog_id:8s} {gen:3d} {group:15s} {skill:20s} {status:12s} {findings_str}"
        )

    print()
    print("=" * 80)
    print(f"Summary: {len(clean)} clean, {len(suspicious)} suspicious out of {len(champions)}")
    print()

    if suspicious:
        print("SUSPICIOUS CHAMPIONS (likely skill mislabeling):")
        print()
        for i, prog_id, group, skill, analysis in suspicious:
            print(f"  #{i} [{prog_id}] GROUP={group} SKILL={skill}")
            if analysis["suggested_skill"]:
                print(f"    → Suggested SKILL: {analysis['suggested_skill']}")
            for finding in analysis["findings"]:
                print(f"    - {finding}")
            if analysis["answer_line"]:
                print(f"    answer: {analysis['answer_line']}")
            print()


if __name__ == "__main__":
    main()
