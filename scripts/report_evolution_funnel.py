#!/usr/bin/env python
"""Print the mutation funnel for a run: how many children survive each gate.

    python scripts/report_evolution_funnel.py rq_output/evo_probe_4b

Reads ``<run>/rq_archive/evolution_log.jsonl``. Three things it reports that the
per-iteration metrics do not:

* the failure modes behind ``verify_failed``, which is otherwise one opaque
  bucket holding both "the model's mathematics is wrong" and "the model wrote
  no assert";
* duplicate children -- ``program_id`` is md5 of the source, so a repeated id is
  a byte-identical program that paid for execution and a judge call twice;
* whether the judge is DETERMINISTIC, by checking that every repeat of the same
  source got the same verdict. It is configured at temperature 0, and until
  patches/verl_agent_loop_sampling.py that setting was silently dropped.
"""

from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

BUCKETS = [
    (r"\bA1\b|no assert", "A1  no assert at all"),
    (r"\bA2\b", "A2  no independent cross-check"),
    (r"\bA3v\b", "A3v check is the same computation"),
    (r"\bA4d\b", "A4d check derived from the answer"),
    (r"\bA5v\b", "A5v check == answer expression"),
    (r"\bP1\b", "P1  sampled value absent from problem"),
    (r"\bP2\b", "P2  statement names its own technique"),
    (r"AssertionError|answer=.*check=", "assert FIRED (routes disagree)"),
    (r"missing GROUP|missing SKILL|unknown GROUP|unknown SKILL", "label missing / out of vocabulary"),
    (r"does not vary its visible problem", "constant across seeds"),
    (r"no parseable generate", "unparseable output"),
    (r"Sample larger|IndexError|ZeroDivision|division by zero|empty range", "sampling invalid for some seed"),
    (r"NameError|not defined|ImportError", "undefined name / missing import"),
    (r"Timeout|timed out", "timeout / non-terminating"),
    (r"Recursion", "runaway recursion"),
    (r"duplicate", "duplicate behaviour"),
    (r"Syntax|Indentation", "syntax error"),
    (r"label mismatch", "judge: labels disagree"),
    (r"judge failed closed", "judge: no parseable label"),
    (r"execute failed", "other runtime error"),
]


def bucket(rep: dict) -> str:
    text = " ".join(
        [str(rep.get("reason") or "")] + [str(x) for x in (rep.get("ast_findings") or [])]
    )
    for pattern, name in BUCKETS:
        if re.search(pattern, text, re.I):
            return name
    return (text[:60].strip() or "(no reason recorded)")


def main(run_dir: str) -> int:
    path = Path(run_dir) / "rq_archive" / "evolution_log.jsonl"
    if not path.exists():
        print(f"no evolution log at {path}")
        return 1
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        print("evolution log is empty -- no iteration has completed yet")
        return 1

    print(f"{'iter':>4} {'tried':>6} {'INSERTED':>9} {'verify_fail':>12} "
          f"{'judge_rej':>10} {'no_code':>8} {'judge_seen':>11} {'judge_ok':>9} {'cells':>6}")
    total_tried = total_ins = 0
    for r in rows:
        m = r["metrics"]
        total_tried += m["attempted"]
        total_ins += m["inserted"]
        print(f"{m['outer_iteration']:>4} {m['attempted']:>6} {m['inserted']:>9} "
              f"{m['status_verify_failed']:>12} {m['status_judge_rejected']:>10} "
              f"{m.get('status_no_code', 0):>8} {m['judge_reached']:>11} "
              f"{m['judge_agreed']:>9} {int(m['coverage'] * m['total_niches']):>6}")
    rate = 100 * total_ins / total_tried if total_tried else 0.0
    print(f"\nACCEPT RATE: {total_ins}/{total_tried} = {rate:.1f}%")

    reports = [rep for r in rows for rep in r["reports"]]
    counts = collections.Counter(bucket(rep) for rep in reports)
    print(f"\n{'-'*62}\nfailure modes ({len(reports)} candidates)")
    for name, n in counts.most_common():
        print(f"  {n:4d}  {100*n/len(reports):5.1f}%  {name}")

    ids = collections.Counter(
        rep.get("child_id") for rep in reports if rep.get("child_id")
    )
    repeats = {cid: n for cid, n in ids.items() if n > 1}
    print(f"\n{'-'*62}\nduplicate children: {len(repeats)} ids repeated, "
          f"{sum(repeats.values()) - len(repeats)} wasted evaluations")
    for cid, n in sorted(repeats.items(), key=lambda kv: -kv[1])[:5]:
        verdicts = {
            str(rep.get("reason"))[:70]
            for rep in reports
            if rep.get("child_id") == cid
        }
        flag = "OK  " if len(verdicts) == 1 else "VARY"
        print(f"  {cid}  x{n}  judge verdict {flag} ({len(verdicts)} distinct)")
        if len(verdicts) > 1:
            for v in sorted(verdicts):
                print(f"        - {v}")
    if repeats:
        varying = sum(
            1
            for cid in repeats
            if len({str(rep.get("reason"))[:70] for rep in reports
                    if rep.get("child_id") == cid}) > 1
        )
        print(f"\n  judge determinism: {len(repeats) - varying}/{len(repeats)} "
              f"repeated children got a consistent verdict"
              + ("" if varying == 0 else "   <-- temperature override still not reaching vLLM"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "rq_output/evo_probe_4b"))
