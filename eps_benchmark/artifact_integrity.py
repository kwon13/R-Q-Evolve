"""Pure, standalone integrity primitives for the fixed EPS artifact."""

from __future__ import annotations

import hashlib
import json


LOCKED_ROW_FIELDS = (
    "benchmark",
    "sample_id",
    "program_name",
    "program_id",
    "program_sha256",
    "group",
    "skill",
    "seed",
    "problem",
    "answer",
    "instance_sha256",
    "index",
)

# Changing either digest requires a new benchmark version, not an in-place v1
# rewrite.  The legacy digest is used by the existing evaluator; the second
# digest additionally locks GROUP, SKILL, index, and benchmark name.
# v1 (superseded, archived under v1_archive/): its problem statements carried a
# trailing "State only the integer." absent from the training seed programs, and
# labeled_trees answers reached 12 digits.  v2 removes the phrase everywhere and
# caps that generator at 1e9.
#   v1 benchmark_sha256 = 64bea577c7f3499d8f0e1547acf31a8d7b136dad9d33eb37b6a1e2690dcdae95
#   v1 artifact_sha256  = 7b3f10a024b3ae5a09782216c65cbd9fe1fc966e57cd6edebd29f303e311567a
EXPECTED_BENCHMARK_SHA256 = (
    "7094ddcde5f2dc08211d65e8b992f928187cae6789df0eb84dce0bcf61375537"
)
EXPECTED_ARTIFACT_SHA256 = (
    "4ae707a98277227607881e2ad6dcbdef3ec97048ff329ce99dfcea793ad8f8d3"
)


def artifact_sha256(rows: list[dict]) -> str:
    """Hash every score- or label-relevant field in ordered rows."""

    payload = [
        {field: row[field] for field in LOCKED_ROW_FIELDS}
        for row in rows
    ]
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
