"""Persistent killable worker for potentially pathological difflib calls."""

from __future__ import annotations

from difflib import SequenceMatcher
import json
import sys


def _sequence(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return value
    raise TypeError("similarity inputs must be strings or lists of strings")


def _handle(request: dict) -> dict:
    left = _sequence(request.get("left"))
    right = _sequence(request.get("right"))
    matcher = SequenceMatcher(
        None,
        left,
        right,
        autojunk=bool(request.get("autojunk", False)),
    )
    operation = request.get("operation")
    if operation == "ratio":
        value = matcher.ratio()
    elif operation == "matched":
        value = sum(block.size for block in matcher.get_matching_blocks())
    else:
        raise ValueError("unknown similarity operation")
    return {"ok": True, "value": value}


def main() -> None:
    sys.stdout.write('{"ready":true}\n')
    sys.stdout.flush()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            response = _handle(request)
        except BaseException as exc:
            response = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
        sys.stdout.write(json.dumps(response, separators=(",", ":")) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
