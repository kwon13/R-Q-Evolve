"""Hard-time-bounded sequence similarity for model-controlled material.

``difflib.SequenceMatcher(autojunk=False)`` has quadratic worst cases on long,
repetitive sequences. Archive admission runs on the trainer driver, so one such
candidate can otherwise pin its main thread at 100% CPU while every GPU waits.
The exact historical metric is retained, but evaluated in a subprocess that can
be killed and restarted when it exceeds its wall-clock budget.
"""

from __future__ import annotations

import atexit
import json
import os
import select
import subprocess
import sys
import threading
from pathlib import Path
from typing import Sequence


_WORKER_PATH = Path(__file__).with_name("_similarity_worker.py")
_TIMEOUT = float(os.environ.get("RQ_SIMILARITY_TIMEOUT", "1.0"))
_COLD_TIMEOUT = float(os.environ.get("RQ_SIMILARITY_COLD_TIMEOUT", "5.0"))


class SimilarityTimeout(RuntimeError):
    """The exact comparison exceeded its hard wall-clock budget."""


class _SimilarityClient:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        assert self._proc.stdout is not None
        ready, _, _ = select.select(
            [self._proc.stdout], [], [], max(_TIMEOUT, _COLD_TIMEOUT)
        )
        if not ready:
            self._kill()
            raise SimilarityTimeout(
                "similarity worker startup exceeded "
                f"{max(_TIMEOUT, _COLD_TIMEOUT):.3f}s"
            )
        line = self._proc.stdout.readline()
        try:
            response = json.loads(line)
        except Exception as exc:
            self._kill()
            raise RuntimeError("invalid similarity worker startup response") from exc
        if not response.get("ready"):
            self._kill()
            raise RuntimeError("similarity worker did not become ready")

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def close(self) -> None:
        with self._lock:
            self._kill()

    def run(
        self,
        operation: str,
        left: str | Sequence[str],
        right: str | Sequence[str],
        *,
        autojunk: bool,
        timeout: float | None = None,
    ) -> float:
        request = {
            "operation": operation,
            "left": left,
            "right": right,
            "autojunk": bool(autojunk),
        }
        with self._lock:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            try:
                budget = float(timeout if timeout is not None else _TIMEOUT)
                assert self._proc.stdin is not None
                assert self._proc.stdout is not None
                self._proc.stdin.write(
                    json.dumps(request, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                self._proc.stdin.flush()
            except Exception as exc:
                self._kill()
                raise RuntimeError(f"similarity worker write failed: {exc}") from exc

            ready, _, _ = select.select([self._proc.stdout], [], [], budget)
            if not ready:
                self._kill()
                raise SimilarityTimeout(
                    f"{operation} comparison exceeded {budget:.3f}s"
                )
            line = self._proc.stdout.readline()
            if not line:
                self._kill()
                raise RuntimeError("similarity worker exited without a response")
            try:
                response = json.loads(line)
            except Exception as exc:
                self._kill()
                raise RuntimeError("invalid similarity worker response") from exc
            if not response.get("ok"):
                raise RuntimeError(
                    "similarity worker failed: "
                    f"{response.get('error_type')}: {response.get('error')}"
                )
            return float(response["value"])


_CLIENT = _SimilarityClient()
atexit.register(_CLIENT.close)


def sequence_ratio(
    left: str | Sequence[str],
    right: str | Sequence[str],
    *,
    autojunk: bool = False,
) -> float:
    """The exact ``SequenceMatcher.ratio`` under a hard timeout."""

    return _CLIENT.run("ratio", left, right, autojunk=autojunk)


def matched_size(
    left: str | Sequence[str],
    right: str | Sequence[str],
    *,
    autojunk: bool = False,
) -> int:
    """Sum of matching-block sizes under a hard timeout."""

    return int(_CLIENT.run("matched", left, right, autojunk=autojunk))
