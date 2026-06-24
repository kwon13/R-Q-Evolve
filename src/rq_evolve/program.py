import ast
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concepts import concept_group_for_type

# Absolute path to the hermetic sandbox worker. Resolved once at import so the
# client can (re)spawn it regardless of the trainer's cwd.
_SANDBOX_WORKER_PATH = Path(__file__).with_name("_sandbox_worker.py")


class _SandboxClient:
    """One persistent subprocess that runs generated programs under a hard kill.

    ``signal.alarm`` (the previous timeout) cannot interrupt a generated program
    spinning in a C-level call (huge ``**``/``factorial``/``sympy`` work): the
    SIGALRM handler only runs once control returns to the interpreter, which for a
    runaway C loop is never -- so one such program pegged the trainer's MAIN thread
    at 100% CPU with every GPU idle. Here each ``generate`` runs in a separate
    spawned interpreter; if it overruns the wall-clock budget the parent SIGKILLs
    it (a thread or signal cannot stop C-level work; killing the process does) and
    lazily respawns a fresh worker for the next call.

    A single worker serialised by a lock is enough: ``ProblemProgram.execute`` is
    driven from the trainer's between-step evolution path (and archive refresh),
    not from many threads at once. The lock keeps it correct if that ever changes.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_SANDBOX_WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None

    def run(self, source: str, seed: int, timeout: float) -> dict | None:
        """Return ``{"problem","answer"}`` or None (bad program / timeout / crash)."""
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._spawn()
                self._proc.stdin.write(
                    json.dumps({"source": source, "seed": seed}) + "\n"
                )
                self._proc.stdin.flush()
            except Exception:
                self._kill()
                return None

            ready, _, _ = select.select([self._proc.stdout], [], [], timeout)
            if not ready:
                # Overran the budget -> the worker is wedged in C-level work.
                # Kill it; the next call respawns a clean one.
                self._kill()
                return None
            line = self._proc.stdout.readline()
            if not line:  # worker died mid-request (e.g. MemoryError-killed)
                self._kill()
                return None
            try:
                resp = json.loads(line)
            except Exception:
                return None
            if not resp.get("ok"):
                return None
            return {"problem": resp["problem"], "answer": resp["answer"]}


# Process-global client: one resident worker shared by all ProblemProgram.execute
# calls in this interpreter.
_SANDBOX = _SandboxClient()

ALLOWED_IMPORT_ROOTS = {
    "collections",
    "fractions",
    "functools",
    "itertools",
    "math",
    "random",
    "sympy",
}


@dataclass(slots=True)
class ProblemInstance:
    problem: str
    answer: str
    program_id: str
    seed: int
    verified: bool = False


@dataclass
class ProblemProgram:
    """A Python source file defining ``generate(seed) -> (problem, answer)``."""

    source_code: str
    program_id: str = ""
    parent_id: str = ""
    generation: int = 0
    p_hat: float = 0.0
    h_score: float = 0.0
    rq_score: float = 0.0
    fitness: float = 0.0
    niche_h: int = -1
    niche_div: int = -1
    last_reeval_step: int = -1
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.program_id:
            self.program_id = hashlib.md5(
                self.source_code.encode("utf-8")
            ).hexdigest()[:12]

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: Any) -> "ProblemProgram":
        path = Path(path)
        metadata = dict(kwargs.pop("metadata", {}))
        metadata.setdefault("source_file", path.name)
        return cls(path.read_text(encoding="utf-8"), metadata=metadata, **kwargs)

    def _top_level_string_constant(self, name: str) -> str | None:
        try:
            tree = ast.parse(self.source_code)
        except SyntaxError:
            return None

        for node in tree.body:
            value_node = None
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if name in names:
                    value_node = node.value
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == name:
                    value_node = node.value
            if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
                return value_node.value.strip()
        return None

    def declared_concept_type(self) -> str | None:
        return self._top_level_string_constant("CONCEPT_TYPE")

    def declared_concept_group(self) -> str | None:
        return self._top_level_string_constant("CONCEPT_GROUP")

    def get_concept_type(self) -> str | None:
        return self.metadata.get("concept_type") or self.declared_concept_type()

    def get_concept_group(self) -> str | None:
        return (
            self.metadata.get("concept_group")
            or self.declared_concept_group()
            or concept_group_for_type(self.get_concept_type())
        )

    def execute(self, seed: int, timeout: float = 5.0) -> ProblemInstance | None:
        """Run ``generate(seed)`` in a hard-killable sandbox subprocess.

        The import-guarded namespace and builtin blocklist live in
        ``_sandbox_worker.py``; the worker is run in a separate spawned
        interpreter so a generator that spins in a C-level call (which
        ``signal.alarm`` could never interrupt) is SIGKILLed at ``timeout``
        instead of wedging the trainer. Any failure -- bad program, timeout,
        worker crash -- comes back as None, exactly as the old in-process path
        signalled it.
        """
        resp = _SANDBOX.run(self.source_code, seed, timeout)
        if resp is None:
            return None
        return ProblemInstance(
            problem=resp["problem"],
            answer=resp["answer"],
            program_id=self.program_id,
            seed=seed,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_code": self.source_code,
            "program_id": self.program_id,
            "parent_id": self.parent_id,
            "generation": self.generation,
            "p_hat": self.p_hat,
            "h_score": self.h_score,
            "rq_score": self.rq_score,
            "fitness": self.fitness,
            "niche_h": self.niche_h,
            "niche_div": self.niche_div,
            "last_reeval_step": self.last_reeval_step,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemProgram":
        return cls(**payload)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProblemProgram":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

