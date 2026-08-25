import ast
from .safe_parse import safe_ast_parse
import hashlib
import json
import os
import select
import subprocess
import sys
import threading
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

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
        # False until this worker has answered once. The first request after a
        # spawn also pays interpreter startup and the worker's `_warm_imports`
        # (sympy: ~1.5 s), so it gets `cold_timeout` and every later request
        # gets the tight steady-state `timeout`.
        self._warm = False

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_SANDBOX_WORKER_PATH)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._warm = False

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=2)
            except Exception:
                pass
            self._proc = None
        self._warm = False

    def run(
        self, source: str, seed: int, timeout: float, cold_timeout: float
    ) -> dict:
        """Return ``{"ok": True, "problem", "answer"}`` or ``{"ok": False, ...}``.

        Failures carry ``error_type``/``error`` so a caller can tell a child's
        own AssertionError -- its cross-check catching a real problem/answer
        mismatch -- from broken code, a guarded import, or a timeout.

        Two budgets, because they measure different things: ``cold_timeout``
        covers the first request a freshly spawned worker answers (process
        startup + `_warm_imports`), ``timeout`` covers every later one, where
        the only thing left to wait for is the generated program itself.
        """
        with self._lock:
            try:
                if self._proc is None or self._proc.poll() is not None:
                    self._spawn()
                budget = float(timeout if self._warm else max(timeout, cold_timeout))
                self._proc.stdin.write(
                    json.dumps({"source": source, "seed": seed}) + "\n"
                )
                self._proc.stdin.flush()
            except Exception as exc:
                self._kill()
                return {"ok": False, "error_type": "SandboxWriteError",
                        "error": str(exc)[:200]}

            ready, _, _ = select.select([self._proc.stdout], [], [], budget)
            if not ready:
                # Overran the budget -> the worker is wedged in C-level work.
                # Kill it; the next call respawns a clean one.
                self._kill()
                return {"ok": False, "error_type": "Timeout",
                        "error": f"no result within {budget}s"}
            self._warm = True
            line = self._proc.stdout.readline()
            if not line:  # worker died mid-request (e.g. MemoryError-killed)
                self._kill()
                return {"ok": False, "error_type": "WorkerDied",
                        "error": "worker exited mid-request (memory limit?)"}
            try:
                return json.loads(line)
            except Exception as exc:
                return {"ok": False, "error_type": "ProtocolError",
                        "error": str(exc)[:200]}


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
    # Paper notation: s = success rate, U = uncertainty, R_Q = s(1-s)U.
    # There is no separate `fitness`: it held a byte-for-byte copy of rq_score
    # and nothing ever read it to decide anything.
    s_hat: float = 0.0
    u_score: float = 0.0
    rq_score: float = 0.0
    # Grid coordinate on the GROUP x SKILL archive. Both are derived from the
    # program's own labels, so they are a cache of program_to_cell, never an
    # independent fact.
    niche_group: int = -1
    niche_skill: int = -1
    # Why the most recent execute() returned None. Diagnostic only, never
    # persisted: to_dict does not carry it.
    last_execution_error: str | None = field(default=None, compare=False)
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
            tree = safe_ast_parse(self.source_code)
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

    # --- MAP-Elites axis labels -------------------------------------------
    # A program declares GROUP (domain) and SKILL (reasoning move) at module
    # top level. ``declared_*`` reads the source text; ``get_*`` prefers the
    # value verification already resolved into metadata, so a program keeps its
    # labels after being restored from an archive snapshot whose source may
    # predate the current vocabulary.

    def declared_group(self) -> str | None:
        # CONCEPT_GROUP is the pre-migration spelling of the same axis, so a
        # snapshot written before the rename still reports its domain.
        return (
            self._top_level_string_constant("GROUP")
            or self._top_level_string_constant("CONCEPT_GROUP")
        )

    def declared_skill(self) -> str | None:
        # No fallback: SKILL has no pre-migration equivalent. CONCEPT_TYPE was a
        # refinement of the domain (``number_theory.crt_count``), not a
        # reasoning move, so reading it here would fabricate a skill label.
        return self._top_level_string_constant("SKILL")

    def get_group(self) -> str | None:
        return self.metadata.get("group") or self.declared_group()

    def get_skill(self) -> str | None:
        return self.metadata.get("skill") or self.declared_skill()

    # --- Pre-migration accessors ------------------------------------------
    # Still called by archive.py, evolution.py, prompts.py and
    # expansion_experiment.py. Removed once those move to GROUP x SKILL.

    def declared_concept_group(self) -> str | None:
        """LEGACY alias of :meth:`declared_group` -- same axis, old spelling."""
        return self.declared_group()

    def get_concept_group(self) -> str | None:
        """LEGACY alias of :meth:`get_group` -- same axis, old spelling."""
        return self.get_group()

    def declared_concept_type(self) -> str | None:
        """LEGACY. The retired CONCEPT_TYPE label; None on a GROUP/SKILL program.

        NOT a stand-in for SKILL. Any comparison of two such programs' concept
        types compares None with None and always succeeds, so it can never be
        the thing that decides a label question.
        """
        return self._top_level_string_constant("CONCEPT_TYPE")

    def get_concept_type(self) -> str | None:
        """LEGACY. See :meth:`declared_concept_type` for the inert-comparison caveat."""
        return self.metadata.get("concept_type") or self.declared_concept_type()

    def execute(
        self, seed: int, timeout: float = 1.0, cold_timeout: float = 15.0
    ) -> ProblemInstance | None:
        """Run ``generate(seed)`` in a hard-killable sandbox subprocess.

        The import-guarded namespace and builtin blocklist live in
        ``_sandbox_worker.py``; the worker is run in a separate spawned
        interpreter so a generator that spins in a C-level call (which
        ``signal.alarm`` could never interrupt) is SIGKILLed at ``timeout``
        instead of wedging the trainer. Any failure -- bad program, timeout,
        worker crash -- comes back as None, exactly as the old in-process path
        signalled it.

        ``timeout`` is 1.0, not the 5.0 it used to be. Re-running the 4B run's
        22 surviving champions x 5 seeds through this path costs 0.6 ms per
        execute, so the four extra seconds only ever bought waiting time on a
        runaway -- and the 4B run hit 246 of those, all serialised behind the
        single global worker. What the old value was really covering was
        cold start, which now has its own budget: ``cold_timeout`` applies to
        the first request a freshly spawned worker answers (measured 1.5 s,
        nearly all of it the worker's eager `import sympy`).
        """
        resp = _SANDBOX.run(self.source_code, seed, timeout, cold_timeout)
        if not resp.get("ok"):
            # Kept for the caller that wants to know WHY, without changing the
            # None-on-failure contract every existing call site relies on.
            self.last_execution_error = (
                f"{resp.get('error_type', 'Unknown')}: {resp.get('error', '')}"
            )
            return None
        self.last_execution_error = None
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
            "s_hat": self.s_hat,
            "u_score": self.u_score,
            "rq_score": self.rq_score,
            "niche_group": self.niche_group,
            "niche_skill": self.niche_skill,
            "last_reeval_step": self.last_reeval_step,
            "metadata": self.metadata,
        }

    # Wire keys written before the notation was aligned with the paper. Every
    # archive.json on disk -- including the completed 4B/8B runs kept under
    # analysis/ -- spells these the old way, and a resume that silently reset a
    # champion's score to 0.0 would be worse than one that crashed.
    _LEGACY_KEYS = {"p_hat": "s_hat", "h_score": "u_score"}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ProblemProgram":
        """Rebuild from :meth:`to_dict`, ignoring fields this version dropped.

        Snapshots outlive schema changes: a pre-migration archive still carries
        ``niche_h``/``niche_div`` from the retired H axis, and ``fitness`` from
        back when R_Q was stored twice. Those are meaningless now and the
        archive re-derives the cell from the labels anyway, so unknown keys are
        skipped rather than raising and taking the whole resume down. The two
        keys that were RENAMED rather than dropped are mapped, not skipped.
        """
        known = {f.name for f in fields(cls)}
        resolved: dict[str, Any] = {}
        for key, value in payload.items():
            name = cls._LEGACY_KEYS.get(key, key)
            # An explicit new-style key always wins over its legacy spelling.
            if name in known and name not in payload or name == key:
                resolved[name] = value
        return cls(**{k: v for k, v in resolved.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "ProblemProgram":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

