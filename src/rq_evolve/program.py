import ast
import hashlib
import importlib.util
import json
import math
import random
import signal
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .concepts import concept_group_for_type

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
        """Run ``generate(seed)`` in a small import-guarded namespace."""

        def guarded_import(
            name: str,
            globals_: dict | None = None,
            locals_: dict | None = None,
            fromlist: tuple = (),
            level: int = 0,
        ):
            root = name.split(".", 1)[0]
            if root not in ALLOWED_IMPORT_ROOTS:
                raise ImportError(f"import not allowed: {name}")
            return __import__(name, globals_, locals_, fromlist, level)

        import builtins as _builtins

        # Permissive blocklist sandbox: expose every builtin EXCEPT IO, dynamic
        # code execution, and introspection-mutation. This mirrors evo-sample's
        # full-__builtins__ generosity (so generators using next/divmod/map/
        # filter/reversed/frozenset/itertools-style helpers run) while still
        # blocking the dangerous ones. Defense in depth: imports go through
        # guarded_import (ALLOWED_IMPORT_ROOTS only) and lint_generator_source
        # rejects forbidden source patterns before execution.
        _FORBIDDEN_BUILTINS = {
            "open", "eval", "exec", "compile", "input",
            "getattr", "setattr", "delattr",
            "globals", "locals", "vars",
            "exit", "quit", "help", "breakpoint",
        }
        safe_builtins = {
            name: getattr(_builtins, name)
            for name in dir(_builtins)
            if not name.startswith("_") and name not in _FORBIDDEN_BUILTINS
        }
        # Dunder builtins are excluded by the filter above. Re-add the two we
        # need: a sandboxed importer, and class-definition support.
        safe_builtins["__import__"] = guarded_import
        safe_builtins["__build_class__"] = _builtins.__build_class__

        def timeout_handler(signum, frame):
            raise TimeoutError("program execution timed out")

        spec = importlib.util.spec_from_loader("rq_generated_program", loader=None)
        module = importlib.util.module_from_spec(spec)
        module.__dict__.update(
            {
                "__builtins__": safe_builtins,
                "math": math,
                "random": random,
            }
        )

        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, int(timeout)))
        try:
            exec(self.source_code, module.__dict__)
            generate = getattr(module, "generate", None)
            if generate is None:
                return None
            result = generate(seed)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                return None
            problem, answer = str(result[0]), str(result[1])
            if not problem.strip() or not answer.strip():
                return None
            return ProblemInstance(
                problem=problem,
                answer=answer,
                program_id=self.program_id,
                seed=seed,
            )
        except Exception:
            return None
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

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

