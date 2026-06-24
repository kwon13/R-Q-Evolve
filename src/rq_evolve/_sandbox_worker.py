"""Hermetic subprocess sandbox for executing generated problem programs.

Run as a PLAIN SCRIPT (``python .../_sandbox_worker.py``), never ``-m``: launched
as a path, Python puts only this file's directory on ``sys.path`` and runs it as
``__main__`` -- it does NOT import the ``rq_evolve`` package, so a fresh worker
starts fast and never re-imports torch/vllm/verl. It imports only the stdlib
(plus ``sympy`` on demand, inside a generated program's own ``import``).

Protocol. The parent (``program._SandboxClient``) keeps ONE of these alive and
sends one request per line on stdin::

    {"source": "<generator source>", "seed": <int>}\n

and reads one reply per line back::

    {"ok": true, "problem": "...", "answer": "..."}\n   (or {"ok": false})

A request that overruns the parent's wall-clock budget is killed (SIGKILL) and a
fresh worker is spawned for the next call. That hard kill is the whole point:
``signal.alarm`` (the previous in-process timeout) cannot interrupt a generated
program spinning in a C-level call -- ``2**(2**30)``, ``math.factorial(10**8)``,
a huge ``sympy`` simplify -- because the SIGALRM handler can't run until the C
call returns to the interpreter. Killing the process actually stops it.

The sandbox namespace mirrors ``ProblemProgram.execute`` EXACTLY. It is kept as a
self-contained copy (not imported from ``program.py``) precisely so that importing
anything here costs nothing beyond the stdlib.
"""

import importlib.util
import json
import math
import os
import random
import resource
import sys

# Mirror of program.ProblemProgram's sandbox policy -- keep in sync by hand.
ALLOWED_IMPORT_ROOTS = {
    "collections",
    "fractions",
    "functools",
    "itertools",
    "math",
    "random",
    "sympy",
}
_FORBIDDEN_BUILTINS = {
    "open", "eval", "exec", "compile", "input",
    "getattr", "setattr", "delattr",
    "globals", "locals", "vars",
    "exit", "quit", "help", "breakpoint",
}

# Address-space cap so a memory-bomb generator (e.g. ``10**10**9``, whose integer
# alone is gigabytes) raises MemoryError in here instead of OOM-killing the node.
# 8 GiB is far above anything legitimate problem generation needs, well below the
# node's RAM. The wall-clock kill in the parent handles CPU spinners; this only
# guards the allocation axis.
_MEM_LIMIT_BYTES = 8 * 1024**3


def _run(source: str, seed: int):
    import builtins as _builtins

    def guarded_import(name, globals_=None, locals_=None, fromlist=(), level=0):
        root = name.split(".", 1)[0]
        if root not in ALLOWED_IMPORT_ROOTS:
            raise ImportError(f"import not allowed: {name}")
        return __import__(name, globals_, locals_, fromlist, level)

    safe_builtins = {
        n: getattr(_builtins, n)
        for n in dir(_builtins)
        if not n.startswith("_") and n not in _FORBIDDEN_BUILTINS
    }
    safe_builtins["__import__"] = guarded_import
    safe_builtins["__build_class__"] = _builtins.__build_class__

    spec = importlib.util.spec_from_loader("rq_generated_program", loader=None)
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(
        {"__builtins__": safe_builtins, "math": math, "random": random}
    )

    exec(source, module.__dict__)
    generate = getattr(module, "generate", None)
    if generate is None:
        return None
    result = generate(seed)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        return None
    problem, answer = str(result[0]), str(result[1])
    if not problem.strip() or not answer.strip():
        return None
    return {"problem": problem, "answer": answer}


def main() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
    except Exception:
        pass

    # fd dance: the ORIGINAL stdout (the pipe back to the parent) becomes the
    # private protocol channel; fd 1 and Python-level sys.stdout are redirected to
    # /dev/null so a ``print(...)`` inside a generated program can never corrupt
    # the JSON wire.
    protocol = os.fdopen(os.dup(1), "w")
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    sys.stdout = os.fdopen(os.open(os.devnull, os.O_WRONLY), "w")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out = _run(req["source"], int(req["seed"]))
            resp = {"ok": out is not None}
            if out is not None:
                resp.update(out)
        except BaseException:
            resp = {"ok": False}
        protocol.write(json.dumps(resp) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
