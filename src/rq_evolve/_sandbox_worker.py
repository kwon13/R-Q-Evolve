import importlib.util
import itertools
import json
import math
import os
import random
import resource
import sys

# Names the model writes without importing. `comb` alone killed 370 candidates on
# the 4B run -- 3 of the 4 stage-2 worked examples open with a bare
# `import random`, so the file's first line is emitted long before the model
# knows it will need `comb`, and there is no going back to add the import.
# Injecting the names is the same move `math` and `random` already get below;
# re-running the 400 dead sources with these bound recovered 32.5% of them
# outright (the rest failed their own assert, i.e. the maths was wrong anyway).
_PRELUDE_NAMES = {
    "comb": math.comb,
    "factorial": math.factorial,
    "perm": math.perm,
    "gcd": math.gcd,
    "lcm": math.lcm,
    "isqrt": math.isqrt,
    "combinations": itertools.combinations,
    "permutations": itertools.permutations,
    "product": itertools.product,
    "accumulate": itertools.accumulate,
}

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
        {"__builtins__": safe_builtins, "math": math, "random": random,
         **_PRELUDE_NAMES}
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


def _warm_imports() -> None:
    """Pay the expensive allowed imports at STARTUP, not inside a timed call.

    `sympy` is the only allowed root that costs real time to import: measured
    1,491 ms on the first call in a fresh worker against 1.1 ms once cached.
    The worker is shared and respawned after every kill, so a lazily imported
    sympy lands inside whichever program happens to be the first sympy user
    after a respawn -- and under the tight steady-state budget the client now
    uses, that program would be killed as a runaway, respawning the worker and
    putting the next sympy program in exactly the same position. Importing here
    moves the cost under the client's one-off cold-start budget instead.
    """
    try:
        import sympy  # noqa: F401
    except Exception:
        # Not installed -> `guarded_import` will raise for the program that
        # actually wants it, which is the pre-existing behaviour.
        pass


def main() -> None:
    try:
        resource.setrlimit(resource.RLIMIT_AS, (_MEM_LIMIT_BYTES, _MEM_LIMIT_BYTES))
    except Exception:
        pass

    _warm_imports()

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
            else:
                resp["error"] = "generate returned no usable (problem, answer)"
        except BaseException as exc:
            # Carry the failure back. Collapsing every failure into a bare
            # ok=False made "execute failed at seed=0" the single largest
            # rejection reason in a run with 58% of candidates dying here,
            # with no way to tell an AssertionError (the child's own
            # cross-check catching a real problem/answer mismatch) from a
            # NameError (broken code) or a guarded import.
            resp = {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[:300],
            }
        protocol.write(json.dumps(resp) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
