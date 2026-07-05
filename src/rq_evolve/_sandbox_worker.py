import importlib.util
import json
import math
import os
import random
import resource
import sys

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
