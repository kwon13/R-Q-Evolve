"""Hermetic answer grader: one request per stdin line, one verdict per stdout line.

Runs `math_verify` in a process the parent can KILL. The in-process watchdog it
replaces could not: `reward.py` ran the comparison in a daemon thread and gave up
on it after a timeout, but Python cannot stop a thread, so the abandoned one kept
burning a core for as long as the process lived. The trigger found in the wild is
four characters:

    parse("\\boxed{51!!}") -> factorial(factorial(51))

`51!` is a 67-digit number and sympy then tries to take ITS factorial. That never
finishes, and no length guard can catch it -- the input is 4 bytes with unbounded
cost. A separate process can simply be SIGKILLed.

Grading runs on this worker's MAIN thread, so `math_verify`'s own SIGALRM budget
works natively here and stops anything that yields to the interpreter. The
parent's kill is the second layer, for work that never yields at all.
"""
import json
import sys


def _grade(pred: str, gold: str) -> bool:
    from math_verify import parse, verify

    # \boxed-wrapped on both sides, matching reward.answers_match: a bare
    # fragment ("\dfrac{1}{2}", "\frac34") misses math_verify's extractor and
    # reports a false non-match.
    g = parse("\\boxed{" + str(gold) + "}")
    p = parse("\\boxed{" + str(pred) + "}")
    return bool(verify(g, p))


def _warm() -> None:
    """Pay the math_verify/sympy import (~1.5 s) before the first request."""
    try:
        from math_verify import parse  # noqa: F401
        _grade("1", "1")
    except Exception:
        pass


def main() -> None:
    _warm()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            out = {"ok": True, "match": _grade(req["pred"], req["gold"])}
        except Exception as exc:  # parse/verify failure grades as non-match
            out = {"ok": False, "error": f"{type(exc).__name__}: {str(exc)[:150]}"}
        sys.stdout.write(json.dumps(out) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
