"""verl-compatible reward function for boxed math answers."""

from __future__ import annotations

import threading


def _ensure_math_verify_thread_safe() -> None:
    """Make math_verify's timeout actually enforce a limit off the main thread.

    ``math_verify.parse``/``verify`` wrap work in a ``signal.SIGALRM`` timeout,
    but ``signal`` only works on the main thread. In a verl reward worker (a
    non-main thread) every call raises ``ValueError: signal only works in main
    thread``. A naive "just drop the timeout on worker threads" fix is WORSE: a
    pathological model answer (e.g. one that parses to a relational expr) makes
    sympy ``solve()`` spin effectively forever, pegging one reward worker at 100%
    CPU with the GPU idle and the whole run wedged.

    So off the main thread we enforce the same wall-clock budget with a daemon
    watchdog thread: run the work in a daemon thread, ``join`` for
    ``timeout_seconds``, and raise ``TimeoutException`` (which math_verify already
    handles -> graded as non-match) if it overruns. A hung call leaks one daemon
    thread (Python can't kill a thread stuck in C-level sympy) but the grader
    returns and the run keeps moving; the length guard in ``answers_match`` keeps
    such leaks rare. Idempotent and process-global.
    """
    from math_verify import utils as _mv_utils

    if getattr(_mv_utils.timeout, "_rq_safe", False):
        return
    _orig = _mv_utils.timeout

    def _safe_timeout(timeout_seconds: int = 10):
        if threading.current_thread() is threading.main_thread():
            return _orig(timeout_seconds)

        def decorator(func):
            def wrapper(*args, **kwargs):
                from math_verify.errors import TimeoutException

                box: dict = {}

                def run():
                    try:
                        box["r"] = func(*args, **kwargs)
                    except BaseException as exc:  # propagate to caller
                        box["e"] = exc

                th = threading.Thread(target=run, daemon=True)
                th.start()
                th.join(timeout_seconds)
                if th.is_alive():
                    raise TimeoutException("math_verify timed out (worker thread)")
                if "e" in box:
                    raise box["e"]
                return box.get("r")

            return wrapper

        return decorator

    _safe_timeout._rq_safe = True
    _mv_utils.timeout = _safe_timeout
    # parser.py / grader.py did ``from .utils import timeout`` at import, so the
    # name is already bound in those modules -- rebind there too.
    import math_verify.grader as _g
    import math_verify.parser as _p

    _p.timeout = _safe_timeout
    _g.timeout = _safe_timeout


def extract_boxed(text: str) -> str | None:
    r"""Return the content of the LAST complete ``\boxed{...}`` in ``text``.

    Brace-matched (not regex) so arbitrarily nested answers extract correctly --
    e.g. ``\boxed{\frac{\sqrt{3}}{2}}`` -> ``\frac{\sqrt{3}}{2}``. The old
    single-level regex returned None for any answer with 2+ nested brace groups,
    which silently scored common MATH-500 answers as wrong.
    """
    text = text or ""
    token = r"\boxed"
    result: str | None = None
    pos = 0
    while True:
        idx = text.find(token, pos)
        if idx == -1:
            break
        i = idx + len(token)
        while i < len(text) and text[i].isspace():
            i += 1
        if i < len(text) and text[i] == "{":
            depth = 0
            for j in range(i, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        result = text[i + 1 : j].strip()
                        break
        pos = idx + len(token)
    return result


def normalize_answer(text: str) -> str:
    text = str(text).strip().lower()
    for left, right in [("{", "}"), ("[", "]"), ("(", ")")]:
        if text.startswith(left) and text.endswith(right):
            text = text[1:-1].strip()
    return text.rstrip(".").replace(",", "").replace(" ", "")


def answers_match(predicted: str, ground_truth: str) -> bool:
    """Answer equality via ``math_verify`` (R-Zero parity), no fallback.

    ``math_verify`` is a hard dependency: the import is intentionally outside the
    try/except so a missing install fails loud instead of silently degrading to a
    weaker grader. Parse/verify failures on a given pair count as a non-match.
    """
    pred_s, gold_s = str(predicted), str(ground_truth)
    # Guard: a clean competition answer is short. An over-long prediction is a
    # junk blob (run-on expression, pasted reasoning) that only feeds sympy's
    # expensive solve()/simplify() -- skip math_verify and fall back to a cheap
    # normalized string check so it can never wedge a reward worker.
    if len(pred_s) > 200 or len(gold_s) > 200:
        return normalize_answer(pred_s) == normalize_answer(gold_s)

    _ensure_math_verify_thread_safe()
    from math_verify import parse, verify

    try:
        # Wrap both sides in \boxed{} so math_verify's LaTeX extraction triggers.
        # Parsing a bare fragment (e.g. "\dfrac{1}{2}" or "\frac34") otherwise
        # fails its extractor and reports a false non-match; the \boxed form makes
        # \dfrac/\frac, fraction/decimal, and spacing equivalences all resolve.
        # verify(gold, target): ground truth first, prediction second.
        gold = parse("\\boxed{" + str(ground_truth) + "}")
        pred = parse("\\boxed{" + str(predicted) + "}")
        return bool(verify(gold, pred))
    except Exception:
        return False


def compute_score(
    data_source=None,
    solution_str=None,
    ground_truth=None,
    extra_info=None,
    *,
    response_str_list: list[str] | None = None,
    ground_truth_list: list[str] | None = None,
    data_sources=None,
    solution_strs: list[str] | None = None,
    ground_truths: list[str] | None = None,
    extra_infos=None,
    **kwargs,
) -> dict | list[dict]:
    """Reward function compatible with recent and legacy verl calls.

    Recent verl reward managers call:
    ``compute_score(data_source, solution_str, ground_truth, extra_info)``.
    Older/batch integrations may call ``compute_score(responses, truths)`` or
    pass ``solution_strs`` / ``ground_truths`` keyword lists.
    """
    if _looks_like_batch(data_source) and _looks_like_batch(solution_str) and ground_truth is None:
        response_str_list = list(data_source)
        ground_truth_list = list(solution_str)
        data_source = None
        solution_str = None

    responses = solution_strs if solution_strs is not None else response_str_list
    truths = ground_truths if ground_truths is not None else ground_truth_list
    if responses is not None or truths is not None:
        if responses is None or truths is None:
            raise ValueError("batch compute_score requires responses and ground truths")
        infos = extra_infos if extra_infos is not None else [None] * len(responses)
        return [
            _skipped_score() if _is_skip(info) else _score_one(response, truth)
            for response, truth, info in zip(responses, truths, infos)
        ]

    # Eval rows (math-benchmark val set) carry a skip sentinel: the trainer
    # re-grades them on the main thread, so the worker thread does no sympy work.
    if _is_skip(extra_info):
        return _skipped_score()

    if solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires solution_str and ground_truth")
    return _score_one(solution_str, ground_truth)


def _looks_like_batch(value) -> bool:
    return isinstance(value, (list, tuple))


def _score_one(response: str, ground_truth: str) -> dict:
    predicted = extract_boxed(response)
    correct = predicted is not None and answers_match(predicted, ground_truth)
    return {
        "score": 1.0 if correct else 0.0,
        "overall": 1.0 if correct else 0.0,
        "accuracy": 1.0 if correct else 0.0,
        "format": 1.0 if predicted is not None else 0.0,
    }


# Sentinel placed by the math-benchmark val dataset (math_eval.MathBenchmarkDataset)
# in each row's ``extra_info``. The agent loop's reward worker runs compute_score
# in a non-main thread (NaiveRewardManager -> loop.run_in_executor); grading the
# base model's pathological boxed outputs there pegs CPU and stalls vLLM
# generation -> GPU 0% mid-eval (signal.SIGALRM can't fire off the main thread, so
# the watchdog only leaks daemon threads). For eval rows we skip the worker grade
# entirely and re-grade on the trainer's MAIN thread in RQValidatingTrainer._validate
# (where math_verify's native SIGALRM timeout works). See eval_trainer.py.
SKIP_WORKER_GRADE_KEY = "skip_worker_grade"


def _skipped_score() -> dict:
    """Placeholder reward for eval rows graded later on the main thread."""
    return {"score": 0.0, "overall": 0.0, "accuracy": 0.0, "format": 0.0, "skipped": 1.0}


def _is_skip(extra_info) -> bool:
    return isinstance(extra_info, dict) and bool(extra_info.get(SKIP_WORKER_GRADE_KEY))
