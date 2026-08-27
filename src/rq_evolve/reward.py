import json
import logging
import os
import queue
import select
import subprocess
import sys
import threading
from pathlib import Path

import math_verify.grader as _g
import math_verify.parser as _p
from math_verify import parse, verify
from math_verify import utils as _mv_utils
from math_verify.errors import TimeoutException

# verl loads custom reward files by filesystem path, outside normal package
# import semantics.  Absolute import works in installed/editable runs; the
# sibling fallback keeps the path-loader contract working as well.
try:  # pragma: no branch - exactly one branch is used per loader style
    from rq_evolve.verifier import normalize_verifier
except ImportError:  # loaded directly from ``src/rq_evolve/reward.py``
    import importlib.util as _importlib_util

    _verifier_spec = _importlib_util.spec_from_file_location(
        "rq_evolve_verifier_standalone", Path(__file__).with_name("verifier.py")
    )
    if _verifier_spec is None or _verifier_spec.loader is None:
        raise
    _verifier_module = _importlib_util.module_from_spec(_verifier_spec)
    _verifier_spec.loader.exec_module(_verifier_module)
    normalize_verifier = _verifier_module.normalize_verifier


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
    if getattr(_mv_utils.timeout, "_rq_safe", False):
        return
    _orig = _mv_utils.timeout

    def _safe_timeout(timeout_seconds: int = 10):
        if threading.current_thread() is threading.main_thread():
            return _orig(timeout_seconds)

        def decorator(func):
            def wrapper(*args, **kwargs):
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
    _p.timeout = _safe_timeout
    _g.timeout = _safe_timeout


# --------------------------------------------------------------------------
# Out-of-process grading
# --------------------------------------------------------------------------
_log = logging.getLogger(__name__)

_GRADER_WORKER_PATH = Path(__file__).with_name("_grader_worker.py")
# Budgets. The steady-state one is the grading budget math_verify itself
# defaults to; the cold one additionally covers interpreter start and the
# worker's math_verify/sympy import.
_GRADE_TIMEOUT = float(os.environ.get("RQ_GRADE_TIMEOUT", "10"))
_GRADE_COLD_TIMEOUT = float(os.environ.get("RQ_GRADE_COLD_TIMEOUT", "30"))
# One worker per verify thread (async_rollout's pool is 4) so a single wedged
# comparison cannot hold up the others. Each worker is an interpreter with
# sympy resident, so this is not free -- keep it near the verify fan-out.
_GRADE_WORKERS = int(os.environ.get("RQ_GRADE_WORKERS", "4"))


class _GraderClient:
    """One persistent subprocess that grades under a hard kill.

    Same shape, and the same reason, as ``program._SandboxClient``: a runaway
    comparison cannot be stopped from inside the process running it, so it runs
    somewhere killable. ``parse("\\boxed{51!!}")`` is ``factorial(factorial(51))``
    -- the factorial of a 67-digit number -- and neither a signal nor a thread
    join can end that. SIGKILL can.
    """

    def __init__(self) -> None:
        self._proc: subprocess.Popen | None = None
        self._warm = False

    def _spawn(self) -> None:
        self._proc = subprocess.Popen(
            [sys.executable, str(_GRADER_WORKER_PATH)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
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

    def grade(self, pred: str, gold: str, verifier=None) -> tuple[bool, str | None]:
        """``(match, failure_kind)``; ``failure_kind`` is None on a real verdict."""
        try:
            if self._proc is None or self._proc.poll() is not None:
                self._spawn()
            budget = _GRADE_TIMEOUT if self._warm else max(_GRADE_TIMEOUT, _GRADE_COLD_TIMEOUT)
            self._proc.stdin.write(
                json.dumps({"pred": pred, "gold": gold, "verifier": verifier})
                + "\n"
            )
            self._proc.stdin.flush()
        except Exception:
            self._kill()
            return False, "write_error"

        ready, _, _ = select.select([self._proc.stdout], [], [], budget)
        if not ready:
            # Wedged in work that yields to nothing. Kill it; the next call on
            # this client respawns a clean worker.
            self._kill()
            return False, "timeout"
        self._warm = True
        line = self._proc.stdout.readline()
        if not line:
            self._kill()
            return False, "worker_died"
        try:
            out = json.loads(line)
        except Exception:
            self._kill()
            return False, "protocol_error"
        if not out.get("ok"):
            # parse/verify raised inside the worker -> non-match, same as the
            # in-process path's `except Exception: return False`.
            return False, None
        return bool(out.get("match")), None


class _GraderPool:
    """Fixed pool of grader subprocesses, checked out one per call."""

    def __init__(self, size: int) -> None:
        self._q: queue.Queue = queue.Queue()
        for _ in range(max(1, size)):
            self._q.put(_GraderClient())
        self._lock = threading.Lock()
        self.counters = {"timeout": 0, "worker_died": 0,
                         "write_error": 0, "protocol_error": 0, "graded": 0}

    def grade(self, pred: str, gold: str, verifier=None) -> tuple[bool, str | None]:
        client = self._q.get()
        try:
            match, kind = client.grade(pred, gold, verifier)
        finally:
            self._q.put(client)
        with self._lock:
            self.counters["graded"] += 1
            if kind:
                self.counters[kind] = self.counters.get(kind, 0) + 1
        return match, kind

    def stats(self) -> dict:
        with self._lock:
            return dict(self.counters)


_GRADERS = _GraderPool(_GRADE_WORKERS)


def grader_stats() -> dict:
    """Counters for the iteration metrics: how often grading had to be killed."""
    return _GRADERS.stats()


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


def answers_match(predicted: str, ground_truth: str, verifier=None) -> bool:
    """Grade one extracted answer under a declarative verifier contract.

    With ``verifier=None`` this is byte-for-byte the legacy expression path:
    :mod:`math_verify` is a hard dependency and parse/verify failures count as a
    non-match.  Boolean, finite ``one_of``, and complete finite-set contracts are
    dispatched in the same killable worker.  Invalid verifier data fails closed;
    generated code is never executed by the grader.
    """
    pred_s, gold_s = str(predicted), str(ground_truth)
    try:
        spec = normalize_verifier(verifier, answer=gold_s)
    except (TypeError, ValueError):
        return False
    # Cheap pre-filter, NOT a safety guard. An over-long prediction is a junk
    # blob (run-on expression, pasted reasoning) whose only effect is to feed
    # sympy expensive work, and a normalized string check settles it for free.
    #
    # It was load-bearing once and should not be again: length is a poor proxy
    # for cost. The comparison that wedged a run was `51!!`, four characters,
    # which math_verify parses as factorial(factorial(51)) -- the factorial of a
    # 67-digit number. No length threshold catches that. The kill budget below
    # is what makes cost bounded; this line only saves time.
    if spec["mode"] == "expression" and (
        len(pred_s) > 200 or len(gold_s) > 200
    ):
        return normalize_answer(pred_s) == normalize_answer(gold_s)

    # Graded in a subprocess the parent can SIGKILL. The in-process path this
    # replaces ran math_verify in a daemon thread and abandoned it on timeout;
    # Python cannot stop a thread, so the abandoned one kept a core busy for the
    # life of the process -- two of them, in one actor and the driver, took a
    # run to 0% GPU. A timeout here grades as non-match, exactly as a parse or
    # verify failure always has.
    match, kind = _GRADERS.grade(pred_s, gold_s, spec)
    if kind == "timeout":
        _log.warning(
            "grader killed after %.0fs: pred=%r gold=%r (graded as non-match)",
            _GRADE_TIMEOUT, pred_s[:80], gold_s[:80],
        )
    return match


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
            _skipped_score()
            if _is_skip(info)
            else _score_one(response, truth, info)
            for response, truth, info in zip(responses, truths, infos)
        ]

    # Eval rows (math-benchmark val set) carry a skip sentinel: the trainer
    # re-grades them on the main thread, so the worker thread does no sympy work.
    if _is_skip(extra_info):
        return _skipped_score()

    if solution_str is None or ground_truth is None:
        raise ValueError("compute_score requires solution_str and ground_truth")
    return _score_one(solution_str, ground_truth, extra_info)


def _looks_like_batch(value) -> bool:
    return isinstance(value, (list, tuple))


def _score_one(response: str, ground_truth: str, extra_info=None) -> dict:
    predicted = extract_boxed(response)
    verifier = (
        extra_info.get("verifier") if isinstance(extra_info, dict) else None
    )
    correct = predicted is not None and answers_match(
        predicted, ground_truth, verifier
    )
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
