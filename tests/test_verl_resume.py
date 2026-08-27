"""Fail-closed coupling between verl checkpoint resume and MAP resume."""

from types import SimpleNamespace

import pytest

from rq_evolve.archive import ArchiveSchemaError
from rq_evolve.verl_adapter import VerlTrainerAdapter


class _Evolver:
    def __init__(self, *, load_result=False, load_error=None):
        self.load_result = load_result
        self.load_error = load_error
        self.loaded = []
        self.saved = []
        self.archive = SimpleNamespace(champions=lambda: [])
        self.dataset = SimpleNamespace(snapshot=lambda: [object()])

    def load_state(self, directory):
        self.loaded.append(directory)
        if self.load_error is not None:
            raise self.load_error
        return self.load_result

    def save_state(self, directory):
        self.saved.append(directory)


def _adapter(bootstrapped):
    adapter = VerlTrainerAdapter.__new__(VerlTrainerAdapter)
    adapter._bootstrap_seed_archive = lambda evolver: bootstrapped.append(evolver)
    return adapter


def _context(tmp_path, evolver, resume_mode):
    archive_dir = tmp_path / "rq_archive"
    return {
        "evolver": evolver,
        "archive_dir": archive_dir,
        "train_sampler": SimpleNamespace(epoch=0),
        "verl_config": SimpleNamespace(
            trainer={"resume_mode": resume_mode},
        ),
    }


def test_resume_disabled_bootstraps_without_reading_archive(tmp_path):
    bootstrapped = []
    evolver = _Evolver(load_error=AssertionError("load must not be called"))
    context = _context(tmp_path, evolver, "disable")

    _adapter(bootstrapped)._resume_or_bootstrap(context)

    assert evolver.loaded == []
    assert bootstrapped == [evolver]
    assert evolver.saved == [context["archive_dir"]]


def test_resume_disabled_refuses_existing_archive_snapshot(tmp_path):
    bootstrapped = []
    evolver = _Evolver()
    context = _context(tmp_path, evolver, "disable")
    context["archive_dir"].mkdir()
    (context["archive_dir"] / "archive.json").write_text("{}", encoding="utf-8")

    with pytest.raises(RuntimeError, match="requires a clean rq_archive"):
        _adapter(bootstrapped)._resume_or_bootstrap(context)

    assert evolver.loaded == []
    assert evolver.saved == []
    assert bootstrapped == []


def test_resume_schema_error_is_propagated_not_bootstrapped(tmp_path):
    bootstrapped = []
    evolver = _Evolver(
        load_error=ArchiveSchemaError("incompatible archive schema")
    )
    context = _context(tmp_path, evolver, "auto")

    with pytest.raises(ArchiveSchemaError, match="incompatible archive"):
        _adapter(bootstrapped)._resume_or_bootstrap(context)

    assert len(evolver.loaded) == 1
    assert evolver.saved == []
    assert bootstrapped == []
