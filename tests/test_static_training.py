import json
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from rq_evolve.config import RQEvolveConfig
from rq_evolve.dataset import (
    DynamicProblemDataset,
    VerlDynamicDataset,
    load_static_training_jsonl,
    validate_static_training_schedule,
)
from rq_evolve.verl_adapter import StaticTrainingSampler
from rq_evolve.verl_adapter import VerlAdapterConfig, VerlTrainerAdapter


class _Tokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        # Stable fake: system/user framing plus one ID per whitespace token.
        return [0, 1] + [
            index
            for index, _ in enumerate(messages[1]["content"].split(), start=2)
        ]

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(range(len(str(text).split())))


def _row(sample_id: str = "sample-1", problem: str = "What is 1 + 1?"):
    return {
        "sample_id": sample_id,
        "independent_run_id": "run-1",
        "parent_program_id": "parent-1",
        "generator_unit_id": "generator-1",
        "condition": "plain",
        "problem": problem,
        "answer": "2",
    }


def _write_jsonl(path, rows):
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_static_jsonl_normalizes_aliases_and_recomputes_tokens(tmp_path):
    path = tmp_path / "plain.jsonl"
    row = _row()
    row["prompt_token_count"] = 7
    _write_jsonl(path, [row])

    rows, report = load_static_training_jsonl(
        path,
        _Tokenizer(),
        condition="plain",
    )

    assert rows[0]["parent_id"] == "parent-1"
    assert rows[0]["generator_id"] == "generator-1"
    assert rows[0]["prompt_token_count"] == 7
    assert rows[0]["reference_answer_token_count"] == 1
    assert rows[0]["token_count"] == 7
    assert report["source_rows"] == 1
    assert report["token_count"] == 7
    assert len(report["source_sha256"]) == 64


def test_static_jsonl_rejects_tokenizer_mismatch_and_duplicate_instances(tmp_path):
    mismatch = tmp_path / "mismatch.jsonl"
    row = _row()
    row["prompt_token_count"] = 999
    _write_jsonl(mismatch, [row])
    with pytest.raises(ValueError, match="training tokenizer computes 7"):
        load_static_training_jsonl(mismatch, _Tokenizer(), condition="plain")

    duplicate = tmp_path / "duplicate.jsonl"
    _write_jsonl(
        duplicate,
        [_row("sample-1"), _row("sample-2")],
    )
    with pytest.raises(ValueError, match=r"duplicate \(problem, answer\)"):
        load_static_training_jsonl(duplicate, _Tokenizer(), condition="plain")


def test_static_schedule_requires_exact_complete_exposure():
    report = {
        "source_rows": 4,
        "source_sha256": "abc",
        "prompt_tokens": 20,
        "max_prompt_tokens": 6,
        "reference_answer_tokens": 4,
        "token_count": 20,
    }
    schedule = validate_static_training_schedule(
        report,
        batch_size=2,
        total_training_steps=4,
        trainer_total_epochs=2,
        static_epochs=2,
        expected_rows=4,
        expected_tokens=20,
        require_expected=True,
        max_prompt_length=8,
    )
    assert schedule["schedule_valid"] is True
    assert schedule["row_exposures"] == 8
    assert schedule["token_exposures"] == 40

    with pytest.raises(ValueError, match="exact static exposure requires"):
        validate_static_training_schedule(
            report,
            batch_size=2,
            total_training_steps=5,
            trainer_total_epochs=3,
            static_epochs=2,
            expected_rows=4,
            expected_tokens=20,
            require_expected=True,
        )
    with pytest.raises(ValueError, match="must exactly equal"):
        validate_static_training_schedule(
            report,
            batch_size=2,
            total_training_steps=4,
            trainer_total_epochs=3,
            static_epochs=2,
            expected_rows=4,
            expected_tokens=20,
            require_expected=True,
        )


def test_static_schedule_audit_returns_issues_without_expected_pins():
    report = {
        "source_rows": 3,
        "source_sha256": "abc",
        "prompt_tokens": 15,
        "max_prompt_tokens": 5,
        "reference_answer_tokens": 3,
        "token_count": 15,
    }
    schedule = validate_static_training_schedule(
        report,
        batch_size=2,
        total_training_steps=1,
        trainer_total_epochs=1,
        static_epochs=1,
        expected_rows=None,
        expected_tokens=None,
        require_expected=False,
        raise_on_error=False,
    )
    assert schedule["schedule_valid"] is False
    assert any("drop_last" in issue for issue in schedule["issues"])
    assert not any("static_expected" in issue for issue in schedule["issues"])


def test_static_sampler_is_finite_and_checkpoint_pins_file_hash():
    dynamic = DynamicProblemDataset(
        [
            {"problem": "one", "answer": "1"},
            {"problem": "two", "answer": "2"},
        ]
    )
    dataset = VerlDynamicDataset(dynamic, _Tokenizer(), min_size=1)
    sampler = StaticTrainingSampler(
        dataset,
        source_sha256="same-file",
        epochs=1,
        shuffle=False,
    )
    assert list(iter(sampler)) == [0, 1]
    with pytest.raises(RuntimeError, match="data exhausted"):
        iter(sampler)

    resumed = StaticTrainingSampler(
        dataset,
        source_sha256="different-file",
        epochs=1,
        shuffle=False,
    )
    with pytest.raises(RuntimeError, match="JSONL changed"):
        resumed.load_state_dict(sampler.state_dict())


def test_static_config_requires_condition_and_rejects_orphan_static_fields():
    with pytest.raises(ValueError, match="static_condition"):
        RQEvolveConfig.from_dict(
            {"training_data": {"static_training_jsonl": "plain.jsonl"}}
        )
    with pytest.raises(ValueError, match="static_training_jsonl is required"):
        RQEvolveConfig.from_dict(
            {"training_data": {"static_expected_rows": 10}}
        )


def test_static_adapter_setup_never_constructs_evolver_or_archive(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "plain.jsonl"
    _write_jsonl(data_path, [_row()])
    rq_config = RQEvolveConfig.from_dict(
        {
            "evolution": {
                "use_evaluator": True,
                "evaluator_provider": "openai",
            },
            "training_data": {
                "static_training_jsonl": str(data_path),
                "static_condition": "plain",
                "static_expected_rows": 1,
                "static_expected_tokens": 7,
                "static_epochs": 1,
            },
        }
    )
    adapter = VerlTrainerAdapter(
        VerlAdapterConfig(inline_config={}),
        rq_config,
        project_root=tmp_path,
    )
    verl_config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 1,
                "val_batch_size": 1,
                "max_prompt_length": 32,
                "truncation": "left",
                "shuffle": False,
                "seed": 1,
            },
            "trainer": {
                "total_training_steps": 1,
                "total_epochs": 1,
                "default_local_dir": str(tmp_path / "checkpoints"),
                "resume_mode": "disable",
            },
            "actor_rollout_ref": {"model": {"path": "unused"}},
            "ray_init": {},
        }
    )

    fake_ray = SimpleNamespace(
        is_initialized=lambda: True,
        init=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("Ray should already be treated as initialized")
        ),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(adapter, "assert_verl_available", lambda: None)
    monkeypatch.setattr(adapter, "_load_verl_config", lambda: verl_config)
    monkeypatch.setattr(adapter, "_patch_reward_config", lambda config: None)
    monkeypatch.setattr(adapter, "_apply_lora_config", lambda config: None)
    monkeypatch.setattr(
        adapter,
        "_build_tokenizer_and_processor",
        lambda config: (_Tokenizer(), None),
    )
    monkeypatch.setattr(
        adapter,
        "_build_evolver",
        lambda backend: (_ for _ in ()).throw(
            AssertionError("static setup must not build an Evolver")
        ),
    )

    class _Trainer:
        def init_workers(self):
            return None

    monkeypatch.setattr(adapter, "_build_trainer", lambda **kwargs: _Trainer())

    context = adapter._setup()

    assert context["static_training"] is True
    assert context["evolver"] is None
    assert context["backend"] is None
    assert context["archive_dir"] is None
    assert isinstance(context["train_sampler"], StaticTrainingSampler)


def test_invalid_static_data_fails_before_ray_initialization(tmp_path, monkeypatch):
    data_path = tmp_path / "plain.jsonl"
    _write_jsonl(data_path, [_row()])
    rq_config = RQEvolveConfig.from_dict(
        {
            "training_data": {
                "static_training_jsonl": str(data_path),
                "static_condition": "plain",
                "static_expected_rows": 2,
                "static_expected_tokens": 7,
                "static_epochs": 1,
            },
        }
    )
    adapter = VerlTrainerAdapter(
        VerlAdapterConfig(inline_config={}),
        rq_config,
        project_root=tmp_path,
    )
    verl_config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 1,
                "val_batch_size": 1,
                "max_prompt_length": 32,
            },
            "trainer": {
                "total_training_steps": 1,
                "total_epochs": 1,
                "resume_mode": "disable",
            },
            "actor_rollout_ref": {"model": {"path": "unused"}},
            "ray_init": {},
        }
    )
    ray_started = []
    fake_ray = SimpleNamespace(
        is_initialized=lambda: False,
        init=lambda **kwargs: ray_started.append(kwargs),
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(adapter, "assert_verl_available", lambda: None)
    monkeypatch.setattr(adapter, "_load_verl_config", lambda: verl_config)
    monkeypatch.setattr(adapter, "_patch_reward_config", lambda config: None)
    monkeypatch.setattr(adapter, "_apply_lora_config", lambda config: None)
    monkeypatch.setattr(
        adapter,
        "_build_tokenizer_and_processor",
        lambda config: (_Tokenizer(), None),
    )

    with pytest.raises(ValueError, match="static_expected_rows=2"):
        adapter._setup()

    assert ray_started == []


def test_static_training_rejects_resume_before_ray_initialization(
    tmp_path, monkeypatch
):
    data_path = tmp_path / "plain.jsonl"
    _write_jsonl(data_path, [_row()])
    rq_config = RQEvolveConfig.from_dict(
        {
            "training_data": {
                "static_training_jsonl": str(data_path),
                "static_condition": "plain",
                "static_expected_rows": 1,
                "static_expected_tokens": 7,
                "static_epochs": 1,
            },
        }
    )
    adapter = VerlTrainerAdapter(
        VerlAdapterConfig(inline_config={}),
        rq_config,
        project_root=tmp_path,
    )
    verl_config = OmegaConf.create(
        {
            "data": {
                "train_batch_size": 1,
                "val_batch_size": 1,
                "max_prompt_length": 32,
            },
            "trainer": {
                "total_training_steps": 1,
                "total_epochs": 1,
                "resume_mode": "auto",
                "default_local_dir": str(tmp_path / "shared-output"),
            },
            "actor_rollout_ref": {
                "model": {"path": "/models/declared-base"}
            },
            "ray_init": {},
        }
    )
    ray_started = []
    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(
            is_initialized=lambda: False,
            init=lambda **kwargs: ray_started.append(kwargs),
        ),
    )
    monkeypatch.setattr(adapter, "assert_verl_available", lambda: None)
    monkeypatch.setattr(adapter, "_load_verl_config", lambda: verl_config)
    monkeypatch.setattr(adapter, "_patch_reward_config", lambda config: None)
    monkeypatch.setattr(adapter, "_apply_lora_config", lambda config: None)
    monkeypatch.setattr(
        adapter,
        "_build_tokenizer_and_processor",
        lambda config: (_Tokenizer(), None),
    )

    with pytest.raises(ValueError, match="resume_mode must be 'disable'"):
        adapter._setup()

    assert ray_started == []
