"""Compatibility checks for the verl 0.7 legacy and 0.9 unified workers."""

from types import SimpleNamespace

from rq_evolve import verl_adapter
from rq_evolve.verl_backend import VerlPolicyBackend


def _config(*, estimator="rloo", use_kl_loss=True, lora_rank=0):
    return SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(
            actor={"strategy": "fsdp", "use_kl_loss": use_kl_loss},
            model={"lora_rank": lora_rank},
        ),
        algorithm={"adv_estimator": estimator, "use_kl_in_reward": False},
        critic={"enable": None},
    )


def test_selects_verl09_unified_workers(monkeypatch):
    actor_cls = type("ActorRolloutRefWorker", (), {})
    critic_cls = type("TrainingWorker", (), {})
    module = SimpleNamespace(
        ActorRolloutRefWorker=actor_cls,
        TrainingWorker=critic_cls,
    )
    monkeypatch.setattr(verl_adapter.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(
        verl_adapter.importlib,
        "import_module",
        lambda name: module if name == "verl.workers.engine_workers" else None,
    )

    selected = verl_adapter._select_worker_classes(
        _config(), default_ray_worker_group_cls="ray-wg"
    )

    assert selected == (actor_cls, None, None, "ray-wg", True)


def test_selects_verl07_legacy_workers(monkeypatch):
    actor_cls = type("ActorRolloutRefWorker", (), {})
    async_actor_cls = type("AsyncActorRolloutRefWorker", (), {})
    module = SimpleNamespace(
        ActorRolloutRefWorker=actor_cls,
        AsyncActorRolloutRefWorker=async_actor_cls,
    )
    monkeypatch.setattr(
        verl_adapter.importlib.util, "find_spec", lambda name: object()
    )
    monkeypatch.setattr(
        verl_adapter.importlib,
        "import_module",
        lambda name: module if name == "verl.workers.fsdp_workers" else None,
    )

    selected = verl_adapter._select_worker_classes(
        _config(), default_ray_worker_group_cls="ray-wg"
    )

    assert selected == (async_actor_cls, None, None, "ray-wg", False)


def test_verl09_actor_role_fuses_reference_policy():
    roles = SimpleNamespace(ActorRollout="actor", ActorRolloutRef="actor-ref")

    assert (
        verl_adapter._select_actor_role(
            roles, _config(use_kl_loss=True), unified_engine_workers=True
        )
        == "actor-ref"
    )
    assert (
        verl_adapter._select_actor_role(
            roles,
            _config(use_kl_loss=True, lora_rank=8),
            unified_engine_workers=True,
        )
        == "actor"
    )


def test_entropy_uses_trainer_dataproto_conversion():
    expected = object()

    class Trainer:
        def _compute_old_log_prob(self, batch):
            assert batch == "padded"
            return expected, {"mfu": 0.0}

    backend = VerlPolicyBackend(trainer=Trainer())

    assert backend._compute_actor_log_probs("padded") is expected
