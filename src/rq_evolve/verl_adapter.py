import importlib
import importlib.metadata as metadata
import importlib.util
import inspect
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .archive import MAPElitesArchive
from .config import RQEvolveConfig
from .dataset import VerlDynamicDataset
from .evolution import RQEvolver
from .verl_backend import VerlPolicyBackend


@dataclass(slots=True)
class VerlAdapterConfig:
    # Either config_path (separate yaml) or inline_config (embedded
    # `verl_config:` block in the rq_evolve yaml) must be provided. Inline
    # takes precedence when both are set.
    config_path: str | None = None
    reward_function: str = "./src/rq_evolve/reward.py:compute_score"
    inline_config: Any = None


class EvolvingSampler:
    """Sampler that runs R_Q evolution at the start of each verl epoch."""

    def __init__(
        self,
        dataset: VerlDynamicDataset,
        evolver: RQEvolver,
        *,
        shuffle: bool = True,
        seed: int = 1,
        evolve_on_first_epoch: bool = True,
        archive_dir: str | Path | None = None,
    ) -> None:
        self.dataset = dataset
        self.evolver = evolver
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.evolve_on_first_epoch = bool(evolve_on_first_epoch)
        self.archive_dir = archive_dir
        self.epoch = 0

    def __iter__(self) -> Iterator[int]:
        if self.epoch > 0 or self.evolve_on_first_epoch:
            metrics = self.evolver.run_outer_iteration(self.epoch)
            print(f"[RQ-Evolve] outer iteration {self.epoch}: {metrics}")
            # Persist after every evolution so a restart resumes from the evolved
            # grid, not seeds (the verl weight checkpoint excludes the archive).
            #  - archive.json: latest snapshot (resume) + archive_iter{N}.json:
            #    per-step version, so the evolution trajectory is recoverable.
            #  - evolution_log.jsonl: append-only per-iteration metrics + every
            #    candidate report (inserted/rejected/why, rq_scores).
            if self.archive_dir is not None:
                self.evolver.save_state(self.archive_dir, iteration=self.epoch)
                self.evolver.append_evolution_log(self.archive_dir, self.epoch, metrics)
            self._log_evolve_metrics_to_wandb(metrics)

        n = len(self.dataset)
        if self.shuffle:
            import torch

            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(n, generator=generator).tolist()
        else:
            indices = list(range(n))
        self.epoch += 1
        return iter(indices)

    def _log_evolve_metrics_to_wandb(self, metrics: dict) -> None:
        """Best-effort: send the evolve metrics to wandb (was stdout-only).

        commit=False merges them into the next training-step commit instead of
        advancing wandb's step counter, avoiding "step must increase" conflicts
        with verl's own logging. Wrapped in try/except so logging never breaks
        training.
        """
        try:
            import wandb

            if wandb.run is None:
                return
            payload = {
                f"evolve/{k}": v
                for k, v in metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
            if payload:
                wandb.log(payload, commit=False)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.dataset)


class VerlTrainerAdapter:
    """Wire R_Q-Evolve into the installed verl PPO/GRPO trainer."""

    def __init__(
        self,
        config: VerlAdapterConfig,
        rq_config: RQEvolveConfig,
        *,
        project_root: str | Path,
    ) -> None:
        self.config = config
        self.rq_config = rq_config
        self.project_root = Path(project_root)

    def assert_verl_available(self) -> None:
        if importlib.util.find_spec("verl") is None:
            raise RuntimeError("verl is not installed in this Python environment")

    def fit(self) -> None:
        self.assert_verl_available()

        import ray
        from omegaconf import OmegaConf

        verl_config = self._load_verl_config()
        self._patch_reward_config(verl_config)
        OmegaConf.resolve(verl_config)

        if not ray.is_initialized():
            ray_init = verl_config.get("ray_init", {})
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                        "PYTHONPATH": str(self.project_root / "src")
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                    }
                },
                num_cpus=ray_init.get("num_cpus", None),
            )

        tokenizer, processor = self._build_tokenizer_and_processor(verl_config)
        backend = VerlPolicyBackend()
        evolver = self._build_evolver(backend)

        # The MAP-Elites archive lives outside the verl weight checkpoint, so we
        # persist/restore it ourselves. archive_dir is needed now (the
        # EvolvingSampler writes here each outer iteration), but the actual
        # resume/bootstrap is DEFERRED until after backend.bind() below — the
        # bootstrap evaluates every seed with the live solver (real R_Q), which
        # needs the worker group.
        archive_dir = (
            Path(str(verl_config.trainer.get("default_local_dir", "./rq_output/verl_ckpt")))
            / "rq_archive"
        )

        train_batch_size = int(verl_config.data.train_batch_size)
        train_dataset = VerlDynamicDataset(
            evolver.dataset,
            tokenizer,
            max_prompt_length=int(verl_config.data.max_prompt_length),
            truncation=verl_config.data.get("truncation", "left"),
            min_size=train_batch_size,
        )
        # Validation dataset: math benchmarks when enabled (one data_source per
        # benchmark -> verl reports per-benchmark accuracy via _validate); else a
        # dummy mirror of the train dataset (verl requires a non-empty val set).
        val_dataset = None
        if getattr(self.rq_config, "math_eval", None) and self.rq_config.math_eval.enabled:
            from .math_eval import build_math_eval_val_dataset

            val_dataset = build_math_eval_val_dataset(
                self.rq_config.math_eval,
                tokenizer,
                int(verl_config.data.max_prompt_length),
            )
        if val_dataset is None:
            val_dataset = VerlDynamicDataset(
                evolver.dataset,
                tokenizer,
                max_prompt_length=int(verl_config.data.max_prompt_length),
                truncation=verl_config.data.get("truncation", "left"),
                min_size=max(1, int(verl_config.data.get("val_batch_size") or train_batch_size)),
            )
        train_sampler = EvolvingSampler(
            train_dataset,
            evolver,
            # NOTE: OmegaConf .get(key, default) returns the default ONLY when the
            # key is absent. verl's base ppo_trainer.yaml defines data.seed: null,
            # so .get("seed", 1) yields None, not 1 -> guard with `or`.
            shuffle=bool(verl_config.data.get("shuffle") if verl_config.data.get("shuffle") is not None else True),
            seed=int(verl_config.data.get("seed") or 1),
            evolve_on_first_epoch=bool(self.rq_config.verl.evolve_on_first_epoch),
            archive_dir=archive_dir,
        )

        trainer = self._build_trainer(
            verl_config=verl_config,
            tokenizer=tokenizer,
            processor=processor,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            train_sampler=train_sampler,
        )
        trainer.init_workers()
        backend.bind(trainer)

        # Backend is now bound to the worker group -> solver rollouts work.
        # Resume the archive if a snapshot exists; otherwise bootstrap by
        # evaluating EVERY seed with the live solver. Real R_Q gives each seed a
        # real h_score, so seeds spread across H bins (instead of collapsing into
        # one placeholder bin) and the dataset is refreshed before epoch 0.
        resumed = False
        try:
            resumed = evolver.load_state(archive_dir)
        except Exception as exc:  # corrupt/partial snapshot — fall back to seeds
            print(f"[RQ-Evolve] archive restore failed ({exc!r}); bootstrapping from seeds")
        if resumed:
            print(
                f"[RQ-Evolve] restored archive "
                f"({len(evolver.archive.champions())} champions) from {archive_dir}"
            )
        else:
            self._bootstrap_seed_archive(evolver)
            evolver.save_state(archive_dir)

        trainer.fit()

    def _load_verl_config(self):
        from omegaconf import OmegaConf

        # Inline config (embedded `verl_config:` block in the rq_evolve yaml)
        # takes precedence over a separate config_path.
        user_override = self.config.inline_config
        if user_override is None:
            if not self.config.config_path:
                raise ValueError(
                    "VerlAdapterConfig needs either inline_config or config_path"
                )
            user_path = Path(self.config.config_path)
            if not user_path.is_absolute():
                user_path = self.project_root / user_path
            if not user_path.exists():
                raise FileNotFoundError(f"missing verl config: {user_path}")
            user_override = OmegaConf.load(user_path)

        package_root = _verl_package_root()
        # Prefer the pre-flattened reference config (verl >= 0.5 uses Hydra
        # `defaults:` composition with ${model_engine} interpolations in
        # ppo_trainer.yaml; plain OmegaConf.load can't resolve those).
        base_candidates = [
            package_root / "trainer" / "config" / "_generated_ppo_trainer.yaml",
            package_root / "trainer" / "config" / "ppo_trainer.yaml",
        ]
        base = next((OmegaConf.load(path) for path in base_candidates if path.exists()), OmegaConf.create({}))
        return OmegaConf.merge(base, user_override)

    def _patch_reward_config(self, config) -> None:
        from omegaconf import OmegaConf, open_dict

        reward_path, reward_name = _split_reward_function(
            self.config.reward_function,
            self.project_root,
        )
        with open_dict(config):
            if "custom_reward_function" not in config or config.custom_reward_function is None:
                config.custom_reward_function = {}
            config.custom_reward_function.path = str(reward_path)
            config.custom_reward_function.name = reward_name

            if "reward_model" not in config or config.reward_model is None:
                config.reward_model = {}
            if OmegaConf.select(config, "reward_model.reward_manager") is None:
                config.reward_model.reward_manager = "naive"
            if OmegaConf.select(config, "reward_model.enable") is None:
                config.reward_model.enable = False

            if "reward" in config and config.reward is not None:
                if "custom_reward_function" not in config.reward or config.reward.custom_reward_function is None:
                    config.reward.custom_reward_function = {}
                config.reward.custom_reward_function.path = str(reward_path)
                config.reward.custom_reward_function.name = reward_name
                if "reward_manager" not in config.reward or config.reward.reward_manager is None:
                    config.reward.reward_manager = {}
                if not isinstance(config.reward.reward_manager, str):
                    if OmegaConf.select(config, "reward.reward_manager.source") is None:
                        config.reward.reward_manager.source = "register"
                    if OmegaConf.select(config, "reward.reward_manager.name") is None:
                        config.reward.reward_manager.name = "naive"
                if OmegaConf.select(config, "reward.reward_model.enable") is None:
                    if "reward_model" not in config.reward or config.reward.reward_model is None:
                        config.reward.reward_model = {}
                    config.reward.reward_model.enable = False

            if "data" in config and config.data is not None:
                config.data.reward_fn_key = "data_source"

    def _build_tokenizer_and_processor(self, config):
        copy_to_local = _optional_import_attr(("verl.utils.fs", "copy_to_local"))
        hf_tokenizer = _optional_import_attr(("verl.utils", "hf_tokenizer"))
        hf_processor = _optional_import_attr(("verl.utils", "hf_processor"))
        if hf_tokenizer is None:
            hf_tokenizer = _import_attr([("verl.utils.tokenizer", "get_tokenizer")])
        if hf_processor is None:
            hf_processor = _optional_import_attr(("verl.utils.tokenizer", "get_processor"))

        model_path = config.actor_rollout_ref.model.path
        local_path = (
            copy_to_local(
                model_path,
                use_shm=config.actor_rollout_ref.model.get("use_shm", False),
            )
            if copy_to_local is not None
            else model_path
        )
        trust_remote_code = bool(
            config.data.get("trust_remote_code", False)
            or config.actor_rollout_ref.model.get("trust_remote_code", False)
        )
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        processor = (
            hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)
            if hf_processor is not None
            else None
        )
        return tokenizer, processor

    def _build_evolver(self, backend: VerlPolicyBackend) -> RQEvolver:
        archive = MAPElitesArchive(**asdict(self.rq_config.archive))
        return RQEvolver(
            archive=archive,
            backend=backend,
            evolution_config=self.rq_config.evolution,
            training_config=self.rq_config.training_data,
        )

    def _bootstrap_seed_archive(self, evolver: RQEvolver) -> None:
        """Evaluate every seed with the LIVE solver (real R_Q) and insert it.

        MUST be called AFTER backend.bind(trainer): evaluate_instance runs a
        solver rollout, which needs the worker group. Real R_Q gives each seed a
        real h_score, so seeds land in their true H bins rather than collapsing
        into one placeholder bin. Seeds still compete per niche, so two seeds in
        the same (H bin, concept_group) cell keep only the higher-R_Q one — a
        MAP-Elites property, not a bug. Then refresh the training dataset so
        epoch 0 trains on these seeds (with evolve_on_first_epoch=false).
        """
        seed_dir = Path(self.rq_config.evolution.seed_programs_dir)
        if not seed_dir.is_absolute():
            seed_dir = self.project_root / seed_dir
        seeds = evolver.load_seed_programs(seed_dir)
        if not seeds:
            raise ValueError(f"no valid seed programs in {seed_dir}")

        # vLLM launches with dummy (random) weights and is only synced via
        # update_weights. The training loop and the evolve phase push their own
        # weights, but bootstrap is a third generation site: push the live actor
        # weights ONCE here so the solver rollouts use the real policy. Without
        # this every seed scores ~random -> p_hat 0 -> empty training dataset.
        # (Previously begin_session pushed per session; that was removed when the
        # resident model made per-session pushes redundant -- but bootstrap still
        # needs the initial sync.)
        evolver.backend.sync_weights()

        inserted = 0
        for program in seeds:
            inst, reason = evolver.verify_program(program)
            if inst is None:
                print(f"[RQ-Evolve] seed rejected after load: {program.program_id} {reason}")
                continue
            # Real R_Q via solver rollout; sets program.p_hat / h_score / rq_score.
            result = evolver.evaluate_instance(program, inst)
            if evolver.archive.try_insert(
                program=program,
                h_value=result.uncertainty,
                problem_text=inst.problem,
                rq_score=result.rq_score,
            ):
                inserted += 1
            else:
                print(
                    f"[RQ-Evolve] seed not inserted (niche conflict / gate): "
                    f"{program.program_id} p_hat={result.p_hat:.2f} h={result.uncertainty:.3f}"
                )
        print(
            f"[RQ-Evolve] bootstrapped {inserted}/{len(seeds)} seeds with real R_Q; "
            f"{len(evolver.archive.champions())} champions on a "
            f"{evolver.archive.n_h_bins}x{evolver.archive.n_div_bins} grid"
        )
        evolver.refresh_dataset()
        if len(evolver.dataset.snapshot()) == 0:
            raise RuntimeError("bootstrap archive produced an empty training dataset")

    def _build_trainer(
        self,
        *,
        verl_config,
        tokenizer,
        processor,
        train_dataset,
        val_dataset,
        train_sampler,
    ):
        import ray

        RayPPOTrainer = _import_attr(
            [
                ("verl.trainer.ppo.ray_trainer", "RayPPOTrainer"),
                ("verl.trainer.ray_trainer", "RayPPOTrainer"),
            ]
        )
        Role = _import_attr(
            [
                ("verl.trainer.ppo.ray_trainer", "Role"),
                ("verl.trainer.ppo.utils", "Role"),
                ("verl.trainer.ray_trainer", "Role"),
            ]
        )
        ResourcePoolManager = _import_attr(
            [
                ("verl.trainer.ppo.ray_trainer", "ResourcePoolManager"),
                ("verl.single_controller.ray", "ResourcePoolManager"),
                ("verl.trainer.ray_trainer", "ResourcePoolManager"),
            ]
        )
        RayWorkerGroup = _import_attr(
            [
                ("verl.single_controller.ray", "RayWorkerGroup"),
            ]
        )
        collate_fn = _import_attr(
            [
                ("verl.utils.dataset.rl_dataset", "collate_fn"),
                ("verl.utils.dataset", "collate_fn"),
            ]
        )

        actor_rollout_cls, critic_cls, reward_model_cls, ray_worker_group_cls = _select_worker_classes(
            verl_config,
            default_ray_worker_group_cls=RayWorkerGroup,
        )
        actor_role = getattr(Role, "ActorRollout", getattr(Role, "ActorRolloutRef", None))
        if actor_role is None:
            raise RuntimeError("installed verl exposes neither Role.ActorRollout nor Role.ActorRolloutRef")

        global_pool_id = "global_pool"
        n_gpus_per_node = int(verl_config.trainer.get("n_gpus_per_node", 1))
        nnodes = max(1, int(verl_config.trainer.get("nnodes", 1)))
        role_worker_mapping = {actor_role: ray.remote(actor_rollout_cls)}
        mapping = {actor_role: global_pool_id}

        critic_role = getattr(Role, "Critic", None)
        if critic_role is not None and critic_cls is not None:
            role_worker_mapping[critic_role] = ray.remote(critic_cls)
            mapping[critic_role] = global_pool_id

        reward_model_enabled = bool(_cfg_select(verl_config, "reward_model.enable", False))
        reward_role = getattr(Role, "RewardModel", None)
        if reward_model_enabled and reward_role is not None and reward_model_cls is not None:
            role_worker_mapping[reward_role] = ray.remote(reward_model_cls)
            mapping[reward_role] = global_pool_id

        ref_role = getattr(Role, "RefPolicy", None)
        if ref_role is not None and _needs_reference_policy(verl_config):
            role_worker_mapping[ref_role] = ray.remote(actor_rollout_cls)
            mapping[ref_role] = global_pool_id

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec={global_pool_id: [n_gpus_per_node] * nnodes},
            mapping=mapping,
        )

        kwargs: dict[str, Any] = {
            "config": verl_config,
            "tokenizer": tokenizer,
            "processor": processor,
            "role_worker_mapping": role_worker_mapping,
            "resource_pool_manager": resource_pool_manager,
            "ray_worker_group_cls": ray_worker_group_cls,
            "train_dataset": train_dataset,
            "val_dataset": val_dataset,
            "collate_fn": collate_fn,
            "train_sampler": train_sampler,
        }
        if _supports_kwarg(RayPPOTrainer.__init__, "reward_fn"):
            reward_fn, val_reward_fn = _build_reward_managers(verl_config, tokenizer)
            kwargs["reward_fn"] = reward_fn
            kwargs["val_reward_fn"] = val_reward_fn
        if _supports_kwarg(RayPPOTrainer.__init__, "device_name"):
            kwargs["device_name"] = verl_config.trainer.get("device", "cuda")

        return RayPPOTrainer(**kwargs)


def describe_verl_runtime() -> dict[str, str]:
    spec = importlib.util.find_spec("verl")
    if spec is None:
        return {
            "python": sys.executable,
            "verl_version": "<not installed>",
            "verl_origin": "<not installed>",
        }
    try:
        version = metadata.version("verl")
    except metadata.PackageNotFoundError:
        version = "<unknown>"
    return {
        "python": sys.executable,
        "verl_version": version,
        "verl_origin": spec.origin or "<namespace package>",
    }


def _split_reward_function(spec: str, project_root: Path) -> tuple[Path, str]:
    if ":" in spec:
        path_text, name = spec.rsplit(":", 1)
    else:
        path_text, name = spec, "compute_score"
    path = Path(path_text)
    if not path.is_absolute():
        path = project_root / path
    return path, name


def _verl_package_root() -> Path:
    spec = importlib.util.find_spec("verl")
    if spec is None:
        raise RuntimeError("verl is not installed in this Python environment")
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations))).resolve()
    if spec.origin is None:
        raise RuntimeError("unable to locate installed verl package")
    return Path(spec.origin).resolve().parent


def _import_attr(candidates: list[tuple[str, str]]):
    errors: list[str] = []
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            return getattr(module, attr_name)
        except (ImportError, AttributeError) as exc:
            errors.append(f"{module_name}.{attr_name}: {exc}")
    raise ImportError("could not import any verl candidate:\n" + "\n".join(errors))


def _optional_import_attr(candidate: tuple[str, str]):
    try:
        return _import_attr([candidate])
    except ImportError:
        return None


def _select_worker_classes(config, *, default_ray_worker_group_cls):
    strategy = str(config.actor_rollout_ref.actor.get("strategy", "fsdp"))
    if strategy in {"fsdp", "fsdp2"}:
        worker_module = importlib.import_module("verl.workers.fsdp_workers")
        ray_worker_group_cls = default_ray_worker_group_cls
    elif strategy == "megatron":
        worker_module = importlib.import_module("verl.workers.megatron_workers")
        ray_worker_group_cls = _import_attr(
            [("verl.single_controller.ray.megatron", "NVMegatronRayWorkerGroup")]
        )
    else:
        raise NotImplementedError(f"unsupported actor strategy: {strategy}")

    # In verl 0.7.x we must pick AsyncActorRolloutRefWorker. It exposes
    # update_weights via @register(blocking=False) which the trainer's
    # checkpoint_manager.update_weights routes to via
    # actor_rollout_wg.update_weights. The non-async class doesn't expose it
    # -> AttributeError: 'RayWorkerGroup' object has no attribute 'update_weights'.
    # Our backend no longer calls actor_rollout_wg.generate_sequences
    # directly (we route generate through async_rollout_manager instead), so
    # the previous "this event loop is already running" failure mode no longer
    # applies — compute_log_prob is plain sync and is safe to call regardless.
    actor_cls = getattr(worker_module, "ActorRolloutRefWorker")
    async_actor_cls = getattr(worker_module, "AsyncActorRolloutRefWorker", actor_cls)
    critic_cls = getattr(worker_module, "CriticWorker", None)
    reward_model_cls = getattr(worker_module, "RewardModelWorker", None)
    return (
        async_actor_cls,
        critic_cls,
        reward_model_cls,
        ray_worker_group_cls,
    )


def _needs_reference_policy(config) -> bool:
    return bool(
        _cfg_select(config, "algorithm.use_kl_in_reward", False)
        or _cfg_select(config, "actor_rollout_ref.actor.use_kl_loss", False)
    )


def _cfg_select(config, dotted_key: str, default=None):
    try:
        from omegaconf import OmegaConf

        value = OmegaConf.select(config, dotted_key)
        return default if value is None else value
    except Exception:
        current = config
        for key in dotted_key.split("."):
            if isinstance(current, dict):
                current = current.get(key, default)
            else:
                current = getattr(current, key, default)
            if current is default:
                return default
        return current


def _supports_kwarg(callable_obj, name: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return True
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def _build_reward_managers(config, tokenizer):
    load_reward_manager = _import_attr(
        [
            ("verl.trainer.ppo.reward", "load_reward_manager"),
            ("verl.workers.reward", "load_reward_manager"),
        ]
    )
    reward_kwargs = {}
    reward_model = getattr(config, "reward_model", None)
    if reward_model is not None:
        reward_kwargs = reward_model.get("reward_kwargs", {})
    return (
        load_reward_manager(config, tokenizer, num_examine=0, **reward_kwargs),
        load_reward_manager(config, tokenizer, num_examine=1, **reward_kwargs),
    )
