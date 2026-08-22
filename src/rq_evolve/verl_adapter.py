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
from .dataset import (
    DynamicProblemDataset,
    VerlDynamicDataset,
    load_static_training_jsonl,
    validate_static_training_schedule,
)
from .evolution import RQEvolver
from .openai_evaluator import (
    load_project_dotenv,
    validate_openai_evaluator_environment,
)
from .verl_backend import VerlPolicyBackend


# Ray creates Unix domain sockets inside its temp dir, and AF_UNIX paths are
# capped at 107 bytes. The session directory plus "/sockets/plasma_store" eats
# roughly 63 of those, so anything much over 40 characters fails at startup --
# which is exactly how the first attempt at relocating this broke the run.
_RAY_TEMP_DIR_BUDGET = 40


def _ray_temp_dir(project_root: Path) -> str | None:
    """Keep Ray's spill off the small root filesystem, without breaking startup.

    Ray defaults to /tmp, which lives on the root partition; that partition hit
    96% during a run and the raylet logged "over 95%" continuously. Returning
    None falls back to Ray's own default, which is always preferable to failing
    to start because the relocated path was too long for a socket.
    """
    configured = os.environ.get("RAY_TMPDIR")
    candidate = (
        Path(configured)
        if configured
        else project_root / ".raytmp"
    )
    if len(str(candidate)) > _RAY_TEMP_DIR_BUDGET:
        print(
            f"[RQ-Evolve] ray temp dir {candidate} is {len(str(candidate))} "
            f"chars (> {_RAY_TEMP_DIR_BUDGET}); Unix sockets would exceed the "
            "107-byte limit, so falling back to Ray's default under /tmp. Set "
            "RAY_TMPDIR to a shorter path to keep spill off the root disk."
        )
        return None
    candidate.mkdir(parents=True, exist_ok=True)
    return str(candidate)


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
        # Outer-iteration index of the MAP snapshot currently driving Solver
        # training (the archive_iter{N}.json active while global_steps advance).
        # -1 = no evolution yet (epoch-0 bootstrap). Stored in data.pt so a resume
        # knows which snapshots belong to the abandoned future and must be cleared.
        self._active_iteration = -1
        # Async-RL instrumentation, wired by attach_instrumentation() once the
        # trainer exists (all optional -- the sampler works without them).
        self.rollout_metrics = None
        self.version_tracker = None
        self.metrics_logger = None

    def attach_instrumentation(
        self, *, rollout_metrics=None, version_tracker=None, metrics_logger=None
    ) -> None:
        self.rollout_metrics = rollout_metrics
        self.version_tracker = version_tracker
        self.metrics_logger = metrics_logger

    def __iter__(self) -> Iterator[int]:
        if self.epoch > 0 or self.evolve_on_first_epoch:
            import time as _time

            # Stamp the outer-iteration index on the backend so streamed
            # rollout JSONL lines carry it, and start this phase's metrics.
            backend = getattr(self.evolver, "backend", None)
            if backend is not None and hasattr(backend, "current_iteration"):
                backend.current_iteration = self.epoch
            if self.rollout_metrics is not None:
                self.rollout_metrics.reset()
            evolve_t0 = _time.monotonic()
            metrics = self.evolver.run_outer_iteration(self.epoch)
            # The whole evolve phase runs while the optimizer is idle (verl
            # HYBRID mode time-shares the GPUs) -> this IS the trainer idle time.
            trainer_idle_s = _time.monotonic() - evolve_t0
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
            self._log_map_figure_to_wandb(metrics)
            self._log_rollout_instrumentation(trainer_idle_s)
            # This evolution's snapshot now drives the Solver training that follows
            # (and any global_step_* checkpoint saved during it).
            self._active_iteration = self.epoch

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

    # --- torchdata Stateful protocol: persist the MAP INTO the verl data.pt ----
    # checkpoint so the grid is restored atomically with the model weights.
    # _BatchSamplerIterator.state_dict()/load_state_dict() call these because this
    # sampler now exposes both methods (see torchdata stateful_dataloader/sampler).
    def state_dict(self) -> dict:
        """Snapshot the LIVE MAP + bookkeeping at the current global_step. Captured
        whenever verl saves a checkpoint (train_dataloader.state_dict()), so the
        grid travels with the weights and a resume needs no separate archive."""
        ev = self.evolver
        return {
            "epoch": int(self.epoch),
            "active_iteration": int(self._active_iteration),
            "map_payload": {
                "archive": ev.archive.to_payload(),
                "used_seeds": {pid: sorted(s) for pid, s in ev.used_seeds.items()},
                "current_iteration": ev.current_iteration,
            },
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the MAP from data.pt. Runs inside verl _load_checkpoint (i.e.
        AFTER fit()'s archive.json load), so the checkpoint-aligned grid OVERRIDES
        the latest archive.json -> MAP exactly matches the restored weights. Then
        the abandoned-future snapshots (evolved after this checkpoint) are moved
        aside so new snapshots regenerate cleanly from here."""
        if not isinstance(state_dict, dict):
            return
        self.epoch = int(state_dict.get("epoch", self.epoch))
        self._active_iteration = int(
            state_dict.get("active_iteration", self._active_iteration)
        )
        payload = state_dict.get("map_payload")
        if not payload:  # pre-feature checkpoint -> keep fit()'s archive.json state
            return
        ev = self.evolver
        try:
            n = ev.archive.load_payload(payload.get("archive", {}))
            ev.used_seeds = {
                pid: set(s) for pid, s in (payload.get("used_seeds") or {}).items()
            }
            ev.current_iteration = int(
                payload.get("current_iteration", ev.current_iteration)
            )
            ev.refresh_dataset()
            print(
                f"[RQ-Evolve] resume: restored MAP from data.pt ({n} champions, "
                f"active iter {self._active_iteration}, epoch {self.epoch}) -- "
                f"overrides latest archive.json to match the weight checkpoint"
            )
        except Exception as exc:  # never let checkpoint housekeeping kill a resume
            print(f"[RQ-Evolve] resume: data.pt MAP restore FAILED ({exc!r}); "
                  f"keeping archive.json state")
            return
        self._archive_post_checkpoint_snapshots(self._active_iteration)
        # Rewrite the live snapshot files (archive.json / used_seeds) to the
        # restored point so disk is immediately consistent with the weights -- not
        # the now-discarded latest. A crash before the first post-resume evolution
        # then resumes from K, not the abandoned future.
        if self.archive_dir is not None:
            try:
                ev.save_state(self.archive_dir)
            except Exception as exc:
                print(f"[RQ-Evolve] resume: live-snapshot rewrite skipped ({exc!r})")

    def _archive_post_checkpoint_snapshots(self, active_iteration: int) -> None:
        """Move MAP snapshots / log lines evolved AFTER the resumed checkpoint into
        a backup folder, so on-disk history matches the restored grid. Everything
        with outer-iteration index > active_iteration is the abandoned future."""
        if self.archive_dir is None:
            return
        import json
        import re
        import shutil
        from datetime import datetime

        cutoff = int(active_iteration)
        d = Path(self.archive_dir)
        try:
            backup = d / f"_stale_{datetime.now().strftime('%Y%m%d_%H%M%S')}_after_iter{cutoff}"
            moved: list[str] = []
            pat = re.compile(r"archive_iter(\d+)\.json$")
            for p in d.glob("archive_iter*.json"):
                m = pat.search(p.name)
                if m and int(m.group(1)) > cutoff:
                    backup.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(p), str(backup / p.name))
                    moved.append(p.name)
            # Append-only logs: split into kept (<= cutoff) and dropped (> cutoff).
            for fname, key in (("evolution_log.jsonl", "iteration"),):
                f = d / fname
                if not f.exists():
                    continue
                kept: list[str] = []
                dropped: list[str] = []
                for line in f.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        idx = int(json.loads(line).get(key, -1))
                    except Exception:
                        kept.append(line)  # unparseable -> keep (never lose silently)
                        continue
                    (kept if idx <= cutoff else dropped).append(line)
                if dropped:
                    backup.mkdir(parents=True, exist_ok=True)
                    (backup / fname).write_text("\n".join(dropped) + "\n", encoding="utf-8")
                    f.write_text(("\n".join(kept) + "\n") if kept else "", encoding="utf-8")
                    moved.append(f"{fname}(-{len(dropped)} lines)")
            if moved:
                print(f"[RQ-Evolve] resume: moved post-checkpoint MAP history "
                      f"(> iter {cutoff}) to {backup.name} -> {moved}")
        except Exception as exc:
            print(f"[RQ-Evolve] resume: snapshot cleanup skipped ({exc!r})")

    def _log_rollout_instrumentation(self, trainer_idle_s: float) -> None:
        """Per-outer-iteration rollout metrics -> wandb (commit=False) + JSONL."""
        snapshot: dict = {}
        payload: dict = {"evolve/trainer_idle_s": float(trainer_idle_s)}
        if self.rollout_metrics is not None:
            snapshot = self.rollout_metrics.snapshot()
            payload.update(self.rollout_metrics.to_wandb("rollout/"))
        if self.version_tracker is not None:
            payload.update(self.version_tracker.to_wandb("rollout/"))
        try:
            import wandb

            if wandb.run is not None:
                wandb.log(payload, commit=False)
        except Exception:
            pass
        if self.metrics_logger is not None:
            record = {"iteration": int(self.epoch), "trainer_idle_s": float(trainer_idle_s)}
            record.update(snapshot)
            if self.version_tracker is not None:
                record["policy_version"] = self.version_tracker.policy_version
                record["adapter_version"] = self.version_tracker.adapter_version
                record["source_checkpoint"] = self.version_tracker.source_checkpoint
            try:
                self.metrics_logger.log(record)
            except Exception as exc:
                print(f"[RQ-Evolve] rollout_metrics.jsonl write failed ({exc!r})")

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

    def _log_map_figure_to_wandb(self, metrics: dict) -> None:
        """Send one picture of the GROUP x SKILL archive per outer iteration.

        ``coverage`` says how many of the 48 cells are filled and cannot say
        which. "Only two domains" and "only two reasoning moves" give the same
        number and need opposite fixes, so the grid itself is logged as an image
        and the cells filled since the previous iteration are outlined.

        Best-effort throughout: a logging backend that cannot draw must never
        take a training run down.
        """
        try:
            import wandb

            if wandb.run is None:
                return
            from .map_figure import occupied_cells, render_archive_figure

            archive = self.evolver.archive
            previous = getattr(self, "_logged_map_cells", None)
            current = occupied_cells(archive)
            figure = render_archive_figure(
                archive,
                iteration=int(self.epoch),
                new_cells=(current - previous) if previous is not None else None,
                stats=metrics,
            )
            self._logged_map_cells = current
            if figure is None:
                return
            payload = {"evolve/map": wandb.Image(figure)}
            hook = getattr(self.evolver, "replay_hook", None)
            if hook is not None:
                payload.update(hook.stats.to_wandb("evolve/replay_"))
            # Which SKILLs the judge is willing to emit at all. A label it never
            # returns is a cell no child can be archived into, so this bounds
            # reachable coverage independently of the agreement rate.
            counts = getattr(self.evolver, "judge_skill_counts", None)
            if counts:
                payload.update(
                    {f"evolve/judge_skill/{k}": v for k, v in counts.items()}
                )
            wandb.log(payload, commit=False)
            try:
                import matplotlib.pyplot as plt

                plt.close(figure)
            except Exception:
                pass
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.dataset)


class StaticTrainingSampler:
    """Finite, stateful sampler for a pre-audited fixed training JSONL.

    Unlike ``EvolvingSampler``, this class has no Evolver/archive reference and
    cannot mutate the dataset.  It permits exactly ``epochs`` complete passes;
    the adapter separately requires VERL's total steps to consume those passes
    exactly, with no ``drop_last`` loss.
    """

    def __init__(
        self,
        dataset: VerlDynamicDataset,
        *,
        source_sha256: str,
        epochs: int,
        shuffle: bool = True,
        seed: int = 1,
    ) -> None:
        self.dataset = dataset
        self.source_sha256 = str(source_sha256)
        self.epochs = int(epochs)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.epoch = 0
        if self.epochs < 1:
            raise ValueError("static training sampler epochs must be >= 1")

    def __iter__(self) -> Iterator[int]:
        if self.epoch >= self.epochs:
            raise RuntimeError(
                f"static training data exhausted after {self.epochs} complete "
                "epoch(s); increase training_data.static_epochs explicitly"
            )
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

    def __len__(self) -> int:
        return len(self.dataset)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": int(self.epoch),
            "source_sha256": self.source_sha256,
            "epochs": self.epochs,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if not isinstance(state_dict, dict):
            return
        checkpoint_hash = str(state_dict.get("source_sha256", ""))
        if checkpoint_hash and checkpoint_hash != self.source_sha256:
            raise RuntimeError(
                "static training JSONL changed since the checkpoint was saved "
                f"({checkpoint_hash} != {self.source_sha256})"
            )
        checkpoint_epochs = int(state_dict.get("epochs", self.epochs))
        if checkpoint_epochs != self.epochs:
            raise RuntimeError(
                "training_data.static_epochs changed since the checkpoint was "
                f"saved ({checkpoint_epochs} != {self.epochs})"
            )
        self.epoch = int(state_dict.get("epoch", self.epoch))


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
        if (
            not self.static_training_enabled
            and self.rq_config.evolution.use_evaluator
            and self.rq_config.evolution.evaluator_provider == "openai"
        ):
            load_project_dotenv(self.project_root)
            validate_openai_evaluator_environment()

    @property
    def static_training_enabled(self) -> bool:
        return bool(self.rq_config.training_data.static_training_jsonl)

    def assert_verl_available(self) -> None:
        if importlib.util.find_spec("verl") is None:
            raise RuntimeError("verl is not installed in this Python environment")

    def _install_replay_hook(self, ctx: dict) -> None:
        """Serve the solver's training batch from the re-scoring rollouts.

        Without this the trainer samples the policy a second time over prompts
        the re-scoring already rolled out under the same weights. The hook is
        installed on the rollout manager instance (no verl edit) and falls
        through to real generation for anything it cannot serve, so the worst
        case is the old cost, never a wrong batch.
        """
        evolver = ctx.get("evolver")
        trainer = ctx.get("trainer")
        if evolver is None or trainer is None:
            return
        if not self.rq_config.training_data.replay_training_batch:
            return

        manager = getattr(trainer, "async_rollout_manager", None)
        if manager is None:
            print(
                "[RQ-Evolve] replay disabled: the trainer exposes no "
                "async_rollout_manager to wrap",
                flush=True,
            )
            return

        m = int(self.rq_config.evolution.rollouts_per_seed)
        rollout_n = int(
            ctx["verl_config"].actor_rollout_ref.rollout.n
            if "verl_config" in ctx
            else m
        )
        if rollout_n != m:
            # The trainer repeats each prompt rollout.n times before generating.
            # If that differs from m the cached group cannot line up row for
            # row, and every step would silently fall through to generation --
            # paying the doubled budget the replay exists to remove.
            raise ValueError(
                "replay_training_batch requires "
                f"actor_rollout_ref.rollout.n ({rollout_n}) == "
                f"evolution.rollouts_per_seed ({m}). Set them equal, or turn "
                "off training_data.replay_training_batch to keep the separate "
                "sampling pass."
            )

        from .replay_hook import ReplayRolloutHook

        hook = ReplayRolloutHook(evolver.replay, rollouts_per_seed=m)
        if hook.install(manager):
            self.replay_hook = hook
            evolver.replay_hook = hook
            print(
                f"[RQ-Evolve] replay hook installed (m={m}): the solver update "
                "reuses the re-scoring rollouts instead of sampling again",
                flush=True,
            )

    def fit(self) -> None:
        ctx = self._setup()
        if not ctx["static_training"]:
            self._resume_or_bootstrap(ctx)

        # Resume: verl restores trainer.global_steps from the checkpoint, so the run
        # simply continues until global_steps reaches the configured target.

        # Mirror the Solver's GRPO metrics (verl logs them as actor/* , critic/*)
        # under a clean ``solver/*`` namespace alongside ``evolve/*`` metrics.
        _install_solver_metric_alias()
        self._install_replay_hook(ctx)

        ctx["trainer"].fit()

    def audit_static_training_data(self) -> dict[str, Any]:
        """Tokenize and report a fixed JSONL without starting Ray or training."""

        if not self.static_training_enabled:
            raise ValueError(
                "training_data.static_training_jsonl is not configured"
            )
        from omegaconf import OmegaConf

        verl_config = self._load_verl_config()
        OmegaConf.resolve(verl_config)
        tokenizer, _ = self._build_tokenizer_and_processor(verl_config)
        _, report = self._load_static_rows(tokenizer)
        return self._validate_static_schedule(
            report,
            verl_config,
            require_expected=False,
        )

    def _setup(self) -> dict:
        """Everything up to (excluding) archive resume/bootstrap: config, ray,
        tokenizer, trainer, workers, backend bind, and async-RL instrumentation.
        Shared by fit() and dry_run_rollout() so the dry run exercises the
        EXACT production stack."""
        self.assert_verl_available()

        import ray
        from omegaconf import OmegaConf

        verl_config = self._load_verl_config()
        self._patch_reward_config(verl_config)
        # LoRA plumbing + fail-fast validation BEFORE ray/worker startup: an
        # invalid LoRA/model/vLLM combination should die here with a clear
        # message, not minutes later inside a worker.
        self._apply_lora_config(verl_config)
        OmegaConf.resolve(verl_config)

        tokenizer, processor = self._build_tokenizer_and_processor(verl_config)
        static_training = self.static_training_enabled
        static_report = None
        static_rows = None
        backend = None
        evolver = None
        archive_dir = None

        train_batch_size = int(verl_config.data.train_batch_size)
        if static_training:
            static_rows, raw_report = self._load_static_rows(tokenizer)
            static_report = self._validate_static_schedule(
                raw_report,
                verl_config,
                require_expected=True,
            )

        # A stale, malformed, truncated, or compute-mismatched fixed dataset
        # must fail above, before Ray starts and reserves CPUs/GPUs.
        if not ray.is_initialized():
            ray_init = verl_config.get("ray_init", {})
            ray.init(
                runtime_env={
                    "env_vars": {
                        "TOKENIZERS_PARALLELISM": "true",
                        "NCCL_DEBUG": "WARN",
                        # async agent-loop rollout requires the vLLM V1 engine
                        "VLLM_USE_V1": "1",
                        "VLLM_LOGGING_LEVEL": "WARN",
                        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                        "PYTHONPATH": str(self.project_root / "src")
                        + os.pathsep
                        + os.environ.get("PYTHONPATH", ""),
                    },
                },
                num_cpus=ray_init.get("num_cpus", None),
                _temp_dir=_ray_temp_dir(self.project_root),
            )

        if static_training:
            assert static_rows is not None
            dynamic_dataset = DynamicProblemDataset(static_rows)
            train_dataset = VerlDynamicDataset(
                dynamic_dataset,
                tokenizer,
                max_prompt_length=int(verl_config.data.max_prompt_length),
                truncation=verl_config.data.get("truncation", "left"),
                # Never modulo-pad a short fixed dataset. The schedule audit
                # above requires an exact number of full batches instead.
                min_size=1,
                data_source=f"rq_static_{self.rq_config.training_data.static_condition}",
            )
            train_sampler = StaticTrainingSampler(
                train_dataset,
                source_sha256=static_report["source_sha256"],
                epochs=int(self.rq_config.training_data.static_epochs),
                shuffle=bool(
                    verl_config.data.get("shuffle")
                    if verl_config.data.get("shuffle") is not None
                    else True
                ),
                seed=int(verl_config.data.get("seed") or 1),
            )
        else:
            backend = VerlPolicyBackend()
            evolver = self._build_evolver(backend)

            # The MAP-Elites archive lives outside the verl weight checkpoint,
            # so we persist/restore it ourselves.  This entire branch is skipped
            # for fixed-data expansion runs.
            archive_dir = (
                Path(
                    str(
                        verl_config.trainer.get(
                            "default_local_dir", "./rq_output/verl_ckpt"
                        )
                    )
                )
                / "rq_archive"
            )
            dynamic_dataset = evolver.dataset
            train_dataset = VerlDynamicDataset(
                dynamic_dataset,
                tokenizer,
                max_prompt_length=int(verl_config.data.max_prompt_length),
                truncation=verl_config.data.get("truncation", "left"),
                min_size=train_batch_size,
            )
            train_sampler = EvolvingSampler(
                train_dataset,
                evolver,
                # NOTE: OmegaConf .get(key, default) returns the default ONLY
                # when the key is absent.  VERL defines data.seed: null.
                shuffle=bool(
                    verl_config.data.get("shuffle")
                    if verl_config.data.get("shuffle") is not None
                    else True
                ),
                seed=int(verl_config.data.get("seed") or 1),
                evolve_on_first_epoch=bool(
                    self.rq_config.verl.evolve_on_first_epoch
                ),
                archive_dir=archive_dir,
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
                dynamic_dataset,
                tokenizer,
                max_prompt_length=int(verl_config.data.max_prompt_length),
                truncation=verl_config.data.get("truncation", "left"),
                min_size=(
                    1
                    if static_training
                    else max(
                        1,
                        int(
                            verl_config.data.get("val_batch_size")
                            or train_batch_size
                        ),
                    )
                ),
                data_source=(
                    f"rq_static_{self.rq_config.training_data.static_condition}"
                    if static_training
                    else "rq_evolved"
                ),
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
        if static_training:
            print("[RQ-Evolve] static Solver-training data audit:")
            for key in (
                "condition",
                "source_path",
                "source_sha256",
                "source_rows",
                "prompt_tokens",
                "reference_answer_tokens",
                "token_count",
                "static_epochs",
                "row_exposures",
                "token_exposures",
            ):
                print(f"  {key}: {static_report[key]}")
            return {
                "trainer": trainer,
                "backend": None,
                "evolver": None,
                "train_sampler": train_sampler,
                "verl_config": verl_config,
                "archive_dir": None,
                "rollout_metrics": None,
                "version_tracker": None,
                "sample_logger": None,
                "static_training": True,
                "static_report": static_report,
            }

        assert backend is not None
        assert evolver is not None
        assert archive_dir is not None
        backend.bind(trainer)

        # --- async-RL instrumentation -------------------------------------
        # Policy/adapter versions bump on EVERY trainer->vLLM weight sync
        # (verl's per-step push, the initial sync, and the evolve-phase wake);
        # rollout samples are stamped with them and gated by staleness_mode.
        from .metrics import PolicyVersionTracker, RolloutMetrics
        from .rollout_log import METRICS_FILE, SAMPLES_FILE, JsonlSampleLogger

        version_tracker = PolicyVersionTracker()
        version_tracker.install(
            trainer,
            lora_enabled=bool(self.rq_config.lora.enabled),
            model_path=str(verl_config.actor_rollout_ref.model.path),
        )
        rollout_metrics = RolloutMetrics()
        sample_logger = JsonlSampleLogger(
            archive_dir / SAMPLES_FILE,
            enabled=bool(self.rq_config.async_rollout.log_samples),
        )
        metrics_logger = JsonlSampleLogger(archive_dir / METRICS_FILE)
        backend.configure_streaming(
            self.rq_config.async_rollout,
            version_tracker=version_tracker,
            sample_logger=sample_logger,
            rollout_metrics=rollout_metrics,
        )
        train_sampler.attach_instrumentation(
            rollout_metrics=rollout_metrics,
            version_tracker=version_tracker,
            metrics_logger=metrics_logger,
        )

        return {
            "trainer": trainer,
            "backend": backend,
            "evolver": evolver,
            "train_sampler": train_sampler,
            "verl_config": verl_config,
            "archive_dir": archive_dir,
            "rollout_metrics": rollout_metrics,
            "version_tracker": version_tracker,
            "sample_logger": sample_logger,
            "static_training": False,
            "static_report": None,
        }

    def _resume_or_bootstrap(self, ctx: dict) -> None:
        evolver = ctx["evolver"]
        archive_dir = ctx["archive_dir"]
        train_sampler = ctx["train_sampler"]

        # Backend is now bound to the worker group -> solver rollouts work.
        # Resume the archive if a snapshot exists; otherwise bootstrap by
        # evaluating EVERY seed with the live solver. Real R_Q gives each seed a
        # real u_score, so seeds spread across H bins (instead of collapsing into
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
            # EvolvingSampler.epoch is ephemeral (always starts at 0 on a fresh
            # process), but it is the iteration index used to name the per-step
            # snapshots archive_iter{N}.json and to label evolution_log.jsonl. On a
            # naive resume it restarts at 0 and OVERWRITES the previous run's
            # snapshots / duplicates log iterations. Continue numbering past the
            # highest archive_iter{N}.json already on disk so prior snapshots are
            # preserved and the iteration labels stay monotonic across restarts.
            resume_epoch = self._next_outer_iteration(archive_dir)
            if resume_epoch > 0:
                train_sampler.epoch = resume_epoch
                print(
                    f"[RQ-Evolve] resume: continuing outer-iteration numbering at "
                    f"epoch {resume_epoch} (preserving archive_iter*.json snapshots)"
                )
        else:
            self._bootstrap_seed_archive(evolver)
            evolver.save_state(archive_dir)

    def dry_run_rollout(self, *, n_problems: int = 24, n_rollouts: int = 2) -> dict:
        """Live async-rollout dry run on the REAL stack (no training step).

        Boots ray + workers + backend + instrumentation exactly like fit(),
        pushes weights once, then streams engineered short/long prompts through
        the chunked scheduler and reports the metrics. Proves, before a real
        run: completions are consumed as they finish, short problems are not
        blocked behind long ones, pending stays bounded, and accepted/rejected
        JSONL lines appear. Requires free GPUs.
        """
        if self.static_training_enabled:
            raise ValueError(
                "dry_run_rollout exercises Evolver generation and is disabled "
                "when static_training_jsonl is configured"
            )
        import time as _time

        ctx = self._setup()
        backend = ctx["backend"]
        metrics = ctx["rollout_metrics"]
        metrics.reset()
        backend.current_iteration = -1

        from .program import ProblemInstance

        instances = []
        for i in range(n_problems):
            if i % 2 == 0:
                instances.append(
                    ProblemInstance(
                        problem=(
                            f"What is {i} + {i + 1}? Answer with just the "
                            f"number inside \\boxed{{}}."
                        ),
                        answer=str(i + i + 1),
                        program_id=f"dryrun_short_{i}",
                        seed=0,
                    )
                )
            else:
                instances.append(
                    ProblemInstance(
                        problem=(
                            f"Write a complete, fully detailed step-by-step "
                            f"derivation (every intermediate step, no skipping) "
                            f"of the sum of the first {100 + i} positive "
                            f"integers, then verify it a second way, and only "
                            f"then give the final answer in \\boxed{{}}."
                        ),
                        answer=str((100 + i) * (101 + i) // 2),
                        program_id=f"dryrun_long_{i}",
                        seed=0,
                    )
                )

        backend.sync_weights()
        t0 = _time.time()
        backend.begin_session()
        try:
            pending = backend.generate_rollouts(instances, n_rollouts)
        finally:
            backend.end_session()
        grouped = backend.finalize_rollouts(pending)

        # completion timeline: chunk_size=1 -> one group per instance
        short_done, long_done = [], []
        for inst, rows in zip(instances, grouped):
            if not rows:
                continue
            done_at = max(r.ts_end for r in rows) - t0
            (short_done if "short" in inst.program_id else long_done).append(done_at)
        report = {
            "metrics": metrics.snapshot(),
            "short_mean_completion_s": (
                sum(short_done) / len(short_done) if short_done else None
            ),
            "long_mean_completion_s": (
                sum(long_done) / len(long_done) if long_done else None
            ),
            "samples_logged": getattr(ctx["sample_logger"], "lines_written", 0),
        }
        print("[RQ-Evolve dry-run] rollout metrics:")
        for key, value in report["metrics"].items():
            print(f"  {key}: {value}")
        print(
            f"[RQ-Evolve dry-run] mean completion: "
            f"short={report['short_mean_completion_s']}s "
            f"long={report['long_mean_completion_s']}s "
            f"(short well below long ==> streaming consumption is working; "
            f"roughly equal ==> a whole-batch barrier is back)"
        )
        return report

    def _apply_lora_config(self, verl_config) -> None:
        """Plumb rq-level lora.* into verl's actor_rollout_ref.model.lora_*.

        Fail-fast rules (all raise BEFORE ray.init):
          * strategy must be fsdp/fsdp2
          * sync_mode=merge_push -> NotImplementedError (deferred; see
            docs/deepseek_support_plan.md; engine-restart via
            scripts/merge_fsdp_to_hf.py is the manual fallback)
          * sync_mode=auto/native_adapter -> the installed vLLM must support
            runtime LoRA for this model class (Qwen3: yes; DeepSeek MoE: no)
        """
        lora = self.rq_config.lora
        if not lora.enabled:
            return
        from omegaconf import open_dict

        from .preflight import load_hf_config, vllm_supports_lora

        strategy = str(verl_config.actor_rollout_ref.actor.get("strategy", "fsdp"))
        if strategy not in ("fsdp", "fsdp2"):
            raise ValueError(
                f"lora.enabled requires actor strategy fsdp/fsdp2, got {strategy!r}"
            )
        if lora.sync_mode == "merge_push":
            raise NotImplementedError(
                "lora.sync_mode=merge_push (merged full-weight push for models "
                "without vLLM runtime-LoRA support) is not implemented yet -- "
                "see docs/deepseek_support_plan.md. Manual fallback: train, "
                "merge the adapter with scripts/merge_fsdp_to_hf.py, and "
                "relaunch the rollout engine on the merged checkpoint."
            )

        model_path = str(verl_config.actor_rollout_ref.model.path)
        trust_remote_code = bool(
            verl_config.actor_rollout_ref.model.get("trust_remote_code", True)
        )
        try:
            hf_config = load_hf_config(model_path, trust_remote_code)
            archs = list(getattr(hf_config, "architectures", None) or [])
            lora_ok = vllm_supports_lora(archs)
        except Exception as exc:
            raise RuntimeError(
                f"LoRA preflight could not resolve vLLM support for "
                f"{model_path}: {exc}. Run scripts/preflight_check.py for the "
                f"full diagnosis."
            ) from exc
        if not lora_ok:
            raise RuntimeError(
                f"vLLM has no runtime-LoRA support for {archs} (required by "
                f"lora.sync_mode={lora.sync_mode!r}). For such models use the "
                f"engine-restart fallback (scripts/merge_fsdp_to_hf.py) or the "
                f"planned merge_push worker -- docs/deepseek_support_plan.md."
            )

        with open_dict(verl_config):
            model_cfg = verl_config.actor_rollout_ref.model
            model_cfg.lora_rank = int(lora.rank)
            model_cfg.lora_alpha = int(lora.alpha)
            model_cfg.target_modules = list(lora.target_modules)
        if lora.dropout and not self._verl_supports_lora_dropout():
            print(
                f"[RQ-Evolve] WARNING: lora.dropout={lora.dropout} configured, but "
                f"the installed verl 0.7.1 fsdp worker does not plumb lora_dropout "
                f"into peft -- EFFECTIVE DROPOUT IS 0.0. (A one-line local verl "
                f"patch is documented in docs/deepseek_support_plan.md.)"
            )
        print(
            f"[RQ-Evolve] LoRA enabled: rank={lora.rank} alpha={lora.alpha} "
            f"target_modules={list(lora.target_modules)} sync=native_adapter "
            f"(first sync pushes full base weights under load_format=dummy, "
            f"later syncs push adapter-only via vLLM add_lora)"
        )

    @staticmethod
    def _verl_supports_lora_dropout() -> bool:
        """Whether the installed verl plumbs lora_dropout into its peft config."""
        try:
            import verl.workers.fsdp_workers as fsdp_workers

            return "lora_dropout" in inspect.getsource(fsdp_workers)
        except Exception:
            return False

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

    def _load_static_rows(self, tokenizer) -> tuple[list[dict], dict[str, Any]]:
        static_path = Path(
            str(self.rq_config.training_data.static_training_jsonl)
        ).expanduser()
        if not static_path.is_absolute():
            static_path = self.project_root / static_path
        return load_static_training_jsonl(
            static_path,
            tokenizer,
            condition=str(self.rq_config.training_data.static_condition),
        )

    def _validate_static_schedule(
        self,
        report: dict[str, Any],
        verl_config,
        *,
        require_expected: bool,
    ) -> dict[str, Any]:
        data_config = self.rq_config.training_data
        batch_size = int(
            verl_config.data.get(
                "gen_batch_size", verl_config.data.train_batch_size
            )
            or verl_config.data.train_batch_size
        )
        raw_total_steps = verl_config.trainer.get("total_training_steps")
        schedule = validate_static_training_schedule(
            report,
            batch_size=batch_size,
            total_training_steps=(
                None if raw_total_steps is None else int(raw_total_steps)
            ),
            trainer_total_epochs=int(verl_config.trainer.total_epochs),
            static_epochs=int(data_config.static_epochs),
            expected_rows=data_config.static_expected_rows,
            expected_tokens=data_config.static_expected_tokens,
            require_expected=require_expected,
            max_prompt_length=int(verl_config.data.max_prompt_length),
            raise_on_error=False,
        )
        resume_mode = str(
            verl_config.trainer.get("resume_mode", "auto") or "auto"
        )
        schedule.update(
            {
                "resume_mode": resume_mode,
                "default_local_dir": str(
                    verl_config.trainer.get(
                        "default_local_dir", "./rq_output/verl_ckpt"
                    )
                ),
                "base_model_path": str(
                    verl_config.actor_rollout_ref.model.path
                ),
            }
        )
        if resume_mode != "disable":
            schedule["issues"].append(
                "trainer.resume_mode must be 'disable' for static paired "
                "training so both conditions start from the declared base "
                "checkpoint instead of an existing condition-specific state"
            )
        schedule["schedule_valid"] = not schedule["issues"]
        if require_expected and schedule["issues"]:
            raise ValueError(
                "invalid static training schedule:\n"
                + "\n".join(
                    f"- {issue}" for issue in schedule["issues"]
                )
            )
        return schedule

    def _build_evolver(self, backend: VerlPolicyBackend) -> RQEvolver:
        archive = MAPElitesArchive(
            **asdict(self.rq_config.archive),
            select_ignores_uncertainty=self.rq_config.evolution.select_ignores_uncertainty,
            select_ignores_variance=self.rq_config.evolution.select_ignores_variance,
            binning=self.rq_config.evolution.archive_binning,
        )
        return RQEvolver(
            archive=archive,
            backend=backend,
            evolution_config=self.rq_config.evolution,
            training_config=self.rq_config.training_data,
        )

    @staticmethod
    def _next_outer_iteration(archive_dir: Path) -> int:
        """Next free outer-iteration index = max archive_iter{N}.json on disk + 1.

        Used on resume so the EvolvingSampler continues numbering past the prior
        run's snapshots instead of restarting at 0 and overwriting them. Returns 0
        when no snapshot exists (fresh archive dir), leaving the sampler at its
        default epoch=0.
        """
        import re

        max_iter = -1
        pat = re.compile(r"archive_iter(\d+)\.json$")
        try:
            for p in Path(archive_dir).glob("archive_iter*.json"):
                m = pat.search(p.name)
                if m:
                    max_iter = max(max_iter, int(m.group(1)))
        except OSError:
            return 0
        return max_iter + 1 if max_iter >= 0 else 0

    def _bootstrap_seed_archive(self, evolver: RQEvolver) -> None:
        """Evaluate every seed with the LIVE solver (real R_Q) and insert it.

        MUST be called AFTER backend.bind(trainer): evaluate_instance runs a
        solver rollout, which needs the worker group. Real R_Q gives each seed a
        real u_score, which decides who holds a cell -- H is no longer a grid
        coordinate, so a placeholder score would silently hand cells to whoever
        was inserted first. Seeds still compete per niche, so two seeds sharing
        a (GROUP, SKILL) cell keep only the higher-R_Q one — a MAP-Elites
        property, not a bug. Then refresh the training dataset so epoch 0
        trains on these seeds (with evolve_on_first_epoch=false).
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
        # this every seed scores ~random -> s_hat 0 -> empty training dataset.
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
            # Real R_Q via solver rollout over n fresh seeds x m rollouts;
            # sets program.s_hat / u_score / rq_score.
            # store_replay: bootstrap rollouts come from theta_0, which is
            # exactly the weights the first update starts from, so they are a
            # valid on-policy warm-up batch. Without them refresh_dataset has
            # nothing to build from and the run dies on an empty dataset.
            result = evolver.evaluate_programs([program], store_replay=True)[0]
            if result is not None:
                # Bootstrap is iteration -1: without a lagged score the seeds
                # would all be ineligible at t=0 and the first batch empty.
                evolver.lagged.record(program.program_id, -1, result.rq_score)
            if result is None:
                print(
                    f"[RQ-Evolve] seed eval failed (all rollouts rejected): "
                    f"{program.program_id} -- skipped"
                )
                continue
            if evolver.archive.try_insert(
                program=program,
                u_value=result.u_score,
                rq_score=result.rq_score,
            ):
                inserted += 1
            else:
                print(
                    f"[RQ-Evolve] seed not inserted (niche conflict / gate): "
                    f"{program.program_id} s_hat={result.s_hat:.2f} h={result.u_score:.3f}"
                )
        print(
            f"[RQ-Evolve] bootstrapped {inserted}/{len(seeds)} seeds with real R_Q; "
            f"{len(evolver.archive.champions())} champions on a "
            f"{evolver.archive.n_group_bins}x{evolver.archive.n_skill_bins} "
            f"GROUP x SKILL grid"
        )
        # warmup: the seed scores were taken at bootstrap, so there is no
        # earlier iteration to lag against. Every later refresh is lagged.
        evolver.refresh_dataset(warmup=True)
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
        # When math-benchmark eval is on, grade the val set on the trainer's MAIN
        # thread instead of the agent loop's reward worker thread (where
        # math_verify's SIGALRM timeout can't fire and a pathological boxed answer
        # pegs CPU -> vLLM starves -> GPU 0% mid-eval). See eval_trainer.py.
        math_eval_on = bool(
            getattr(self.rq_config, "math_eval", None) and self.rq_config.math_eval.enabled
        )
        if math_eval_on:
            from .eval_trainer import make_validating_trainer_cls

            RayPPOTrainer = make_validating_trainer_cls(RayPPOTrainer)
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


# Curated Solver(GRPO) metrics -> clean ``solver/*`` aliases. verl logs these
# under actor/ , critic/ , response_length/ ; we add a parallel solver/ copy so
# each role has its own wandb panel without touching verl's own keys.
_SOLVER_METRIC_ALIAS = {
    "actor/pg_loss": "solver/pg_loss",
    "actor/entropy": "solver/entropy",
    "actor/kl_loss": "solver/kl_loss",
    "actor/grad_norm": "solver/grad_norm",
    "actor/lr": "solver/lr",
    "actor/ppo_kl": "solver/ppo_kl",
    "actor/pg_clipfrac": "solver/pg_clipfrac",
    "critic/score/mean": "solver/score_mean",
    "critic/rewards/mean": "solver/reward_mean",
    "response_length/mean": "solver/response_len_mean",
    "response_length/clip_ratio": "solver/response_len_clip_ratio",
}


def _install_solver_metric_alias() -> None:
    """Monkeypatch ``wandb.log`` to add ``solver/*`` aliases for the Solver's
    verl metrics. Additive and idempotent; wrapped so it can never break logging.

    The actor/* keys that pass through wandb.log are the Solver GRPO step.
    """
    try:
        import wandb
    except ImportError:
        return
    if getattr(wandb, "_rq_solver_alias_installed", False):
        return
    _orig_log = wandb.log

    def _patched_log(data=None, *args, **kwargs):
        try:
            if isinstance(data, dict):
                extra = {
                    dst: data[src]
                    for src, dst in _SOLVER_METRIC_ALIAS.items()
                    if src in data
                }
                if extra:
                    data = {**data, **extra}
        except Exception:
            pass
        return _orig_log(data, *args, **kwargs)

    wandb.log = _patched_log
    wandb._rq_solver_alias_installed = True


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
