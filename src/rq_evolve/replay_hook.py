"""Serve the solver's training batch from the re-scoring rollouts.

verl's trainer calls ``async_rollout_manager.generate_sequences`` once per step.
Left alone it samples the policy again over prompts the re-scoring pass already
rolled out under the SAME weights -- the second half of a doubled budget. This
wraps that call so a prompt already in the replay buffer is answered from it and
only an unseen prompt reaches the engine.

Correctness rests on one thing, and it is checked rather than assumed: the row
handed back must belong to the prompt that was asked for. A silent mismatch here
would train the policy on someone else's response and nothing downstream could
detect it, so every served row's problem text is compared against the request's,
and ANY discrepancy makes the whole batch fall through to real generation with a
loud log line. Serving less is recoverable; serving the wrong thing is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ReplayServeStats:
    steps: int = 0
    served_steps: int = 0
    served_rows: int = 0
    generated_rows: int = 0
    # generate_sequences calls that are not training steps at all. Counted
    # apart from `steps` so hit_rate stays a statement about training.
    non_training_calls: int = 0
    misses: dict[str, int] = field(default_factory=dict)

    def miss(self, reason: str) -> None:
        self.misses[reason] = self.misses.get(reason, 0) + 1

    def to_wandb(self, prefix: str = "replay/") -> dict:
        payload = {
            f"{prefix}steps": self.steps,
            f"{prefix}served_steps": self.served_steps,
            f"{prefix}served_rows": self.served_rows,
            f"{prefix}generated_rows": self.generated_rows,
            f"{prefix}hit_rate": (
                self.served_steps / self.steps if self.steps else 0.0
            ),
            f"{prefix}non_training_calls": self.non_training_calls,
        }
        payload.update({f"{prefix}miss_{k}": v for k, v in self.misses.items()})
        return payload


def _row_key(extra: Any) -> tuple[str, int] | None:
    """(program_id, seed) for one row, or None when the row is not replayable."""
    if not isinstance(extra, dict):
        return None
    program_id = extra.get("program_id")
    seed = extra.get("seed")
    if program_id is None or seed is None:
        return None
    try:
        return (str(program_id), int(seed))
    except (TypeError, ValueError):
        return None


def _row_problem(raw_prompt: Any) -> str | None:
    """The user turn of one row's chat prompt."""
    try:
        for message in reversed(list(raw_prompt)):
            if message.get("role") == "user":
                return str(message.get("content", ""))
    except (TypeError, AttributeError):
        return None
    return None


class ReplayRolloutHook:
    """Wraps ``generate_sequences`` on the rollout manager INSTANCE.

    Instance-level, like ``PolicyVersionTracker.install``: no verl edit, and the
    wrapper disappears with the object.
    """

    def __init__(self, buffer, *, group_size: int) -> None:
        self.buffer = buffer
        self.group_size = int(group_size)
        self.stats = ReplayServeStats()
        self._installed_on: Any = None
        self._original = None
        self._reported_meta_collisions: set[str] = set()

    def install(self, manager) -> bool:
        if manager is None or not hasattr(manager, "generate_sequences"):
            return False
        if self._installed_on is manager:
            return True
        self._original = manager.generate_sequences

        def generate_sequences(gen_batch, *args, **kwargs):
            served = self._try_serve(gen_batch)
            if served is not None:
                return served
            return self._original(gen_batch, *args, **kwargs)

        manager.generate_sequences = generate_sequences
        self._installed_on = manager
        return True

    def uninstall(self) -> None:
        if self._installed_on is not None and self._original is not None:
            self._installed_on.generate_sequences = self._original
        self._installed_on = None
        self._original = None

    # ------------------------------------------------------------------
    def _is_non_training_call(self, gen_batch) -> bool:
        """True when this generate_sequences call is not a training step.

        Three callers share this method. Only the trainer's is replayable: its
        rows come from the dynamic dataset and carry ``extra_info`` naming the
        (program_id, seed) whose rollouts are in the buffer. The other two must
        always generate --

        * this backend's own evolve phase (mutation, judge, R_Q scoring), whose
          batches ``verl_backend._make_prompt_batch`` builds with exactly
          data_source / raw_prompt / raw_prompt_ids / reward_model, and
        * verl's validation batches, which carry no extra_info either.

        They used to fall through ``_plan`` and be counted and logged as steps,
        so an iteration with three evolve-phase calls printed "step 2/3/4:
        generating instead" between two genuinely replayed training steps and
        reported replay/hit_rate 0.4 for a run whose training hit rate was 1.0.
        """
        non_tensor = getattr(gen_batch, "non_tensor_batch", None) or {}
        if "extra_info" in non_tensor and "raw_prompt" in non_tensor:
            return False
        self.stats.non_training_calls += 1
        if self.stats.non_training_calls == 1:
            print(
                "[replay] passing through generate_sequences calls that are not "
                "training steps (evolve-phase mutation/judge/scoring, and "
                "validation): they carry no extra_info, so there is nothing "
                "stored to serve. Expected; logged once, then counted only.",
                flush=True,
            )
        return True

    def _try_serve(self, gen_batch):
        """The stored rows for this batch, or None to fall through."""
        if self._is_non_training_call(gen_batch):
            return None
        self.stats.steps += 1
        rows = self._plan(gen_batch)
        if rows is None:
            self.stats.generated_rows += self._batch_size(gen_batch)
            print(
                f"[replay] training step {self.stats.steps}: generating instead "
                f"({dict(self.stats.misses)})",
                flush=True,
            )
            return None
        try:
            from verl.protocol import DataProto

            served = DataProto.concat(rows)
        except Exception as exc:  # a broken concat must not corrupt the step
            self.stats.miss(f"concat_failed:{type(exc).__name__}")
            self.stats.generated_rows += self._batch_size(gen_batch)
            print(f"[replay] concat failed, generating instead: {exc!r}", flush=True)
            return None
        request_meta = dict(getattr(gen_batch, "meta_info", {}) or {})
        request_meta.setdefault("timing", {})
        collisions = self._install_request_meta(served, request_meta)
        new_collisions = collisions - self._reported_meta_collisions
        if new_collisions:
            print(
                "[replay] removed payload copies of request metadata before "
                f"TensorDict conversion: {sorted(new_collisions)}",
                flush=True,
            )
            self._reported_meta_collisions.update(new_collisions)
        self.stats.served_steps += 1
        self.stats.served_rows += self._batch_size(served)
        print(
            f"[replay] served step from the buffer: {self._batch_size(served)} "
            f"rows, no sampling pass "
            f"({self.stats.served_steps}/{self.stats.steps} training steps "
            f"so far)",
            flush=True,
        )
        return served

    @staticmethod
    def _install_request_meta(served, request_meta: dict) -> set[str]:
        """Make request metadata canonical on a replayed generation batch.

        verl 0.9 converts a DataProto by first combining its tensor and
        per-row non-tensor keys, then inserting every ``meta_info`` key.  It
        rejects a key present on both sides even when the values agree. Agent
        loop outputs can retain controls such as ``global_steps`` per row,
        while the trainer supplies the same control as request metadata. A
        real generation call treats the request value as authoritative, so a
        replay call must do the same and discard only those duplicate control
        copies from the stored payload.
        """
        data_keys: set[str] = set()
        tensors = getattr(served, "batch", None)
        if tensors is not None:
            try:
                data_keys.update(tensors.keys())
            except Exception:
                pass
        non_tensors = getattr(served, "non_tensor_batch", None)
        if non_tensors is not None:
            data_keys.update(non_tensors.keys())

        collisions = data_keys & set(request_meta)
        for key in collisions:
            if tensors is not None:
                try:
                    if key in tensors:
                        tensors.pop(key)
                except Exception:
                    pass
            if non_tensors is not None:
                non_tensors.pop(key, None)
        served.meta_info = request_meta
        return collisions

    @staticmethod
    def _batch_size(batch) -> int:
        """Row count, from whichever side of the DataProto actually has rows.

        The training gen_batch has NO tensors: the dataset returns raw_prompt
        as a chat-message list and one dummy tensor, and ``_get_gen_batch``
        pops no tensor keys, so ``batch.batch`` is empty while the non-tensor
        arrays carry every row. Reading only the tensor side reported 0 rows
        and declined every step.
        """
        sizes: list[int] = []
        tensors = getattr(batch, "batch", None)
        if tensors is not None:
            try:
                sizes.append(int(tensors.batch_size[0]))
            except Exception:
                pass
        for array in (getattr(batch, "non_tensor_batch", None) or {}).values():
            try:
                sizes.append(len(array))
            except Exception:
                pass
        return max(sizes) if sizes else 0

    def _plan(self, gen_batch) -> list | None:
        """One cached DataProto row per request row, or None if anything is off."""
        non_tensor = getattr(gen_batch, "non_tensor_batch", None) or {}
        extras = non_tensor.get("extra_info")
        prompts = non_tensor.get("raw_prompt")
        if extras is None or prompts is None:
            self.stats.miss("no_extra_info")
            print(
                "[replay] no extra_info/raw_prompt on the batch; keys="
                f"{sorted(non_tensor)}",
                flush=True,
            )
            return None

        size = self._batch_size(gen_batch)
        if size == 0 or len(extras) != size or len(prompts) != size:
            self.stats.miss("shape_mismatch")
            print(
                f"[replay] shape mismatch: rows={size} extras={len(extras)} "
                f"prompts={len(prompts)}",
                flush=True,
            )
            return None

        n = self.group_size
        if size % n != 0:
            # The trainer repeats each prompt rollout.n times; if that does not
            # match m, the cached group cannot line up row-for-row.
            self.stats.miss("rollout_n_mismatch")
            return None

        planned: list = []
        for start in range(0, size, n):
            key = _row_key(extras[start])
            if key is None:
                self.stats.miss("unkeyed_row")
                return None
            program_id, seed = key
            # Every row of the group must be the same instance, or the trainer
            # is not repeating the way this assumes.
            if any(_row_key(extras[start + k]) != key for k in range(n)):
                self.stats.miss("group_not_contiguous")
                return None

            group = self._lookup(program_id, seed)
            if group is None:
                self.stats.miss("not_in_buffer")
                print(
                    f"[replay] {program_id}/seed={seed} not in the buffer "
                    f"(buffer holds {len(self.buffer.groups)} programs)",
                    flush=True,
                )
                return None
            payload = getattr(group, "payload", None)
            if payload is None:
                self.stats.miss("no_payload")
                return None
            if self._batch_size(payload) != n:
                self.stats.miss("payload_size_mismatch")
                return None

            # The check that makes a silent swap impossible.
            wanted = _row_problem(prompts[start])
            stored = getattr(getattr(group, "instance", None), "problem", None)
            if wanted is None or stored is None or wanted.strip() != stored.strip():
                self.stats.miss("prompt_mismatch")
                print(
                    "[replay] prompt mismatch for "
                    f"{program_id}/seed={seed}; generating instead",
                    flush=True,
                )
                return None
            planned.append(payload)
        return planned

    def _lookup(self, program_id: str, seed: int):
        for group in self.buffer.get(program_id):
            if int(getattr(group.instance, "seed", -1)) == int(seed):
                return group
        return None
