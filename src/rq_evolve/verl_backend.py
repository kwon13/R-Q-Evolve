import numpy as np
import torch

from .backends import EvolutionBackend, PendingRollouts, RolloutRecord
from .program import ProblemInstance
from .prompts import MutationTask, build_solver_prompt
from .reward import answers_match, extract_boxed


class VerlPolicyBackend(EvolutionBackend):
    """Use the live verl actor/rollout worker as the Questioner/Solver model.

    The backend is bound after ``trainer.init_workers()`` because the worker
    group does not exist before then. Mutation and solver rollout both call
    ``trainer.actor_rollout_wg.generate_sequences``. Rollout uncertainty is
    estimated from the actor forward pass entropy returned by the installed
    verl actor worker.
    """

    def __init__(
        self,
        trainer=None,
        *,
        tokenizer=None,
        max_prompt_length: int | None = None,
        truncation: str = "left",
    ) -> None:
        self.trainer = trainer
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation
        # When True, ``_generate_with_batch`` skips its per-call vLLM wake/sleep:
        # the surrounding session woke vLLM once and will sleep it once at the
        # end (see ``begin_session`` / ``end_session``).
        self._session_active = False
        # Whether vLLM sleep/wake is usable. When the rollout runs with sleep
        # mode off (free_cache_engine/enable_sleep_mode = false), vLLM is
        # resident and cannot be slept -- ``_sleep`` becomes a no-op so we never
        # call sleep() on a non-sleep-enabled engine (and never hit the cumem
        # wake_up path). Set from config in ``bind``.
        self._sleep_enabled = True

    def bind(self, trainer) -> None:
        self.trainer = trainer
        self.tokenizer = trainer.tokenizer
        self.max_prompt_length = int(trainer.config.data.max_prompt_length)
        self.truncation = trainer.config.data.get("truncation", self.truncation)
        rollout_cfg = getattr(
            getattr(trainer.config, "actor_rollout_ref", None), "rollout", None
        )
        free_cache_engine = bool(getattr(rollout_cfg, "free_cache_engine", True))
        enable_sleep_mode = bool(getattr(rollout_cfg, "enable_sleep_mode", True))
        self._sleep_enabled = free_cache_engine and enable_sleep_mode

    def mutate(self, tasks: list[MutationTask]) -> list[str | None]:
        if not tasks:
            return []
        prompts = [task.prompt for task in tasks]
        messages = [getattr(task, "messages", None) for task in tasks]
        output, _ = self._generate_with_batch(
            prompts, messages=messages if any(messages) else None
        )
        responses = output.batch.get("responses")
        if responses is None:
            return [None] * len(tasks)
        return [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]

    # ------------------------------------------------------------------
    # Phase sessions: keep vLLM awake across a batch of generate calls and
    # pay a single cumem wake_up, instead of one wake/sleep per instance
    # (the repeated wake_up is what tripped cumem_allocator's "invalid
    # argument" under hundreds of toggles per outer iteration).
    # ------------------------------------------------------------------

    def _wake(self) -> None:
        """Push current FSDP weights into vLLM and wake it for generation.

        vLLM is launched with load_format=dummy and starts "sleeping"; without
        pushing the live actor weights it would forward with random weights ->
        NaNs -> "CUDA error: illegal memory access" inside flash-attn.
        """
        trainer = self._require_trainer()
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if checkpoint_manager is not None and hasattr(checkpoint_manager, "update_weights"):
            global_steps = int(getattr(trainer, "global_steps", 0) or 0)
            checkpoint_manager.update_weights(global_steps)

    def _sleep(self) -> None:
        """Offload vLLM KV cache + weights so the actor forward has room.

        Entropy is an actor (FSDP) forward; under sleep mode it would OOM against
        the live vLLM reservation, so it must run only after this sleep. Matches
        verl's PPO loop (update_weights -> generate -> sleep).

        No-op when sleep mode is disabled (``_sleep_enabled`` false): vLLM is
        resident, has no cumem allocator, and cannot be slept. Memory headroom is
        provided instead by a lower ``gpu_memory_utilization`` so the actor
        forward coexists with the live vLLM reservation.
        """
        if not self._sleep_enabled:
            return
        trainer = self._require_trainer()
        checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
        if checkpoint_manager is not None and hasattr(checkpoint_manager, "sleep_replicas"):
            checkpoint_manager.sleep_replicas()

    def sync_weights(self) -> None:
        """Push the current FSDP actor weights into vLLM once.

        Call at the START of an evolve phase. Weights are static for the whole
        phase (evolve does no optimizer step), so reevaluate + every inner batch
        reuse the same resident vLLM model -- no need to re-push per session.
        """
        self._wake()

    def begin_session(self) -> None:
        """Open a generate session.

        With sleep mode ON this wakes vLLM per session (restoring its offloaded
        weights/KV), as before. With sleep mode OFF vLLM is resident and its
        weights are synced once per phase via ``sync_weights`` -- so a session is
        pure state-tracking and does NOT re-push weights. While the session is
        open ``_generate_with_batch`` neither wakes nor sleeps vLLM; entropy is
        computed only after ``end_session`` (via ``finalize_rollouts``).
        """
        if self._sleep_enabled:
            self._wake()
        self._session_active = True

    def end_session(self) -> None:
        """Sleep vLLM once at the end of the phase (no-op when sleep disabled)."""
        try:
            self._sleep()
        finally:
            self._session_active = False

    def generate_rollouts(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> PendingRollouts:
        """Generate solver rollouts WITHOUT computing entropy.

        Call inside an open session (vLLM awake). The decoded responses + the
        full batch are stashed so ``finalize_rollouts`` can compute entropy once
        vLLM has been slept.
        """
        n = max(1, int(n_rollouts))
        if not instances:
            return PendingRollouts(instances=[], n_rollouts=n)
        prompts = [build_solver_prompt(inst.problem) for inst in instances]
        output, full_batch = self._generate_with_batch(prompts, n_repeat=n)
        responses = output.batch.get("responses")
        if responses is None:
            return PendingRollouts(instances=list(instances), n_rollouts=n)
        decoded = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]
        return PendingRollouts(
            instances=list(instances),
            n_rollouts=n,
            full_batch=full_batch,
            decoded=decoded,
        )

    def finalize_rollouts(self, pending: PendingRollouts) -> list[list[RolloutRecord]]:
        """Compute entropy (actor forward) and assemble grouped records.

        Call AFTER ``end_session`` so the actor forward runs with vLLM asleep.
        """
        instances = pending.instances
        if not instances:
            return []
        if pending.grouped is not None:
            return pending.grouped
        if pending.full_batch is None or not pending.decoded:
            return [[] for _ in instances]

        entropies = self._response_entropies(pending.full_batch)
        decoded = pending.decoded
        n = pending.n_rollouts
        grouped: list[list[RolloutRecord]] = []
        for ci, inst in enumerate(instances):
            rows: list[RolloutRecord] = []
            for ri in range(n):
                idx = ci * n + ri
                text = decoded[idx] if idx < len(decoded) else ""
                pred = extract_boxed(text)
                rows.append(
                    RolloutRecord(
                        response=text,
                        predicted_answer=pred,
                        correct=bool(pred and answers_match(pred, inst.answer)),
                        entropy=entropies[idx] if idx < len(entropies) else 0.0,
                    )
                )
            grouped.append(rows)
        return grouped

    def rollout(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> list[list[RolloutRecord]]:
        """Single-shot rollout: wake -> generate -> sleep -> entropy.

        Convenience wrapper that opens its own session. Batch paths in
        evolution.py call ``begin_session``/``generate_rollouts``/``end_session``/
        ``finalize_rollouts`` directly to share one wake across many instances.
        """
        if not instances:
            return []
        self.begin_session()
        try:
            pending = self.generate_rollouts(instances, n_rollouts)
        finally:
            self.end_session()
        return self.finalize_rollouts(pending)

    def _generate_with_batch(
        self, prompts: list[str], n_repeat: int = 1, messages: list | None = None
    ):
        trainer = self._require_trainer()
        batch = self._make_prompt_batch(prompts, messages=messages)
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "raw_prompt",
                "data_source",
                "reward_model",
            ],
        )
        batch.non_tensor_batch["uid"] = np.array(
            [str(i) for i in range(len(batch.batch))],
            dtype=object,
        )
        if n_repeat > 1:
            gen_batch = gen_batch.repeat(repeat_times=n_repeat, interleave=True)

        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        # verl 0.7.x retired vLLM SPMD; actor_rollout_wg.generate_sequences raises
        # NotImplementedError for the vLLM rollout. The trainer instead routes
        # rollout through async_rollout_manager (AgentLoopManager), which
        # @auto_await turns into a synchronous call that returns a DataProto.
        # The manager chunks across len(agent_loop_workers); pad to that divisor.
        # vLLM is launched with load_format=dummy and starts in "sleeping"
        # state; the trainer's normal step does checkpoint_manager.update_weights
        # before every generate_sequences to push actor (FSDP) weights into vLLM.
        # Without this, vLLM runs forward with random weights -> NaNs ->
        # "CUDA error: illegal memory access" inside flash-attn.
        rollout_manager = getattr(trainer, "async_rollout_manager", None)
        if rollout_manager is not None and hasattr(rollout_manager, "agent_loop_workers"):
            # In a session the caller already woke vLLM and will sleep it once
            # at the end; outside one, fall back to per-call wake/sleep.
            if not self._session_active:
                self._wake()
            divisor = max(1, len(rollout_manager.agent_loop_workers))
            padded, pad_size = pad_dataproto_to_divisor(gen_batch, divisor)
            out_padded = rollout_manager.generate_sequences(padded)
            if not self._session_active:
                self._sleep()
        else:
            world_size = max(1, int(getattr(trainer.actor_rollout_wg, "world_size", 1)))
            padded, pad_size = pad_dataproto_to_divisor(gen_batch, world_size)
            out_padded = trainer.actor_rollout_wg.generate_sequences(padded)
        output = unpad_dataproto(out_padded, pad_size=pad_size)
        full_batch = batch.repeat(repeat_times=n_repeat, interleave=True).union(output)
        return output, full_batch

    def _response_entropies(self, logprob_batch) -> list[float]:
        trainer = self._require_trainer()
        logprob_batch.batch["response_mask"] = _compute_response_mask(logprob_batch)
        logprob_batch.meta_info["global_token_num"] = torch.sum(
            logprob_batch.batch["attention_mask"], dim=-1
        ).tolist()

        from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

        world_size = max(1, int(getattr(trainer.actor_rollout_wg, "world_size", 1)))
        padded, pad_size = pad_dataproto_to_divisor(logprob_batch, world_size)
        old_padded = self._compute_actor_log_probs(padded)
        old = unpad_dataproto(old_padded, pad_size=pad_size)
        entropy = old.batch.get("entropys")
        if entropy is None:
            entropy = old.batch.get("entropies")
        if entropy is None:
            return [0.0] * logprob_batch.batch.batch_size[0]

        response_mask = logprob_batch.batch["response_mask"]
        values: list[float] = []
        for row, mask in zip(entropy, response_mask):
            valid = row[mask.bool()]
            values.append(float(valid.mean().item()) if valid.numel() else 0.0)
        return values

    def _chat_template_len(self, text: str) -> int:
        """Token count of ``text`` after the chat template the agent loop applies."""
        tok = self._require_tokenizer()
        try:
            ids = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=True,
            )
        except Exception:
            # tokenizer without a chat template: fall back to raw encode
            ids = tok.encode(text, add_special_tokens=True)
        return len(ids)

    def _truncate_to_chat_budget(self, prompt: str, max_prompt_length: int) -> str:
        """Cap ``prompt`` so the chat-templated form fits ``max_prompt_length``.

        Keeps the head (the instructions / output-format spec live at the start
        of both the mutation and solver prompts) and drops trailing content
        (the parent example program), which is the safe end to clip. Verifies
        against the real chat-template length and trims further if token-merge
        effects at the boundary still overflow.
        """
        if self._chat_template_len(prompt) <= max_prompt_length:
            return prompt
        tok = self._require_tokenizer()
        # fixed tokens the template adds around the content (role markers,
        # default system prompt, generation prompt)
        overhead = self._chat_template_len("")
        margin = 16
        content_ids = tok.encode(prompt, add_special_tokens=False)
        budget = max(0, max_prompt_length - overhead - margin)
        truncated = tok.decode(content_ids[:budget], skip_special_tokens=True)
        # re-check: decode/re-encode round trips can shift the count slightly
        while budget > 0 and self._chat_template_len(truncated) > max_prompt_length:
            budget = max(0, budget - 64)
            truncated = tok.decode(content_ids[:budget], skip_special_tokens=True)
        return truncated

    def _truncate_messages_to_budget(self, messages: list[dict], max_prompt_length: int) -> list[dict]:
        """Cap a multi-turn conversation to the budget, clipping middle turns.

        The system turn (rules) and the final user turn (rejection reason + fix
        request) are preserved intact; the long, clippable middle -- the
        original-task user turn and the assistant (rejected output) turn -- is
        shortened from its tail until the chat-templated conversation fits
        ``max_prompt_length``. Mirrors ``_truncate_to_chat_budget`` but at the
        message granularity so the fix instruction is never the part that gets cut.
        """
        tok = self._require_tokenizer()

        def total(msgs):
            return len(
                tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
            )

        if total(messages) <= max_prompt_length:
            return messages
        msgs = [dict(m) for m in messages]
        clippable = list(range(1, len(msgs) - 1))  # exclude system + final turn
        for _ in range(256):
            over = total(msgs) - max_prompt_length
            if over <= 0:
                break
            sizes = [
                (len(tok.encode(msgs[i]["content"], add_special_tokens=False)), i)
                for i in clippable
            ]
            sizes = [(s, i) for s, i in sizes if s > 0]
            if not sizes:
                break
            _, i = max(sizes)
            ids = tok.encode(msgs[i]["content"], add_special_tokens=False)
            keep = max(0, len(ids) - max(64, over + 16))
            clipped = tok.decode(ids[:keep], skip_special_tokens=True) if keep else ""
            msgs[i]["content"] = (clipped + "\n...[truncated]...") if keep else "...[truncated]..."
        return msgs

    def _make_prompt_batch(self, prompts: list[str], messages: list | None = None):
        import verl.utils.torch_functional as verl_F
        from verl import DataProto

        tokenizer = self._require_tokenizer()
        max_prompt_length = self.max_prompt_length or 1024
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        if messages is None:
            messages = [None] * len(prompts)

        # Per item, build (a) the rendered prompt text used for input_ids and
        # (b) the chat-message list handed to the agent loop as raw_prompt.
        # Multi-turn items (fix-retry) carry a [system,user,assistant,user]
        # conversation; single-turn items wrap the text as one user message.
        # Both are length-capped: verl 0.7.x's AgentLoopWorker re-tokenizes
        # raw_prompt with the chat template and NEVER truncates it
        # (tokenizer.pad(padding="max_length") only right-pads), so an
        # over-length prompt would otherwise crash _postprocess's torch.cat.
        rendered: list[str] = []
        raw_msgs: list[list[dict]] = []
        for p, m in zip(prompts, messages):
            if m is not None:
                m = self._truncate_messages_to_budget(m, max_prompt_length)
                rendered.append(
                    tokenizer.apply_chat_template(
                        m, add_generation_prompt=True, tokenize=False
                    )
                )
                raw_msgs.append(m)
            else:
                p = self._truncate_to_chat_budget(p, max_prompt_length)
                rendered.append(p)
                raw_msgs.append([{"role": "user", "content": p}])

        model_inputs = tokenizer(
            rendered,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        )
        input_ids, attention_mask = verl_F.postprocess_data(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            max_length=max_prompt_length,
            pad_token_id=pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        position_ids = _compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = [
            tokenizer.encode(text, add_special_tokens=False)[-max_prompt_length:]
            for text in rendered
        ]
        # AgentLoopWorker reads kwargs["raw_prompt"] as chat messages and applies
        # the chat template itself (single user turn, or the full fix conversation).
        raw_prompt_arr = np.empty(len(prompts), dtype=object)
        for i, m in enumerate(raw_msgs):
            raw_prompt_arr[i] = m
        raw_prompt_ids_arr = np.empty(len(prompts), dtype=object)
        for i, ids in enumerate(raw_prompt_ids):
            raw_prompt_ids_arr[i] = ids
        # The agent loop unconditionally invokes the reward loop worker (the
        # naive reward manager reads non_tensor_batch["data_source"] and
        # non_tensor_batch["reward_model"]["ground_truth"]). Our backend
        # computes its own reward in evolution.py and discards verl's value,
        # but the call still has to type-check -> provide placeholders.
        data_source_arr = np.array(["rq_evolve"] * len(prompts), dtype=object)
        reward_model_arr = np.empty(len(prompts), dtype=object)
        for i in range(len(prompts)):
            reward_model_arr[i] = {"ground_truth": ""}
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids_arr,
            "raw_prompt": raw_prompt_arr,
            "data_source": data_source_arr,
            "reward_model": reward_model_arr,
        }
        return DataProto.from_single_dict(data)

    def _require_trainer(self):
        if self.trainer is None:
            raise RuntimeError("VerlPolicyBackend is not bound to a trainer")
        return self.trainer

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("VerlPolicyBackend is not bound to a tokenizer")
        return self.tokenizer

    def _compute_actor_log_probs(self, padded_batch):
        worker_group = self._require_trainer().actor_rollout_wg
        if hasattr(worker_group, "compute_log_prob"):
            return worker_group.compute_log_prob(padded_batch)
        if hasattr(worker_group, "compute_log_probs"):
            try:
                return worker_group.compute_log_probs(padded_batch, calculate_entropy=True)
            except TypeError:
                return worker_group.compute_log_probs(padded_batch)
        raise RuntimeError("verl actor worker exposes no compute_log_prob(s) method")


def _compute_response_mask(data):
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _compute_position_id_with_mask(attention_mask):
    try:
        from verl.utils.model import compute_position_id_with_mask
    except ImportError:
        from verl.utils.model_utils import compute_position_id_with_mask

    return compute_position_id_with_mask(attention_mask)
