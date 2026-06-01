"""verl-backed mutation and rollout backend."""

from __future__ import annotations

import numpy as np
import torch

from .backends import EvolutionBackend, RolloutRecord
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

    def bind(self, trainer) -> None:
        self.trainer = trainer
        self.tokenizer = trainer.tokenizer
        self.max_prompt_length = int(trainer.config.data.max_prompt_length)
        self.truncation = trainer.config.data.get("truncation", self.truncation)

    def mutate(self, tasks: list[MutationTask]) -> list[str | None]:
        if not tasks:
            return []
        prompts = [task.prompt for task in tasks]
        output, _ = self._generate_with_batch(prompts)
        responses = output.batch.get("responses")
        if responses is None:
            return [None] * len(tasks)
        return [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]

    def rollout(
        self,
        instances: list[ProblemInstance],
        n_rollouts: int,
    ) -> list[list[RolloutRecord]]:
        if not instances:
            return []

        prompts = [build_solver_prompt(inst.problem) for inst in instances]
        output, full_batch = self._generate_with_batch(
            prompts,
            n_repeat=max(1, int(n_rollouts)),
        )
        responses = output.batch.get("responses")
        if responses is None:
            return [[] for _ in instances]

        decoded = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]
        entropies = self._response_entropies(full_batch)

        grouped: list[list[RolloutRecord]] = []
        n = max(1, int(n_rollouts))
        for ci, inst in enumerate(instances):
            rows: list[RolloutRecord] = []
            for ri in range(n):
                idx = ci * n + ri
                text = decoded[idx]
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

    def _generate_with_batch(self, prompts: list[str], n_repeat: int = 1):
        trainer = self._require_trainer()
        batch = self._make_prompt_batch(prompts)
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
            checkpoint_manager = getattr(trainer, "checkpoint_manager", None)
            if checkpoint_manager is not None and hasattr(checkpoint_manager, "update_weights"):
                global_steps = int(getattr(trainer, "global_steps", 0) or 0)
                checkpoint_manager.update_weights(global_steps)
            divisor = max(1, len(rollout_manager.agent_loop_workers))
            padded, pad_size = pad_dataproto_to_divisor(gen_batch, divisor)
            out_padded = rollout_manager.generate_sequences(padded)
            # Put vLLM back to sleep so its KV cache + weights don't pin GPU
            # memory while we run the actor's compute_log_prob (entropy) next.
            # Matches verl's PPO loop (update_weights -> generate -> sleep).
            if checkpoint_manager is not None and hasattr(checkpoint_manager, "sleep_replicas"):
                checkpoint_manager.sleep_replicas()
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

    def _make_prompt_batch(self, prompts: list[str]):
        import verl.utils.torch_functional as verl_F
        from verl import DataProto

        tokenizer = self._require_tokenizer()
        max_prompt_length = self.max_prompt_length or 1024
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        model_inputs = tokenizer(
            prompts,
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
            tokenizer.encode(prompt, add_special_tokens=False)[-max_prompt_length:]
            for prompt in prompts
        ]
        # verl 0.7.x's AgentLoopWorker (SingleTurnAgentLoop.run) reads
        # kwargs["raw_prompt"] as chat-format messages and applies the chat
        # template itself. Wrap each text prompt as a single-turn user message.
        raw_prompt_arr = np.empty(len(prompts), dtype=object)
        for i, p in enumerate(prompts):
            raw_prompt_arr[i] = [{"role": "user", "content": p}]
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
