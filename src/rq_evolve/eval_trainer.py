"""Main-thread math-benchmark validation for the verl PPO/GRPO trainer.

Why this exists
---------------
verl's stock ``RayPPOTrainer._validate`` relies on the reward score the agent
loop computes *inside* ``async_rollout_manager.generate_sequences`` -- and that
grading runs in the reward worker's thread pool (``NaiveRewardManager`` ->
``loop.run_in_executor``), i.e. NOT the main thread. ``math_verify`` guards its
sympy ``parse``/``verify`` with ``signal.SIGALRM``, which only arms on the main
thread; off it, a pathological boxed answer (common for the base model on hard
competition math) makes sympy spin and our watchdog (reward._ensure_math_verify_
thread_safe) can only leak a CPU-pegged daemon thread. A burst of those during a
~1.5k-prompt eval saturates the worker CPUs, vLLM generation in the same agent
loop workers starves, and GPU utilization drops to 0% mid-eval.

evo-sample never hit this: its custom eval loop grades on the driver's MAIN
thread, where ``math_verify``'s native SIGALRM timeout actually fires. This
module ports that approach. Two halves work together:

  1. The math-benchmark val rows carry ``extra_info[SKIP_WORKER_GRADE_KEY]=True``
     (see math_eval.MathBenchmarkDataset), so ``reward.compute_score`` returns a
     placeholder for them on the worker thread -- no sympy there.
  2. ``RQValidatingTrainer._validate`` re-grades the decoded responses on the
     main thread via ``math_eval.grade_eval`` (SIGALRM works) and feeds the
     scores back through verl's own metric aggregation, so the wandb keys
     (``val-core/<benchmark>/acc/...``) are unchanged. ``grade_eval`` is kept
     identical to the offline checkpoint grader (scripts/eval_vllm_math.py):
     brace-matched extract_boxed + \boxed-wrapped math_verify, NO length guard,
     so val-core == the eval_vllm_math.py number. We deliberately do NOT use
     ``reward.answers_match`` here -- that grader keeps the length guard for
     reward-worker-thread safety, which causes false negatives on long answers.

The override mirrors verl's ``_validate`` generation/plumbing (agent-loop
generate, val_kwargs, per-batch interleave, ``_val_metrics_update``); only the
scoring block differs. It is pinned to the installed verl in azr-bw -- if verl
is upgraded, re-diff against ``trainer/ppo/ray_trainer.py:_validate``.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np

from .math_eval import grade_eval

logger = logging.getLogger(__name__)


def make_validating_trainer_cls(base_cls):
    """Return a subclass of ``base_cls`` (verl RayPPOTrainer) that grades the
    math-benchmark validation set on the main thread."""

    class RQValidatingTrainer(base_cls):
        def _validate(self, merged: bool = False):
            from verl import DataProto
            from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

            data_source_lst = []
            reward_extra_infos_dict: dict[str, list] = defaultdict(list)

            sample_inputs = []
            sample_outputs = []
            sample_gts = []
            sample_scores = []
            sample_turns = []
            sample_uids = []

            import uuid

            for test_data in self.val_dataloader:
                test_batch = DataProto.from_single_dict(test_data)

                if "uid" not in test_batch.non_tensor_batch:
                    test_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                    )

                test_batch = test_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
                )

                ground_truths = [
                    item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None)
                    for item in test_batch
                ]
                sample_gts.extend(ground_truths)

                test_gen_batch = self._get_gen_batch(test_batch)
                test_gen_batch.meta_info = {
                    "eos_token_id": self.tokenizer.eos_token_id,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "recompute_log_prob": False,
                    "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                    "validate": True,
                    "global_steps": self.global_steps,
                }

                size_divisor = self.config.actor_rollout_ref.rollout.agent.num_workers
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
                # Agent-loop generation. The reward worker still runs for these
                # rows but compute_score short-circuits (extra_info skip flag), so
                # no sympy fires off the main thread -> vLLM never starves.
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(
                    test_gen_batch_padded
                )
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

                # Decode + grade on the MAIN thread (math_verify SIGALRM works here).
                output_ids = test_output_gen_batch.batch["responses"]
                output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
                sample_outputs.extend(output_texts)

                test_batch = test_batch.union(test_output_gen_batch)
                test_batch.meta_info["validate"] = True

                input_ids = test_batch.batch["prompts"]
                input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
                sample_inputs.extend(input_texts)
                sample_uids.extend(test_batch.non_tensor_batch["uid"])

                # eval_vllm_math.py parity: extract_boxed + \boxed-wrapped
                # math_verify, no length guard. Main thread -> SIGALRM works.
                scores = [
                    1.0 if (gt is not None and grade_eval(resp, str(gt))) else 0.0
                    for resp, gt in zip(output_texts, ground_truths)
                ]
                sample_scores.extend(scores)
                reward_extra_infos_dict["reward"].extend(scores)
                # "acc" is the variable _val_metrics_update treats as core, so the
                # benchmark accuracy lands under val-core/<data_source>/acc/...
                reward_extra_infos_dict["acc"].extend(scores)

                if "__num_turns__" in test_batch.non_tensor_batch:
                    sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

                data_source_lst.append(
                    test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores))
                )

            self._maybe_log_val_generations(
                inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores
            )

            val_data_dir = self.config.trainer.get("validation_data_dir", None)
            if val_data_dir:
                self._dump_generations(
                    inputs=sample_inputs,
                    outputs=sample_outputs,
                    gts=sample_gts,
                    scores=sample_scores,
                    reward_extra_infos_dict=reward_extra_infos_dict,
                    dump_path=val_data_dir,
                )

            for key_info, lst in reward_extra_infos_dict.items():
                assert len(lst) == 0 or len(lst) == len(sample_scores), (
                    f"{key_info}: {len(lst)=}, {len(sample_scores)=}"
                )

            if merged:
                return {
                    "data_sources": data_source_lst,
                    "sample_uids": sample_uids,
                    "sample_turns": sample_turns,
                    "reward_extra_infos_dict": reward_extra_infos_dict,
                }
            data_sources = np.concatenate(data_source_lst, axis=0)
            metrics = self._val_metrics_update(
                data_sources, sample_uids, reward_extra_infos_dict, sample_turns
            )
            logger.info(
                "[MathEval] main-thread grading done: %d samples across %d data_source(s)",
                len(sample_scores),
                len(set(map(str, data_sources.tolist()))),
            )
            return metrics

    RQValidatingTrainer.__name__ = f"RQValidating{base_cls.__name__}"
    RQValidatingTrainer.__qualname__ = RQValidatingTrainer.__name__
    return RQValidatingTrainer
