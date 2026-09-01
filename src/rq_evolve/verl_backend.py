import time as _time
from collections import deque

import numpy as np
import torch
import verl.utils.torch_functional as verl_F
from omegaconf import OmegaConf
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.utils.model import compute_position_id_with_mask

from .backends import EvolutionBackend, PendingRollouts, RolloutRecord
from .program import ProblemInstance
from .prompts import MutationTask, build_solver_messages
from .reward import answers_match, extract_boxed
from .verifier import normalize_verifier


class VerlPolicyBackend(EvolutionBackend):
    """Use the live verl actor/rollout worker for mutation and solver rollout.

    The backend is bound after ``trainer.init_workers()`` because the worker
    group does not exist before then. Mutation and solver rollout both call
    ``trainer.actor_rollout_wg.generate_sequences``. Rollout u_score is
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
        # Rollout context window, read from config in ``bind``. The evaluator
        # gate uses it to drop a candidate whose prompt would not fit, instead
        # of letting the batched generate raise and abort the run.
        self.max_model_len: int | None = None
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
        # Streaming (producer-consumer) rollout state; wired by
        # ``configure_streaming`` after bind. None -> legacy whole-batch path.
        self._async_cfg = None
        self._version_tracker = None
        self._sample_logger = None
        self._rollout_metrics = None
        self._streaming_in_flight = False
        # Outer-iteration index for JSONL sample records; the EvolvingSampler
        # updates this before each evolve phase.
        self.current_iteration = -1
        # (token id, logprob) of each row's first generated token from the most
        # recent mutate() call, aligned with its task list. Only filled when the
        # tasks asked for logprobs; restricted binary readback gates consume it.
        self.last_mutation_logprobs: list[tuple[int, float] | None] = []

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
        window = getattr(rollout_cfg, "max_model_len", None)
        self.max_model_len = int(window) if window else None
        # Which logits the engine turns into logprobs. Restricted binary
        # readback arithmetic rests on this; see _require_two_way_logprobs.
        self._logprobs_mode = str(
            getattr(rollout_cfg, "logprobs_mode", "processed_logprobs")
        )

    def mutate(self, tasks: list[MutationTask]) -> list[str | None]:
        if not tasks:
            return []
        prompts = [task.prompt for task in tasks]
        messages = [getattr(task, "messages", None) for task in tasks]
        limits = {
            int(task.max_output_tokens)
            for task in tasks
            if task.max_output_tokens is not None
        }
        temperatures = {
            float(task.temperature)
            for task in tasks
            if task.temperature is not None
        }
        top_ps = {
            float(task.top_p)
            for task in tasks
            if task.top_p is not None
        }
        if len(temperatures) > 1 or len(top_ps) > 1:
            raise ValueError(
                "one mutation batch must use one temperature/top_p pair; "
                "group tasks by stage before calling mutate()"
            )
        # logprobs / allowed_token_ids ride on DataProto.meta_info, which is
        # per-batch, not per-row -- so like temperature they must agree across
        # the batch. Binary readback paths build every call from one template,
        # so they do; a mixed batch is a caller bug, not a silent
        # "some rows get logprobs".
        logprob_opts = {
            int(task.logprobs) for task in tasks if task.logprobs is not None
        }
        allowed_opts = {
            tuple(task.allowed_token_ids)
            for task in tasks
            if task.allowed_token_ids is not None
        }
        if len(logprob_opts) > 1 or len(allowed_opts) > 1:
            raise ValueError(
                "one mutation batch must use one logprobs/allowed_token_ids "
                "pair; group tasks by stage before calling mutate()"
            )
        if logprob_opts and any(task.logprobs is None for task in tasks):
            raise ValueError(
                "logprobs must be set on every task in the batch or on none"
            )
        allowed = next(iter(allowed_opts), None)
        if allowed is not None and logprob_opts:
            self._require_two_way_logprobs()
        max_tokens = min(limits) if limits else None
        self.last_mutation_logprobs = [None] * len(tasks)
        output, _ = self._generate_with_batch(
            prompts,
            messages=messages if any(messages) else None,
            max_tokens=max_tokens,
            temperature=next(iter(temperatures), None),
            top_p=next(iter(top_ps), None),
            logprobs=next(iter(logprob_opts), None),
            allowed_token_ids=list(allowed) if allowed is not None else None,
        )
        responses = output.batch.get("responses")
        if responses is None:
            return [None] * len(tasks)
        self.last_mutation_logprobs = self._first_token_logprobs(output, len(tasks))
        return [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]

    def _require_two_way_logprobs(self) -> None:
        """Refuse to score a restricted-vocabulary call under raw logprobs.

        A binary readback reads ONE token's logprob and takes the other side to
        be its complement. That is exact only when the logprob is a log_softmax
        over the MASKED logits -- after ``allowed_token_ids`` has set every
        other token to -inf. vLLM does that only under
        ``logprobs_mode="processed_logprobs"``: at sampler.py:88 the mask is
        applied to the logits, and at sampler.py:164-165 the greedy branch
        recomputes the logprobs from those masked logits. Under "raw_logprobs"
        it keeps the logprobs taken at sampler.py:80-81, BEFORE the mask, so
        P(YES) + P(NO) < 1 and the complement overstates the unsampled side --
        by a different amount on every call.

        verl's own default is "processed_logprobs" (workers/config/rollout.py),
        so this normally passes. It raises rather than warns because the failure
        is silent: the run would finish, and every archive coordinate in it
        would be wrong.
        """
        mode = getattr(self, "_logprobs_mode", "processed_logprobs")
        if mode != "processed_logprobs":
            raise ValueError(
                f"actor_rollout_ref.rollout.logprobs_mode is {mode!r}; the "
                "restricted binary readback gate needs 'processed_logprobs' for its "
                "allowed_token_ids logprob to be a normalised two-way "
                "probability. Set it, or disable the binary DOMAIN labeler."
            )

    def _first_token_logprobs(self, output, n: int) -> list[tuple[int, float] | None]:
        """(token id, logprob) of each row's FIRST generated token.

        ``_generate_with_batch`` unpads before returning, so row k is task k.
        ``rollout_log_probs`` is only present when the request asked for
        logprobs (agent_loop.py:822-824) AND patches/verl_agent_loop_sampling.py
        is applied to forward the key; otherwise every entry is None and the
        caller fails closed; decoded text is never treated as confidence.
        """
        pairs: list[tuple[int, float] | None] = [None] * n
        responses = output.batch.get("responses") if output is not None else None
        logps = output.batch.get("rollout_log_probs") if output is not None else None
        if responses is None or logps is None:
            return pairs
        for k in range(min(n, len(responses), len(logps))):
            try:
                pairs[k] = (int(responses[k][0]), float(logps[k][0]))
            except (IndexError, ValueError, TypeError):
                pairs[k] = None
        return pairs

    def supports_pipelined_mutation(self) -> bool:
        """True when stage-2 calls can be submitted per parent, not per batch.

        Needs the agent-loop workers: they are asyncio Ray actors that accept an
        arbitrary-size DataProto, so a single prompt can be submitted on its own
        with no divisor padding (same property ``RayAgentLoopTransport`` relies
        on for streamed rollouts).
        """
        trainer = self.trainer
        manager = getattr(trainer, "async_rollout_manager", None) if trainer else None
        return bool(getattr(manager, "agent_loop_workers", None))

    def mutate_pipelined(self, stage1_tasks, stage2_builder, *, max_in_flight=None):
        """Two-stage mutation with NO barrier between the stages.

        ``mutate()`` blocks on the whole batch, so the two-stage mutation paid
        its slowest sample twice per iteration: over 32 parents the chance that
        at least one reply runs away is 92% (stage 1) and 77% (stage 2), and
        only 27-32% of the measured wait was doing work. Here each prompt is a
        separate submission, and ``stage2_builder(index, reply)`` is called as
        soon as THAT parent's stage-1 reply lands -- its generator call is
        queued immediately, while other parents are still planning.

        ``stage2_builder`` returns a ``MutationTask`` to run stage 2 for that
        parent, or None to stop there (unparseable plan). Returns
        ``(stage1_replies, stage2_reply_by_index)``: the first is a list aligned
        with ``stage1_tasks``, the second a dict keyed by the same index and
        holding only the parents whose stage 2 actually ran. A failed or
        timed-out request yields None, which downstream reads as a mutation
        failure -- exactly what an unparseable reply already produced.
        """
        import ray

        n = len(stage1_tasks)
        stage1_replies: list[str | None] = [None] * n
        stage2_replies: dict[int, str | None] = {}
        if n == 0:
            return stage1_replies, stage2_replies

        trainer = self._require_trainer()
        workers = list(trainer.async_rollout_manager.agent_loop_workers)
        if not workers:
            raise RuntimeError("no agent_loop_workers to submit mutation prompts to")
        cfg = self._async_cfg
        if max_in_flight is None:
            max_in_flight = int(getattr(cfg, "max_in_flight_chunks", 32) or 32)
        max_in_flight = max(1, int(max_in_flight))
        timeout_s = float(getattr(cfg, "request_timeout_s", 900.0) or 900.0)

        opened_session = not self._session_active
        if opened_session:
            self._wake()
        # Same invariant the streamed rollout path carries: a weight sync while
        # requests are outstanding would split one mutation batch across two
        # policies. `_wake` asserts on this flag, so an overlap mode that tried
        # it would fail loudly instead of quietly mixing versions.
        self._streaming_in_flight = True
        try:
            rr = 0
            # (stage, index) -> submitted at; stage 2 is pushed to the FRONT so
            # a parent that has a plan finishes before new plans are started.
            todo: deque = deque((1, i) for i in range(n))
            pending: dict = {}
            deadlines: dict = {}

            def _submit(stage: int, index: int, task) -> None:
                nonlocal rr
                handle = workers[rr % len(workers)].generate_sequences.remote(
                    self._mutation_gen_batch(task)
                )
                rr += 1
                pending[handle] = (stage, index)
                deadlines[handle] = _time.monotonic() + timeout_s

            stage2_tasks: dict[int, MutationTask] = {}
            while todo or pending:
                while todo and len(pending) < max_in_flight:
                    stage, index = todo.popleft()
                    _submit(
                        stage,
                        index,
                        stage1_tasks[index] if stage == 1 else stage2_tasks[index],
                    )
                if not pending:
                    continue
                ready, _ = ray.wait(list(pending), num_returns=1, timeout=1.0)
                for handle in ready:
                    stage, index = pending.pop(handle)
                    deadlines.pop(handle, None)
                    try:
                        reply = self._decode_single_response(ray.get(handle))
                    except Exception:
                        reply = None
                    if stage == 2:
                        stage2_replies[index] = reply
                        continue
                    stage1_replies[index] = reply
                    task = stage2_builder(index, reply)
                    if task is not None:
                        stage2_tasks[index] = task
                        todo.appendleft((2, index))
                now = _time.monotonic()
                for handle, when in list(deadlines.items()):
                    if now <= when:
                        continue
                    stage, index = pending.pop(handle)
                    del deadlines[handle]
                    try:
                        ray.cancel(handle, force=False)
                    except Exception:
                        pass
                    if stage == 2:
                        stage2_replies[index] = None
                    else:
                        # No plan -> no generator call for this parent, which is
                        # what an unparseable stage-1 reply already means.
                        stage2_builder(index, None)
        finally:
            self._streaming_in_flight = False
            if opened_session:
                self._sleep()
        return stage1_replies, stage2_replies

    def _mutation_gen_batch(self, task: MutationTask):
        """One-row generation DataProto for a single mutation prompt."""
        messages = getattr(task, "messages", None)
        batch = self._make_prompt_batch(
            [task.prompt], messages=[messages] if messages else None
        )
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "raw_prompt",
                "data_source",
                "reward_model",
                "extra_info",
            ],
        )
        if task.max_output_tokens is not None:
            gen_batch.meta_info["max_tokens"] = max(1, int(task.max_output_tokens))
        if task.temperature is not None:
            gen_batch.meta_info["temperature"] = max(0.0, float(task.temperature))
        if task.top_p is not None:
            top_p_value = float(task.top_p)
            if not 0.0 < top_p_value <= 1.0:
                raise ValueError("top_p must be in (0, 1]")
            gen_batch.meta_info["top_p"] = top_p_value
        if task.logprobs is not None:
            gen_batch.meta_info["logprobs"] = int(task.logprobs)
        if task.allowed_token_ids is not None:
            gen_batch.meta_info["allowed_token_ids"] = list(task.allowed_token_ids)
        return gen_batch

    def sampled_token_logprob(self, output) -> tuple[int, float] | None:
        """(token id, logprob) of the FIRST generated token, or None.

        verl's agent loop only fills ``rollout_log_probs`` when the request
        asked for logprobs (agent_loop.py:822-824), which is what
        MutationTask.logprobs turns on -- and that key only reaches the sampler
        because patches/verl_agent_loop_sampling.py forwards it from meta_info.
        Without the patch this returns None so confidence-sensitive callers can
        fail closed instead of trusting decoded text.
        """
        if output is None:
            return None
        responses = output.batch.get("responses")
        logps = output.batch.get("rollout_log_probs")
        if responses is None or logps is None:
            return None
        if len(responses) == 0 or len(logps) == 0:
            return None
        return int(responses[0][0]), float(logps[0][0])

    def _decode_single_response(self, output) -> str | None:
        responses = output.batch.get("responses") if output is not None else None
        if responses is None or len(responses) == 0:
            return None
        return self._require_tokenizer().decode(
            responses[0].tolist(), skip_special_tokens=True
        )

    # ------------------------------------------------------------------
    # Phase sessions: keep vLLM awake across a batch of generate calls and
    # pay a single cumem wake_up, instead of one wake/sleep per instance
    # (the repeated wake_up is what tripped cumem_allocator's "invalid
    # argument" under hundreds of toggles per outer iteration).
    # ------------------------------------------------------------------

    def configure_streaming(
        self,
        async_cfg,
        *,
        version_tracker,
        sample_logger,
        rollout_metrics,
    ) -> None:
        """Enable the chunked streaming rollout path (call after ``bind``).

        With ``async_cfg.streaming_enabled`` false (or this never called) the
        legacy whole-batch path runs unchanged.
        """
        self._async_cfg = async_cfg
        self._version_tracker = version_tracker
        self._sample_logger = sample_logger
        self._rollout_metrics = rollout_metrics
        if async_cfg is not None and async_cfg.streaming_enabled:
            print(
                "[RQ-Evolve] rollout mode: streaming producer-consumer "
                f"(staleness={async_cfg.staleness_mode}, "
                f"max_policy_lag={async_cfg.max_policy_lag}, "
                f"chunk_size={async_cfg.chunk_size}, "
                f"max_in_flight={async_cfg.max_in_flight_chunks}, "
                f"timeout={async_cfg.request_timeout_s}s)"
            )
        else:
            print("[RQ-Evolve] rollout mode: legacy sequential (whole-batch, streaming off)")

    def _streaming_active(self) -> bool:
        if self._async_cfg is None or not self._async_cfg.streaming_enabled:
            return False
        trainer = self.trainer
        manager = getattr(trainer, "async_rollout_manager", None) if trainer else None
        return manager is not None and bool(getattr(manager, "agent_loop_workers", None))

    def _wake(self) -> None:
        """Push current FSDP weights into vLLM and wake it for generation.

        vLLM is launched with load_format=dummy and starts "sleeping"; without
        pushing the live actor weights it would forward with random weights ->
        NaNs -> "CUDA error: illegal memory access" inside flash-attn.
        """
        # Weight sync while chunks are in flight would mix policies WITHIN a
        # chunk (metadata stamps chunk-level versions). The sequential evolve
        # phase can't hit this; the assert keeps future overlap modes honest.
        assert not self._streaming_in_flight, (
            "weight sync requested while streamed rollout chunks are in flight"
        )
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
        if self._streaming_active():
            return self._generate_rollouts_streaming(list(instances), n)
        prompts = [inst.problem for inst in instances]
        messages = [build_solver_messages(inst.problem) for inst in instances]
        output, full_batch = self._generate_with_batch(
            prompts,
            n_repeat=n,
            messages=messages,
            ground_truths=[inst.answer for inst in instances],
            verifier_specs=[inst.verifier for inst in instances],
        )
        responses = output.batch.get("responses")
        if responses is None:
            return PendingRollouts(instances=list(instances), n_rollouts=n)
        payloads = self._slice_per_instance(output, len(instances), n)
        decoded = [
            self.tokenizer.decode(row.tolist(), skip_special_tokens=True)
            for row in responses
        ]
        return PendingRollouts(
            instances=list(instances),
            n_rollouts=n,
            full_batch=full_batch,
            decoded=decoded,
            payloads=payloads,
        )

    def _generate_rollouts_streaming(
        self, instances: list[ProblemInstance], n: int
    ) -> PendingRollouts:
        """Chunked producer-consumer rollout over the agent-loop workers.

        Chunks of ``chunk_size`` instances (x n rollouts each) are submitted
        round-robin directly to the AgentLoopWorker actors; verification,
        filtering, and per-sample JSONL run the moment each chunk completes.
        Entropy stays deferred to ``finalize_rollouts`` (same GPU memory
        profile as the legacy path).
        """
        from .async_rollout import (
            ChunkedRolloutScheduler,
            ChunkJob,
            RayAgentLoopTransport,
        )
        from .rollout_log import JsonlSampleLogger

        trainer = self._require_trainer()
        manager = trainer.async_rollout_manager
        cfg = self._async_cfg
        tracker = self._version_tracker
        stamp = tracker.stamp() if tracker is not None else {}
        current_version_fn = (
            (lambda: tracker.policy_version) if tracker is not None else (lambda: -1)
        )
        sample_logger = self._sample_logger
        if sample_logger is None or not cfg.log_samples:
            sample_logger = JsonlSampleLogger(None, enabled=False)
        metrics = self._rollout_metrics
        if metrics is None:
            from .metrics import RolloutMetrics

            metrics = RolloutMetrics()

        jobs: list[ChunkJob] = []
        chunk_size = max(1, int(cfg.chunk_size))
        for chunk_id, start in enumerate(range(0, len(instances), chunk_size)):
            chunk_instances = instances[start : start + chunk_size]
            prompts = [inst.problem for inst in chunk_instances]
            batch = self._make_prompt_batch(
                prompts,
                messages=[
                    build_solver_messages(inst.problem)
                    for inst in chunk_instances
                ],
                ground_truths=[inst.answer for inst in chunk_instances],
                verifier_specs=[inst.verifier for inst in chunk_instances],
            )
            gen_batch = batch.pop(
                batch_keys=["input_ids", "attention_mask", "position_ids"],
                non_tensor_batch_keys=[
                    "raw_prompt_ids",
                    "raw_prompt",
                    "data_source",
                    "reward_model",
                    "extra_info",
                ],
            )
            batch.non_tensor_batch["uid"] = np.array(
                [f"{chunk_id}-{i}" for i in range(len(batch.batch))], dtype=object
            )
            if n > 1:
                gen_batch = gen_batch.repeat(repeat_times=n, interleave=True)
            jobs.append(
                ChunkJob(
                    chunk_id=chunk_id,
                    instances=chunk_instances,
                    gen_batch=gen_batch,
                    batch=batch,
                    n_rollouts=n,
                    meta=dict(stamp),
                )
            )

        scheduler = ChunkedRolloutScheduler(
            transport=RayAgentLoopTransport(manager.agent_loop_workers),
            cfg=cfg,
            tokenizer=self._require_tokenizer(),
            metrics=metrics,
            sample_logger=sample_logger,
            current_version_fn=current_version_fn,
            iteration=self.current_iteration,
        )
        self._streaming_in_flight = True
        try:
            results = scheduler.run(jobs)
        finally:
            self._streaming_in_flight = False
        return PendingRollouts(
            instances=instances, n_rollouts=n, chunk_results=results
        )

    def _finalize_streaming(self, pending: PendingRollouts) -> list[list[RolloutRecord]]:
        """Deferred entropy over the streamed chunks, then regroup in order.

        Failed chunks contributed no output batch; their (rejected) records
        keep entropy 0.0. Row order within the concat matches the per-chunk
        record order (chunk_id ascending, instances x n interleaved), so the
        entropy vector back-distributes by simple offset.
        """
        results = pending.chunk_results or []
        grouped_all: list[list[RolloutRecord]] = []
        entropy_batches = [r.full_batch for r in results if r.full_batch is not None]
        entropies: list[float] = []
        if entropy_batches:
            from verl.protocol import DataProto as _DataProto

            merged = _DataProto.concat(entropy_batches)
            metrics = self._rollout_metrics
            if metrics is not None:
                with metrics.timed("entropy"):
                    entropies = self._response_entropies(merged)
            else:
                entropies = self._response_entropies(merged)
        offset = 0
        payloads: list = []
        for result in results:
            has_rows = result.full_batch is not None
            # Same slicing as the batched path: rows are instance-major because
            # the chunk repeated each prompt n times with interleave=True.
            chunk_payloads = self._slice_per_instance(
                result.output, len(result.grouped), result.job.n_rollouts
            ) if result.output is not None else None
            for index, rows in enumerate(result.grouped):
                for record in rows:
                    if has_rows and offset < len(entropies):
                        record.entropy = entropies[offset]
                    if has_rows:
                        offset += 1
                grouped_all.append(rows)
                payloads.append(
                    chunk_payloads[index]
                    if chunk_payloads is not None and index < len(chunk_payloads)
                    else None
                )
        pending.payloads = payloads
        return grouped_all

    def finalize_rollouts(self, pending: PendingRollouts) -> list[list[RolloutRecord]]:
        """Compute entropy (actor forward) and assemble grouped records.

        Call AFTER ``end_session`` so the actor forward runs with vLLM asleep.
        """
        instances = pending.instances
        if not instances:
            return []
        if pending.chunk_results is not None:
            return self._finalize_streaming(pending)
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
                        correct=bool(
                            pred and answers_match(pred, inst.answer, inst.verifier)
                        ),
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
        self,
        prompts: list[str],
        n_repeat: int = 1,
        messages: list | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        ground_truths: list[str] | None = None,
        verifier_specs: list[dict] | None = None,
        logprobs: int | None = None,
        allowed_token_ids: list[int] | None = None,
    ):
        trainer = self._require_trainer()
        batch = self._make_prompt_batch(
            prompts,
            messages=messages,
            ground_truths=ground_truths,
            verifier_specs=verifier_specs,
        )
        gen_batch = batch.pop(
            batch_keys=["input_ids", "attention_mask", "position_ids"],
            non_tensor_batch_keys=[
                "raw_prompt_ids",
                "raw_prompt",
                "data_source",
                "reward_model",
                "extra_info",
            ],
        )
        batch.non_tensor_batch["uid"] = np.array(
            [str(i) for i in range(len(batch.batch))],
            dtype=object,
        )
        if n_repeat > 1:
            gen_batch = gen_batch.repeat(repeat_times=n_repeat, interleave=True)
        if max_tokens is not None:
            # verl's rollout workers accept per-call vLLM sampling overrides
            # through DataProto.meta_info. Response tensors remain padded to the
            # configured response length, so downstream shapes are unchanged.
            gen_batch.meta_info["max_tokens"] = max(1, int(max_tokens))
        if temperature is not None:
            gen_batch.meta_info["temperature"] = max(0.0, float(temperature))
        if top_p is not None:
            top_p_value = float(top_p)
            if not 0.0 < top_p_value <= 1.0:
                raise ValueError("top_p must be in (0, 1]")
            gen_batch.meta_info["top_p"] = top_p_value
        if logprobs is not None:
            gen_batch.meta_info["logprobs"] = int(logprobs)
        if allowed_token_ids is not None:
            gen_batch.meta_info["allowed_token_ids"] = list(allowed_token_ids)

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
            # LENGTH-NORMALIZED, h = (1/|y|) sum_l H(x, y_<l). The summed
            # variant this replaces made cross-problem variation track response
            # length instead of uncertainty, and correlated strongly with the
            # systematic length asymmetry between successful and failed
            # rollouts (failures run longer) -- measured across three
            # checkpoints. Normalizing also bounds h by log|V|, which is what
            # makes R_Q's concentration bound a constant shared by every
            # program rather than one scaled by each program's response length.
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

        Keeps both the head (instructions/schema) and tail (the live parent,
        evidence/plan, and generation cue), dropping the middle few-shot material
        first. Verifies against the real chat-template length and trims further
        if token-merge effects at the boundaries still overflow.
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

        def render(keep: int) -> str:
            if keep >= len(content_ids):
                return prompt
            head = max(1, int(keep * 0.55))
            tail = max(0, keep - head)
            head_text = tok.decode(
                content_ids[:head],
                skip_special_tokens=True,
            )
            tail_text = (
                tok.decode(content_ids[-tail:], skip_special_tokens=True)
                if tail
                else ""
            )
            return (
                head_text
                + "\n\n...[middle truncated; live request preserved]...\n\n"
                + tail_text
            )

        truncated = render(budget)
        # re-check: decode/re-encode round trips can shift the count slightly
        while budget > 0 and self._chat_template_len(truncated) > max_prompt_length:
            budget = max(0, budget - 64)
            truncated = render(budget)
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
        # Short conversations (e.g. the evaluator's [system, user] pair) have no
        # middle turn, so there is nothing in the range above -- the long content
        # lives in the final user turn itself. Fall back to clipping that turn so
        # the prompt is still capped; otherwise it sails past max_prompt_length
        # and crashes verl's _postprocess torch.cat (mismatched prompt lengths).
        if not clippable and len(msgs) >= 2:
            clippable = [len(msgs) - 1]
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
            if keep and len(msgs) == 2 and i == len(msgs) - 1:
                head = max(1, int(keep * 0.55))
                tail = max(0, keep - head)
                clipped = tok.decode(ids[:head], skip_special_tokens=True)
                if tail:
                    clipped += (
                        "\n...[middle truncated]...\n"
                        + tok.decode(ids[-tail:], skip_special_tokens=True)
                    )
            else:
                clipped = (
                    tok.decode(ids[:keep], skip_special_tokens=True)
                    if keep
                    else ""
                )
            msgs[i]["content"] = (clipped + "\n...[truncated]...") if keep else "...[truncated]..."
        # If system + final live request alone still overflow, retain both ends
        # of the final turn. The head holds task/parent context; the tail holds
        # evidence/plan conclusions and the generation cue.
        for _ in range(256):
            over = total(msgs) - max_prompt_length
            if over <= 0 or len(msgs) < 2:
                break
            i = len(msgs) - 1
            ids = tok.encode(msgs[i]["content"], add_special_tokens=False)
            keep = max(0, len(ids) - max(64, over + 16))
            if not keep:
                msgs[i]["content"] = "...[truncated]..."
                continue
            head = max(1, int(keep * 0.55))
            tail = max(0, keep - head)
            clipped = tok.decode(ids[:head], skip_special_tokens=True)
            if tail:
                clipped += (
                    "\n...[middle truncated; live request preserved]...\n"
                    + tok.decode(ids[-tail:], skip_special_tokens=True)
                )
            msgs[i]["content"] = clipped
        return msgs

    @staticmethod
    def _slice_per_instance(output, n_instances: int, n_rollouts: int) -> list | None:
        """Split one generation output into per-instance DataProto slices.

        ``gen_batch.repeat(..., interleave=True)`` lays the rows out as
        instance-major, so instance i owns rows [i*m, (i+1)*m). Returning the
        REAL slices, rather than rebuilding tensors from decoded text, is what
        makes replay safe: whatever contract the trainer expects of a generation
        output is satisfied because this IS one.
        """
        try:
            total = len(output.batch)
        except Exception:
            return None
        if n_instances <= 0 or total != n_instances * n_rollouts:
            return None
        try:
            return [
                output.slice(i * n_rollouts, (i + 1) * n_rollouts)
                for i in range(n_instances)
            ]
        except Exception:
            return None

    def _make_prompt_batch(
        self,
        prompts: list[str],
        messages: list | None = None,
        ground_truths: list[str] | None = None,
        verifier_specs: list[dict] | None = None,
    ):
        tokenizer = self._require_tokenizer()
        max_prompt_length = self.max_prompt_length or 1024
        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

        if messages is None:
            messages = [None] * len(prompts)

        # Per item, build (a) the rendered prompt text used for input_ids and
        # (b) the chat-message list handed to the agent loop as raw_prompt.
        # Items that arrive as a conversation -- solver rollouts ([system,user])
        # and fix-retries ([system,user,assistant,user]) -- are chat-templated
        # here; the remaining bare strings are mutation prompts, wrapped as one
        # user turn.
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
        # The agent loop scores every generation AS IT PRODUCES IT and stores
        # the result in the output's `rm_scores`; verl's `extract_reward` then
        # reads that tensor directly rather than recomputing. So the ground
        # truth handed in here is the one the training reward is computed from.
        #
        # It used to be an empty placeholder, on the reasoning that this backend
        # grades its own rollouts and discards verl's number. Replay made that
        # false: the solver update now trains on THESE rollouts, and an empty
        # ground truth scores every one of them 0 -- zero reward, zero
        # advantage, zero gradient, silently, for a whole run. Mutation prompts
        # have no ground truth and still pass "".
        data_source_arr = np.array(["rq_evolve"] * len(prompts), dtype=object)
        reward_model_arr = np.empty(len(prompts), dtype=object)
        extra_info_arr = np.empty(len(prompts), dtype=object)
        for i in range(len(prompts)):
            truth = ""
            if ground_truths is not None and i < len(ground_truths):
                truth = str(ground_truths[i] or "")
            verifier = normalize_verifier(
                verifier_specs[i]
                if verifier_specs is not None and i < len(verifier_specs)
                else None,
                answer=truth if truth else None,
            )
            # Keep this first assignment explicit: replay's source-level guard
            # asserts that the live answer, never an empty placeholder, reaches
            # the agent-loop reward path.
            reward_model_arr[i] = {"ground_truth": truth}
            reward_model_arr[i]["verifier"] = verifier
            # Recent verl reward managers pass this object as compute_score's
            # ``extra_info`` argument.  Keeping the same contract in
            # reward_model as well makes the generated batch self-describing
            # across version-specific manager plumbing.
            extra_info_arr[i] = {"verifier": verifier}
        data = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "raw_prompt_ids": raw_prompt_ids_arr,
            "raw_prompt": raw_prompt_arr,
            "data_source": data_source_arr,
            "reward_model": reward_model_arr,
            "extra_info": extra_info_arr,
        }
        return DataProto.from_single_dict(data)

    def _cfg_get(self, dotted_key: str, default=None):
        """Read a dotted key from the trainer's verl config (OmegaConf or dict)."""
        trainer = self._require_trainer()
        node = getattr(trainer, "config", None)
        if node is None:
            return default
        try:
            value = OmegaConf.select(node, dotted_key)
            return default if value is None else value
        except Exception:
            for key in dotted_key.split("."):
                node = getattr(node, key, None) if not isinstance(node, dict) else node.get(key)
                if node is None:
                    return default
            return node

    def _require_trainer(self):
        if self.trainer is None:
            raise RuntimeError("VerlPolicyBackend is not bound to a trainer")
        return self.trainer

    def _require_tokenizer(self):
        if self.tokenizer is None:
            raise RuntimeError("VerlPolicyBackend is not bound to a tokenizer")
        return self.tokenizer

    def _compute_actor_log_probs(self, padded_batch):
        trainer = self._require_trainer()
        # The unified engine used by verl 0.9 accepts an unpadded TensorDict,
        # while R_Q keeps its rollout cache in the padded DataProto format.
        # Let the trainer perform the same DataProto -> TensorDict -> DataProto
        # conversion it uses for its own PPO old-log-prob/entropy pass.  The
        # legacy 0.7 trainer exposes this helper too, returning the same pair.
        compute_old = getattr(trainer, "_compute_old_log_prob", None)
        if compute_old is not None:
            result = compute_old(padded_batch)
            return result[0] if isinstance(result, tuple) else result

        worker_group = trainer.actor_rollout_wg
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
    return compute_position_id_with_mask(attention_mask)
