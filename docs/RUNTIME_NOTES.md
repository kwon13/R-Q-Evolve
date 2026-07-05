# Runtime / Machine Notes

`configs/rq_evolve_base.yaml`의 여러 knob 값은 **특정 장비(97GB GPU × 8, 커스텀
vLLM 0.19.1 빌드)에서 겪은 OOM / 크래시**를 근거로 튜닝돼 있습니다. 이 근거를 config에서
분리해 여기에 모았습니다. config에는 각 knob 옆에 이 문서의 앵커를 가리키는 한 줄
포인터만 남겨 두었습니다.

> 값 자체는 config가 single source of truth입니다. 이 문서는 **왜 그 값인지**만 설명합니다.
> 다른 장비로 옮길 때 먼저 읽어야 하는 "war-story" 모음입니다.

## <a id="remove-padding"></a>`actor_rollout_ref.model.use_remove_padding: true`

remove_padding 없이는 모델이 padding 토큰까지 forward합니다. `max_seq_len=12000`,
vocab=151k에서 bf16 logits 텐서는 micro-batch 6당 **~22GB**로, vLLM + actor + ref가
한 GPU에 공존하면 97GB GPU에서 OOM 납니다.

## <a id="ref-logprob"></a>`actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu`

ref의 log-prob/entropy forward는 micro-batch당 full-vocab(151k) logits 텐서를
만듭니다(**4에서 ~13.7GB**). vLLM이 resident(sleep 비활성)인 상태에서 이 forward가
evolve phase에 실행되는데, 이 phase는 `update_actor` 직후 epoch 경계에서 돌아
optimizer states가 아직 GPU에 있습니다. → micro-batch를 작게 유지해 transient를
~4배 줄임. forward는 step 시간의 작은 부분이라 throughput 손해는 무시할 만함.

## <a id="weight-bucket"></a>`actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes: 4096`

vLLM로의 bucketed weight transfer는 텐서를 고정 크기 버퍼로 묶고, **가장 큰 단일
텐서가 버킷 하나에 들어가야** 합니다. Qwen3-8B의 `embed_tokens.weight`는
151936 × 4096 × 4B(fp32 master weight) ≈ **2374MB**로 verl 기본값 2048MB를 넘칩니다.
→ 여유를 두고 4096MB로 올림.

- 주의(runtime provenance): 실제로는 `rollout.checkpoint_engine.*`에서 읽습니다
  (`vllm_rollout.py` line 166: `self.config.checkpoint_engine.update_weights_bucket_megabytes`).
  assertion 메시지의 `rollout.update_weights_bucket_megabytes`는 `checkpoint_engine`
  레벨을 생략하고 있어 혼동을 줍니다.

## <a id="gpu-memory"></a>`actor_rollout_ref.rollout.gpu_memory_utilization: 0.45`

vLLM이 RESIDENT(sleep mode 비활성, 아래 참조)라 FSDP actor의 training step
(peak ~33GB reserved)과 on-GPU로 공존해야 합니다. 97GB 카드에서 actor의 evolve-phase
log-prob/entropy forward와 resident optimizer states까지 같이 있어야 하므로 vLLM을
modest하게 유지: **0.45×97 ≈ 43GB vLLM + ~42GB actor(optimizer 포함) ≈ 85GB**. entropy
transient는 위 [ref-logprob](#ref-logprob) / [rollout-logprob](#rollout-logprob)로 축소.

## <a id="sleep-wake"></a>vLLM sleep/wake DISABLED (`free_cache_engine: false`, `enable_sleep_mode: false`)

이 커스텀 vLLM 0.19.1 빌드의 cumem allocator는 **~3 sleep/wake 사이클 뒤 wake_up에서
GPU 가상 메모리 re-map에 실패**합니다:

```
CUDA Error: invalid argument at cumem_allocator.cpp:145
```

verl 자체의 per-step wake(step당 1회 wake)에서 R-Q-Evolve와 무관하게 재현되므로 어떤
run이든 ~step 3에서 죽습니다. 그래서:

- `free_cache_engine=false` → verl이 `rollout.resume()/sleep()`(wake_up 호출 지점,
  `fsdp_workers.rollout_mode`)을 건너뜀.
- `enable_sleep_mode=false` → vLLM을 cumem allocator 없이 빌드(그러면 verl이
  `expandable_segments`를 자동 활성화).

결과: vLLM은 resident로 남고, weight는 매 step FSDP에서 `checkpoint_manager.update_weights`로
동기화(naive backend = wake/sleep 없는 plain weight push). 비용은 위 [gpu-memory](#gpu-memory)의
GPU 메모리.

## <a id="rollout-logprob"></a>`actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu`

actor의 `compute_log_prob`(우리 evolve의 uncertainty/entropy 경로)는 **rollout** knob을
읽습니다(`fsdp_workers`: `config_source = self.config.rollout`). full-vocab entropy logits
transient를 줄여 evolve-phase OOM을 피하려고 작게 유지. 근거는 위 [ref-logprob](#ref-logprob)와 동일.
