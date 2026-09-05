#!/usr/bin/env python3
"""Freeze Omni-MATH-2 inputs and run resumable, checkpoint-blind vLLM inference.

No scores are computed here. See judge_omni_math2.py and protocol.json.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "analysis/omni_math2_checkpoint_comparison"
SOURCE = ROOT / "data/benchmarks/omni_math_2/Omni-Math-2.jsonl"
ANNOTATIONS = ROOT / "analysis/omni_problem_types/omni_math_2_current/full/annotations.jsonl"
SOURCE_SHA256 = "52c897861b408e9e2b393e40cf4c824a880f59bea0d7b6d75b550eb5938b5346"
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
DOMAINS = ["Algebra", "Geometry", "Number Theory", "Discrete Mathematics",
           "Applied Mathematics", "Calculus", "Precalculus"]
TYPES = ["decision", "search", "counting", "optimization", "function"]
MODEL_ROOT = ROOT / "rq_output/rq_evolve_8b_domain_type_35cell_8gpu"
RZERO_ROOT = Path("/home/yhoon113/rzero_ckpt/ablation/qwen3-8b-base_rzero/checkpoints")
MODELS = {
    "rzero_256": str(RZERO_ROOT / "global_step_256/hf_merged"),
    "rq_256": str(MODEL_ROOT / "global_step_256/hf_merged"),
    "rzero_96": str(RZERO_ROOT / "global_step_96/hf_merged"),
    "rq_224": str(MODEL_ROOT / "global_step_224/hf_merged"),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def freeze(path, content):
    path = Path(path)
    if path.exists():
        if path.read_text() != content:
            raise ValueError(f"Refusing incompatible frozen file: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        f.write(content)


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def recover_prediction_tail(path):
    """Preserve an interrupted last record before safely resuming the journal."""
    if not path.exists():
        return
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return
    boundary = data.rfind(b"\n") + 1
    try:
        json.loads(data[boundary:])
    except (json.JSONDecodeError, UnicodeDecodeError):
        backup = path.with_name(path.name + f".incomplete-tail.{time.time_ns()}")
        with backup.open("xb") as f:
            f.write(data[boundary:])
        with path.open("r+b") as f:
            f.truncate(boundary)
        print(f"Preserved incomplete prediction tail at {backup}", flush=True)
    else:
        with path.open("ab") as f:
            f.write(b"\n")


def prepare(out):
    if file_hash(SOURCE) != SOURCE_SHA256:
        raise ValueError("Pinned source SHA256 mismatch")
    source, annotations = read_jsonl(SOURCE), read_jsonl(ANNOTATIONS)
    if len(source) != 4428 or len(annotations) != len(source):
        raise ValueError("Unexpected source/annotation cardinality")
    seen = Counter(r["problem"] for r in source if r["tags"] == [])
    manifest = []
    subset_index = 0
    for i, (row, ann) in enumerate(zip(source, annotations)):
        assert ann["index"] == i and ann["problem"] == row["problem"]
        assert ann["answer"] == row["answer"]
        eligible = row["tags"] == []
        flags = []
        if row["tags"]:
            flags.append("author_tagged:" + ",".join(row["tags"]))
        if not str(row["answer"]).strip():
            flags.append("blank_reference_answer")
        # Manually verified before model scores, not inferred from model failure.
        if row["id"] in (2846, 2913):
            flags.append("required_diagram_missing")
        if eligible and seen[row["problem"]] > 1:
            flags.append("exact_statement_duplicate")
        if ann["problem_type"] is None:
            flags.append("problem_type_abstained")
        if not ann["top_level_domains"]:
            flags.append("domain_unmapped")
        gradable = eligible and not any(x in flags for x in
                        ("blank_reference_answer", "required_diagram_missing"))
        manifest.append({
            "id": i, "source_id": row["id"],
            "subset_index": subset_index if eligible else None,
            "problem": row["problem"], "answer": row["answer"],
            "problem_sha256": digest(row["problem"]),
            "domains": ann["top_level_domains"],
            "problem_type": ann["problem_type"],
            "classification_confidence": ann["confidence"],
            "classification_review_reason": ann.get("review_reason"),
            "difficulty": row.get("difficulty"), "source": row.get("source"),
            "tags": row["tags"], "eligible": eligible, "gradable": gradable,
            "qa_flags": flags,
        })
        subset_index += int(eligible)
    freeze(out / "manifest.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in manifest))
    protocol = {
        "dataset": "martheballon/Omni-MATH-2",
        "revision": "693b21ad38ec8c8e0354d3a2f286ca83473821b0",
        "source_path": str(SOURCE), "source_sha256": SOURCE_SHA256,
        "annotations_path": str(ANNOTATIONS), "annotations_sha256": file_hash(ANNOTATIONS),
        "manifest_sha256": file_hash(out / "manifest.jsonl"),
        "source_rows": len(manifest), "inference_rows": subset_index,
        "scorable_rows": sum(r["gradable"] for r in manifest),
        "eligibility": "Author tags == []; infer every eligible row, including unclassified rows.",
        "qa_exclusions_source_ids": {"blank_reference_answer": [4168, 4416],
                                     "required_diagram_missing": [2846, 2913]},
        "qa_note": "Known defects only, not a comprehensive manual answer audit. Reference answers remain fallible.",
        "domains": DOMAINS, "types": TYPES,
        "domain_policy": "Published multi-label top-level domains; each problem contributes to all its domains.",
        "problem_type_policy": "Frozen statement-only computational-output-contract-v1; abstentions not force-labelled.",
        "classification_sha256": file_hash(ROOT / "src/rq_evolve/problem_type.py"),
        "comparisons": {"primary": ["rzero_256", "rq_256"], "secondary": ["rzero_96", "rq_224"]},
        "selection": "Primary fixed step256; secondary maxima of existing seven-benchmark macro before any Omni inference.",
        "selection_caveat": "Equal nominal training steps do not establish equal FLOPs or generated data volume; secondary is not Omni-selected best.",
        "model_paths": MODELS,
        "system_prompt": SYSTEM_PROMPT,
        "prompt_policy": "Match scripts/eval_vllm_math.py _build_prompt; same tokenizer/template behavior checked for all checkpoints.",
        "sampling": {"temperature": 0.0, "top_p": 1.0, "n": 1, "seed": 0,
                     "max_tokens": 4096, "max_model_len": 8192, "dtype": "bfloat16",
                     "tensor_parallel_size": 1, "enforce_eager": True,
                     "sampler": "pytorch", "max_num_seqs": 64, "chunk_size": 256},
        "judge": {"model": "gpt-5-mini-2025-08-07", "scope": "all gradable responses",
                  "protocol": "Documented adaptation of Omni-MATH-2/HLE equivalence judge, not exact code reproduction.",
                  "failures": "null and retried, never scored wrong"},
        "uncertainty": "Paired exact-statement cluster bootstrap; all 35 displayed; Wilson absolute CIs; no minimum-n filtering.",
        "interpretation": "Observed extrema on the labeled subset, not established population worst/best. Small-n and classifier/reference uncertainty remain.",
        "sources": ["https://arxiv.org/html/2601.19532v1",
                    "https://huggingface.co/datasets/martheballon/Omni-MATH-2",
                    "https://developers.openai.com/api/docs/models/gpt-5-mini"],
    }
    freeze(out / "protocol.json", json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    counts = {f"{d}::{t}": sum(r["gradable"] and d in r["domains"] and r["problem_type"] == t
                              for r in manifest) for d in DOMAINS for t in TYPES}
    print(json.dumps({"source": len(manifest), "infer": subset_index,
                      "gradable": protocol["scorable_rows"], "cell_counts": counts}), flush=True)


def audit_selection(out):
    """Snapshot existing-benchmark selection without reading any Omni scores."""
    roots = {"rq": ROOT / "rq_output/prv_rq_evolve_8b_domain_type_35cell_8gpu",
             "rzero": Path("/data1/yhoon113/R-Zero/results/ablation/qwen3-8b-base_rzero/checkpoints")}
    benches = ["aime24", "aime25", "amc23", "gsm8k", "math500", "minerva_math", "olympiadbench"]
    audit = {"criterion": "Unweighted mean of seven existing benchmark pass_at_1 values",
             "uses_omni_scores": False, "benchmarks": benches, "methods": {}}
    for method, root in roots.items():
        candidates = []
        for step in range(32, 257, 32):
            inputs = {}
            for bench in benches:
                path = root / f"global_step_{step}" / "eval" / bench / "summary.json"
                summary = json.loads(path.read_text())
                entry = summary["benchmarks"][bench]
                if entry.get("gpt_recheck_degraded") or not summary.get("gpt_recheck"):
                    raise ValueError(f"Unusable selection input: {path}")
                inputs[bench] = {"path": str(path), "sha256": file_hash(path),
                                 "pass_at_1": entry["pass_at_1"], "model": summary["model"],
                                 "created_at": summary.get("created_at")}
            candidates.append({"step": step, "macro": sum(r["pass_at_1"] for r in inputs.values()) / 7,
                               "inputs": inputs})
        selected = max(candidates, key=lambda r: r["macro"])["step"]
        if selected != {"rq": 224, "rzero": 96}[method]:
            raise ValueError("Existing benchmark best differs from preregistered selection; review required")
        audit["methods"][method] = {"selected_step": selected, "candidates": candidates}
    freeze(out / "checkpoint_selection.json", json.dumps(audit, ensure_ascii=False, indent=2) + "\n")
    print({method: result["selected_step"] for method, result in audit["methods"].items()}, flush=True)


def build_prompt(tokenizer, problem):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": problem}],
            add_generation_prompt=True, tokenize=False, add_special_tokens=True)
    return f"<|im_start|>user\n{SYSTEM_PROMPT}\nQuestion: {problem}<|im_end|>\n<|im_start|>assistant\n"


def infer(args):
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    out = args.root
    protocol = json.loads((out / "protocol.json").read_text())
    if file_hash(out / "manifest.jsonl") != protocol["manifest_sha256"]:
        raise ValueError("Manifest differs from frozen protocol")
    manifest = read_jsonl(out / "manifest.jsonl")
    rows = [r for r in manifest if r["eligible"]]
    if args.limit:
        rows = rows[:args.limit]
    model = Path(MODELS[args.run])
    run_dir = out / "runs" / args.run
    run_dir.mkdir(parents=True, exist_ok=True)
    # RQ exports use transformers 5.x's extra_special_tokens list and nested
    # rope_parameters. Keep signed original assets untouched. The common RZ
    # tokenizer has identical vocabulary/template and token IDs on every source
    # problem; transformers 4.57 can load it without schema adaptation.
    tokenizer_path = Path(MODELS["rzero_256"]) if args.run.startswith("rq_") else model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True, trust_remote_code=False)
    hf_overrides = {}
    raw_config = json.loads((model / "config.json").read_text())
    rope = raw_config.get("rope_parameters")
    if rope is not None:
        if rope.get("rope_type") != "default" or "rope_theta" not in rope:
            raise ValueError("Unrecognized rope_parameters; explicit compatibility review needed")
        hf_overrides["rope_theta"] = float(rope["rope_theta"])
    index_path = model / "model.safetensors.index.json"
    if index_path.exists():
        model_index = json.loads(index_path.read_text())
        shard_paths = [model / name for name in sorted(set(model_index["weight_map"].values()))]
    else:
        shard_paths = [model / "model.safetensors"]
    for path in shard_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    # Cache-friendly audit of complete weights before first load (~seconds on NVMe).
    model_files = {p.name: {"bytes": p.stat().st_size, "sha256": file_hash(p)}
                   for p in shard_paths + [model / "config.json", model / "tokenizer.json",
                                           model / "tokenizer_config.json"]}
    prompts = {r["id"]: build_prompt(tokenizer, r["problem"]) for r in rows}
    settings = json.loads((out / "protocol.json").read_text())["sampling"]
    config = {
        "run": args.run, "model": str(model), "resolved_model": str(model.resolve()),
        "model_files": model_files, "manifest_sha256": file_hash(out / "manifest.jsonl"),
        "sampling": settings, "eos_token_id": tokenizer.eos_token_id,
        "chat_template_sha256": digest(tokenizer.chat_template or "base-fallback"),
        "system_prompt": SYSTEM_PROMPT, "prompt_probe_sha256": digest(build_prompt(tokenizer, "2+2=?")),
        "packages": {p: importlib.metadata.version(p) for p in ("vllm", "torch", "transformers")},
    }
    if args.run.startswith("rq_"):
        config["compatibility"] = {
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_file_sha256": file_hash(tokenizer_path / "tokenizer.json"),
            "hf_overrides": hf_overrides,
            "reason": "Read transformers5 export with4.57; preserve1e6 RoPE theta and equivalent tokenizer.",
        }
    freeze(run_dir / "inference_config.json", json.dumps(config, indent=2) + "\n")
    config_hash = file_hash(run_dir / "inference_config.json")
    prior_path = run_dir / "predictions.jsonl"
    recover_prediction_tail(prior_path)
    prior = read_jsonl(prior_path) if prior_path.exists() else []
    all_by_id = {r["id"]: r for r in manifest}
    seen = {}
    for r in prior:
        if r["id"] in seen:
            raise ValueError("Duplicate prediction id")
        if r["inference_config_sha256"] != config_hash or digest(r["response"]) != r["response_sha256"]:
            raise ValueError("Invalid/incompatible previous prediction")
        if r["problem_sha256"] != all_by_id[r["id"]]["problem_sha256"]:
            raise ValueError("Previous problem mismatch")
        seen[r["id"]] = r
    todo = [r for r in rows if r["id"] not in seen]
    print(f"{now()} {args.run}: {len(seen)} saved; {len(todo)} remaining", flush=True)
    if not todo:
        return
    llm = LLM(model=str(model), tokenizer=str(tokenizer_path), trust_remote_code=False,
              tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=0.75,
              max_model_len=8192, enforce_eager=True, max_num_seqs=64, seed=0,
              hf_overrides=hf_overrides)
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=4096, n=1,
                              seed=0, stop_token_ids=[tokenizer.eos_token_id])
    engine_config = llm.llm_engine.model_config
    expected_theta = hf_overrides.get("rope_theta", raw_config.get("rope_theta"))
    actual_theta = getattr(engine_config.hf_config, "rope_theta", None)
    if actual_theta != expected_theta or engine_config.max_model_len != 8192:
        raise ValueError(f"Loaded engine configuration mismatch: rope_theta={actual_theta}")
    atomic_json(run_dir / "effective_runtime.json", {
        "rope_theta": actual_theta, "max_model_len": engine_config.max_model_len,
        "dtype": str(engine_config.dtype), "max_tokens": sampling.max_tokens,
        "temperature": sampling.temperature, "top_p": sampling.top_p, "n": sampling.n,
        "tokenizer": str(tokenizer_path), "hf_overrides": hf_overrides,
        "checked_at": now(), "inference_config_sha256": config_hash,
    })
    start = time.monotonic()
    with prior_path.open("a", buffering=1) as f:
        for offset in range(0, len(todo), 256):
            chunk = todo[offset:offset + 256]
            for row in chunk:
                n_input = len(tokenizer.encode(prompts[row["id"]]))
                if n_input + 4096 > 8192:
                    raise ValueError(f"Prompt too long; refusing silent truncation id={row['id']}: {n_input}")
            outputs = llm.generate([prompts[r["id"]] for r in chunk], sampling, use_tqdm=True)
            if len(outputs) != len(chunk):
                raise ValueError("Inference cardinality mismatch")
            for row, output in zip(chunk, outputs):
                generated = output.outputs[0]
                result = {"id": row["id"], "source_id": row["source_id"],
                          "problem_sha256": row["problem_sha256"], "response": generated.text,
                          "response_sha256": digest(generated.text),
                          "prompt_sha256": digest(prompts[row["id"]]),
                          "prompt_tokens": len(output.prompt_token_ids),
                          "completion_tokens": len(generated.token_ids),
                          "finish_reason": generated.finish_reason, "stop_reason": generated.stop_reason,
                          "inference_config_sha256": config_hash, "created_at": now()}
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                seen[row["id"]] = result
            f.flush()
            os.fsync(f.fileno())
            atomic_json(run_dir / "inference_progress.json", {
                "run": args.run, "completed": len(seen), "expected": sum(r["eligible"] for r in manifest),
                "completed_this_invocation": min(offset + 256, len(todo)),
                "elapsed_seconds": round(time.monotonic() - start, 1), "updated_at": now(),
                "length_stops": sum(r["finish_reason"] == "length" for r in seen.values()),
                "total_completion_tokens": sum(r["completion_tokens"] for r in seen.values()),
                "complete": len(seen) == sum(r["eligible"] for r in manifest),
            })
            print(f"{now()} {args.run}: {len(seen)} / {sum(r['eligible'] for r in manifest)} saved", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["prepare", "audit-selection", "infer"])
    p.add_argument("--root", type=Path, default=DEFAULT_OUT)
    p.add_argument("--run", choices=MODELS)
    p.add_argument("--limit", type=int, default=0, help="Smoke limit; compatible predictions are reused")
    args = p.parse_args()
    if args.action == "prepare":
        prepare(args.root)
    elif args.action == "audit-selection":
        audit_selection(args.root)
    else:
        if not args.run:
            p.error("infer requires --run")
        run_dir = args.root / "runs" / args.run
        run_dir.mkdir(parents=True, exist_ok=True)
        with (run_dir / "inference.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError(f"Another inference process owns run {args.run}") from None
            infer(args)


if __name__ == "__main__":
    main()
