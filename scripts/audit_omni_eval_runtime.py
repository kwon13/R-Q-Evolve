#!/usr/bin/env python3
"""Recompute CPU-only four-checkpoint config and rendered-prompt parity audit."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis/omni_math2_checkpoint_comparison"


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def digest_json(data: object) -> str:
    return digest_bytes(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                  separators=(",", ":")).encode())


def main() -> None:
    spec = importlib.util.spec_from_file_location(
        "omni_inference", ROOT / "scripts/eval_omni_math2_checkpoints.py")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)
    manifest_path = OUT / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    canonical_path = Path(inference.MODELS["rzero_256"])
    canonical = AutoTokenizer.from_pretrained(canonical_path, local_files_only=True)
    reference_prompts = [inference.build_prompt(canonical, row["problem"]) for row in rows]
    reference_ids = canonical(reference_prompts, add_special_tokens=True)["input_ids"]
    reference_cfg = AutoConfig.from_pretrained(canonical_path).to_dict()
    eligible = [i for i, row in enumerate(rows) if row["eligible"]]
    results = {}
    for name, value in inference.MODELS.items():
        path = Path(value)
        raw_config = json.loads((path / "config.json").read_text())
        tokenizer_adaptation = {"extra_special_tokens": {}} if name.startswith("rq_") else {}
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True,
                                                  **tokenizer_adaptation)
        overrides = {}
        if raw_config.get("rope_parameters"):
            rope = raw_config["rope_parameters"]
            if rope.get("rope_type") != "default":
                raise ValueError("Unreviewed non-default RoPE type")
            overrides["rope_theta"] = float(rope["rope_theta"])
        old_default = AutoConfig.from_pretrained(path).to_dict()
        effective = AutoConfig.from_pretrained(path, **overrides).to_dict()
        difference = {k: {"run": effective.get(k), "reference": reference_cfg.get(k)}
                      for k in sorted(set(effective) | set(reference_cfg))
                      if effective.get(k) != reference_cfg.get(k)}
        relevant_difference = {k: v for k, v in difference.items()
                               if k not in ("_name_or_path", "rope_parameters")}
        prompts = [inference.build_prompt(tokenizer, row["problem"]) for row in rows]
        tokens = tokenizer(prompts, add_special_tokens=True)["input_ids"]
        prompt_mismatch = [row["id"] for row, a, b in zip(rows, prompts, reference_prompts) if a != b]
        token_mismatch = [row["id"] for row, a, b in zip(rows, tokens, reference_ids) if a != b]
        special = {key: getattr(tokenizer, key) for key in
                   ("bos_token", "bos_token_id", "eos_token", "eos_token_id",
                    "pad_token", "pad_token_id")}
        files = {}
        for filename in ("config.json", "generation_config.json", "tokenizer.json",
                         "tokenizer_config.json", "chat_template.jinja"):
            asset = path / filename
            if asset.exists():
                files[filename] = {"bytes": asset.stat().st_size,
                                   "sha256": digest_bytes(asset.read_bytes())}
        results[name] = {
            "source_model_path": str(path), "resolved_model_path": str(path.resolve()),
            "source_files": files, "original_config": raw_config,
            "original_generation_config": json.loads((path / "generation_config.json").read_text()),
            "legacy_default_rope_theta": old_default.get("rope_theta"),
            "hf_overrides": overrides, "effective_rope_theta": effective.get("rope_theta"),
            "effective_config_differences_to_rzero_256": difference,
            "effective_model_parameter_differences_to_rzero_256": relevant_difference,
            "audit_loading_tokenizer_kwargs": tokenizer_adaptation,
            "inference_tokenizer_path": str(canonical_path if name.startswith("rq_") else path),
            "vocab_size_with_added_tokens": len(tokenizer),
            "vocabulary_sha256": digest_json(tokenizer.get_vocab()),
            "vocabulary_equal_to_canonical": tokenizer.get_vocab() == canonical.get_vocab(),
            "special_tokens": special,
            "chat_template_sha256": digest_bytes(tokenizer.chat_template.encode()),
            "chat_template_equal_to_canonical": tokenizer.chat_template == canonical.chat_template,
            "rendered_prompt_count": len(rows), "inference_prompt_count": len(eligible),
            "rendered_prompt_mismatched_ids": prompt_mismatch,
            "rendered_prompt_token_mismatched_ids": token_mismatch,
            "all_rendered_prompts_token_ids_sha256": digest_json(tokens),
            "eligible_rendered_prompts_token_ids_sha256": digest_json([tokens[i] for i in eligible]),
        }
        assert not relevant_difference, (name, relevant_difference)
        assert not prompt_mismatch and not token_mismatch, name
        assert tokenizer.get_vocab() == canonical.get_vocab(), name
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "packages": {p: importlib.metadata.version(p) for p in ("transformers", "tokenizers")},
        "manifest_path": str(manifest_path),
        "manifest_sha256": digest_bytes(manifest_path.read_bytes()),
        "build_prompt_source_path": str(ROOT / "scripts/eval_omni_math2_checkpoints.py"),
        "build_prompt_source_sha256": digest_bytes(inspect.getsource(inference.build_prompt).encode()),
        "system_prompt": inference.SYSTEM_PROMPT,
        "tokenization_kwargs": {"add_special_tokens": True},
        "canonical_tokenizer": str(canonical_path),
        "all_four_models_rendered_prompt_and_token_id_parity": True,
        "all_four_models_effective_model_config_parity": True,
        "generation_config_note": "RQ export declares max_new_tokens=2048; explicit common SamplingParams.max_tokens=4096 is required. Tokenizer BOS is None for all four; generation_config BOS is not inserted by the audited tokenizer.",
        "ignored_effective_config_differences": {"_name_or_path": "Provenance only",
             "rope_parameters": "New-export field preserved as metadata; legacy runtime uses explicitly overridden rope_theta=1000000"},
        "runs": results,
    }
    destination = OUT / "runtime_compatibility.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "models": len(results),
                      "all_rendered_prompts_per_model": len(rows),
                      "inference_prompts_per_model": len(eligible),
                      "different_prompt_token_sequences": 0}))


if __name__ == "__main__":
    main()
