#!/usr/bin/env python3
"""Probe frozen-Solver in-context transfer with compiled child worked examples.

For each operator, the script constructs three paired contexts:

* no-shot: target parent problem only;
* plain-shot: one compiled plain child plus a verified canonical solution;
* reasoning-shot: one compiled reasoning child plus the same condition-blind
  solution template.

It then greedily solves held-out parent seeds, grades the target response, and
streams hidden-state transition norms from a Hugging Face forward pass.  Full
``[tokens, layers, hidden]`` tensors are never retained; only StALT components
and selected prompt-last-token vectors are saved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rq_evolve.expansion_repr import resolve_layers  # noqa: E402
from rq_evolve.expansion_trajectory import (  # noqa: E402
    compute_stalt_from_transitions,
)
from rq_evolve.program import ProblemProgram  # noqa: E402
from rq_evolve.reward import answers_match, extract_boxed  # noqa: E402
from rq_evolve.shot_internal import (  # noqa: E402
    SHOT_DIAGNOSTIC_SCHEMA_VERSION,
    SHOT_PRESENTATIONS,
    build_solver_messages,
    canonical_shot_solution,
    conversation_hash,
    cosine_similarity,
    stable_text_hash,
    summarize_shot_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use compiled mutation problems as worked one-shot examples and "
            "measure frozen-Solver accuracy plus StALT internal trajectories."
        )
    )
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--seed-program", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer")
    parser.add_argument(
        "--operator",
        choices=("in_depth", "in_breadth", "both"),
        default="both",
    )
    parser.add_argument("--shot-seed", type=int, default=0)
    parser.add_argument(
        "--shot-presentation",
        choices=SHOT_PRESENTATIONS,
        default="assistant_turn",
        help=(
            "assistant_turn uses user(example) → assistant(solution) → "
            "user(target). user_context places the example problem, solution, "
            "and target together in one user message."
        ),
    )
    parser.add_argument("--target-seeds", type=int, nargs="+", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--primary-layer-fraction", type=float, default=2 / 3)
    parser.add_argument("--stalt-tau", type=float, default=1.0)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Unique prompts generated per forward pass. Defaults to 1, which "
            "reproduces the original one-at-a-time path exactly."
        ),
    )
    parser.add_argument(
        "--verify-batch-equivalence",
        type=int,
        default=0,
        metavar="N",
        help=(
            "Re-run the first N prompts at batch size 1 and abort if any "
            "response differs. Guards against padding changing greedy output."
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decoder_layers(model: Any) -> list[Any]:
    candidates = (
        getattr(getattr(model, "model", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),
        getattr(getattr(model, "gpt_neox", None), "layers", None),
    )
    for candidate in candidates:
        if candidate is not None:
            return list(candidate)
    raise TypeError("could not locate decoder layers on the Hugging Face model")


class FrozenSolverShotRunner:
    """Greedy HF generation plus memory-bounded internal transition capture."""

    def __init__(
        self,
        model_source: str,
        *,
        tokenizer_source: str | None,
        dtype: str,
        device: str,
        primary_layer_fraction: float,
        max_new_tokens: int,
        stalt_tau: float,
        trust_remote_code: bool,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device
        self.max_new_tokens = int(max_new_tokens)
        self.stalt_tau = float(stalt_tau)
        tokenizer_name = tokenizer_source or model_source
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
        )
        if not callable(getattr(self.tokenizer, "apply_chat_template", None)):
            raise TypeError("tokenizer must provide apply_chat_template")
        if getattr(self.tokenizer, "chat_template", None) in (None, ""):
            raise ValueError("tokenizer has no chat template")
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has no pad or EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left padding puts every prompt's final token at the same index, so a
        # batch shares one prompt boundary and the per-sample slicing below stays
        # exact. Right padding would misalign both the response span and the
        # prompt-last vector.
        self.tokenizer.padding_side = "left"

        if dtype == "auto":
            torch_dtype: Any = "auto"
        else:
            torch_dtype = getattr(torch, dtype, None)
            if torch_dtype is None:
                raise ValueError(f"unknown torch dtype: {dtype!r}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": device},
        )
        self.model.eval()
        self.model.config.use_cache = True
        self.layers = _decoder_layers(self.model)
        self.layer_selection = resolve_layers(
            len(self.layers),
            primary_fraction=primary_layer_fraction,
        )

    def _render(self, messages: list[dict[str, str]]) -> str:
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str) or not rendered:
            raise ValueError("chat template rendered an empty prompt")
        return rendered

    def _internal_metrics(
        self,
        full_ids: Any,
        attention_mask: Any,
        *,
        prompt_tokens: int,
        response_lengths: list[int],
    ) -> list[tuple[dict[str, Any], dict[int, np.ndarray]]]:
        """Capture per-sample transition metrics from one batched forward pass.

        ``prompt_tokens`` is the shared left-padded prompt boundary, so every
        sample's response starts at the same index. ``response_lengths`` trims
        each sample to its own generated span; without that trim the right-hand
        generation padding of an early-finishing sample would contribute
        pad-to-pad transitions and deflate its StALT.
        """
        torch = self.torch
        batch = int(full_ids.shape[0])
        if len(response_lengths) != batch:
            raise ValueError("response_lengths must cover every batch row")
        for length in response_lengths:
            if length < 2:
                raise ValueError(
                    "at least two generated tokens are required for StALT"
                )
        temporal_columns: list[list[np.ndarray]] = [[] for _ in range(batch)]
        layer_columns: list[list[np.ndarray]] = [[] for _ in range(batch)]
        selected_vectors: list[dict[int, np.ndarray]] = [
            {} for _ in range(batch)
        ]
        previous_response: list[Any] = [None] * batch

        def consume(hidden: Any, decoder_index: int | None) -> None:
            values = hidden[0] if isinstance(hidden, (tuple, list)) else hidden
            if values.ndim != 3 or values.shape[0] != batch:
                raise ValueError(
                    "decoder hidden output must have shape [batch,seq,hidden]"
                )
            for row in range(batch):
                response = values[
                    row, prompt_tokens : prompt_tokens + response_lengths[row]
                ].detach()
                temporal = torch.linalg.vector_norm(
                    response[1:].float() - response[:-1].float(),
                    dim=-1,
                )
                temporal_columns[row].append(temporal.cpu().numpy())
                if previous_response[row] is not None:
                    layer_delta = torch.linalg.vector_norm(
                        response[1:].float()
                        - previous_response[row][1:].float(),
                        dim=-1,
                    )
                    layer_columns[row].append(layer_delta.cpu().numpy())
                previous_response[row] = response
                if (
                    decoder_index is not None
                    and decoder_index in self.layer_selection.all_indices
                ):
                    selected_vectors[row][decoder_index] = (
                        values[row, prompt_tokens - 1]
                        .detach()
                        .float()
                        .cpu()
                        .numpy()
                    )

        handles = [
            self.model.get_input_embeddings().register_forward_hook(
                lambda _module, _inputs, output: consume(output, None)
            )
        ]
        for index, layer in enumerate(self.layers):
            handles.append(
                layer.register_forward_hook(
                    lambda _module, _inputs, output, layer_index=index: consume(
                        output,
                        layer_index,
                    )
                )
            )
        try:
            base_model = getattr(self.model, "model", None)
            if base_model is None:
                raise TypeError("causal LM does not expose its decoder base model")
            with torch.inference_mode():
                base_model(
                    input_ids=full_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                )
        finally:
            for handle in handles:
                handle.remove()

        results: list[tuple[dict[str, Any], dict[int, np.ndarray]]] = []
        for row in range(batch):
            if len(temporal_columns[row]) != len(self.layers) + 1:
                raise RuntimeError(
                    "did not capture every temporal transition column"
                )
            if len(layer_columns[row]) != len(self.layers):
                raise RuntimeError(
                    "did not capture every adjacent-layer transition"
                )
            if set(selected_vectors[row]) != set(
                self.layer_selection.all_indices
            ):
                raise RuntimeError("did not capture every selected prompt vector")
            metrics = compute_stalt_from_transitions(
                np.stack(temporal_columns[row], axis=1),
                np.stack(layer_columns[row], axis=1),
                tau=self.stalt_tau,
            )
            results.append((metrics, selected_vectors[row]))
        return results

    def run(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self.run_batch([messages])[0]

    def run_batch(
        self,
        batch_messages: list[list[dict[str, str]]],
    ) -> list[dict[str, Any]]:
        """Greedy-generate a batch and capture each sample's internal metrics.

        A single-item batch needs no padding, so it reproduces the original
        one-at-a-time path exactly; larger batches are verified against it by
        ``--verify-batch-equivalence``.
        """
        torch = self.torch
        if not batch_messages:
            return []
        rendered = [self._render(messages) for messages in batch_messages]
        encoded = self.tokenizer(
            rendered,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)
        prompt_tokens = int(input_ids.shape[1])
        max_positions = int(
            getattr(self.model.config, "max_position_embeddings", 0) or 0
        )
        if max_positions and prompt_tokens + self.max_new_tokens > max_positions:
            raise ValueError(
                "prompt plus max_new_tokens exceeds model context: "
                f"{prompt_tokens}+{self.max_new_tokens}>{max_positions}"
            )
        with torch.inference_mode():
            sequences = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=self.max_new_tokens,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        generated = sequences[:, prompt_tokens:]
        eos_id = self.tokenizer.eos_token_id
        response_lengths: list[int] = []
        for row in range(generated.shape[0]):
            row_ids = generated[row]
            hit = (row_ids == eos_id).nonzero()
            # Keep the EOS token itself, matching the unbatched path where
            # generation halts immediately after emitting it.
            length = (
                int(hit[0].item()) + 1 if hit.numel() else int(row_ids.shape[0])
            )
            response_lengths.append(length)

        # The metrics pass must see the same mask generation used: left padding
        # is masked out, and so is everything after each sample's EOS.
        metrics_mask = torch.zeros_like(sequences)
        metrics_mask[:, :prompt_tokens] = attention_mask
        for row, length in enumerate(response_lengths):
            metrics_mask[row, prompt_tokens : prompt_tokens + length] = 1

        per_sample = self._internal_metrics(
            sequences,
            metrics_mask,
            prompt_tokens=prompt_tokens,
            response_lengths=response_lengths,
        )

        outputs: list[dict[str, Any]] = []
        for row, (metrics, vectors) in enumerate(per_sample):
            response = self.tokenizer.decode(
                generated[row, : response_lengths[row]],
                skip_special_tokens=True,
            ).strip()
            true_prompt_tokens = int(attention_mask[row].sum().item())
            outputs.append(
                {
                    "rendered_prompt": rendered[row],
                    "prompt_tokens": true_prompt_tokens,
                    "response": response,
                    "response_hash": stable_text_hash(response),
                    "metrics": metrics,
                    "prompt_vectors": vectors,
                }
            )
        return outputs


class ShotSeedUnavailable(Exception):
    """The requested shot seed is not in the comparison run's seed set.

    Deliberately not a ``ValueError``: ``_build_shots`` skips an operator on
    ValueError, which is right for a genuinely missing candidate but wrong for a
    misconfigured seed. That distinction is what silently emptied an entire
    diagnostic run, so this one propagates.
    """


def _semantic_row(method_dir: Path, shot_seed: int) -> dict[str, Any]:
    payload = _read_json(method_dir / "05_family_semantics.json")
    rows = [
        row
        for row in payload.get("per_seed", [])
        if int(row.get("seed", -1)) == int(shot_seed)
    ]
    if len(rows) != 1:
        available = sorted(
            int(row["seed"])
            for row in payload.get("per_seed", [])
            if "seed" in row
        )
        raise ShotSeedUnavailable(
            f"{method_dir} has {len(rows)} semantic rows for shot seed "
            f"{shot_seed}. Available seeds: {available}. --shot-seed must be "
            "one of the comparison run's evaluation seeds; a shot seed outside "
            "that set yields no worked example and silently empties the whole "
            "diagnostic."
        )
    return dict(rows[0])


def _build_shots(
    comparison_root: Path,
    operators: tuple[str, ...],
    shot_seed: int,
) -> tuple[
    dict[str, dict[str, dict[str, Any]]],
    list[dict[str, str]],
]:
    summaries = _read_json(comparison_root / "summaries.json")
    by_key = {
        (str(row["operator"]), str(row["method"])): row
        for row in summaries
    }
    shots: dict[str, dict[str, dict[str, Any]]] = {}
    skipped: list[dict[str, str]] = []
    for operator in operators:
        operator_shots: dict[str, dict[str, Any]] = {}
        try:
            for condition, method in (
                ("plain_shot", "legacy"),
                ("reasoning_shot", "metacognitive"),
            ):
                summary = by_key.get((operator, method))
                if summary is None:
                    raise ValueError(f"{operator}/{method} summary is missing")
                if summary.get("generation_path") not in {
                    "registered_compiled",
                    "belief_probe_compiled",
                }:
                    raise ValueError(
                        f"{operator}/{method} has no selected compiled mutation "
                        f"(status={summary.get('status')})"
                    )
                if summary.get("family_semantics_valid") is not True:
                    raise ValueError(
                        f"{operator}/{method} did not pass the family semantic gate"
                    )
                configured_method_dir = summary.get("selected_candidate_dir")
                method_dir = (
                    Path(str(configured_method_dir)).expanduser().resolve()
                    if configured_method_dir
                    else comparison_root / operator / method
                )
                compiler = _read_json(
                    method_dir / "04_compiler_validation.json"
                )
                semantic = _semantic_row(method_dir, shot_seed)
                family = str(summary["generator_family"])
                solution = canonical_shot_solution(family, semantic)
                operator_shots[condition] = {
                    "operator": operator,
                    "condition": condition,
                    "method": method,
                    "shot_seed": int(shot_seed),
                    "generator_family": family,
                    "family_variant": compiler.get("family_variant"),
                    "compiler_source_hash": compiler.get("source_hash"),
                    "problem": semantic["problem"],
                    "answer": semantic["answer"],
                    "solution": solution,
                    "shot_hash": stable_text_hash(
                        str(semantic["problem"]) + "\n" + solution
                    ),
                }
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            skipped.append({"operator": operator, "reason": str(exc)})
            continue
        shots[operator] = operator_shots
    if not shots:
        detail = "; ".join(
            f"{row['operator']}: {row['reason']}" for row in skipped
        )
        raise RuntimeError(
            "no operator yielded a worked example, so the diagnostic would "
            f"produce an empty report: {detail}"
        )
    return shots, skipped


def _json_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, np.generic):
            result[key] = value.item()
        else:
            result[key] = value
    return result


def _report(summary: dict[str, Any], shots: dict[str, Any]) -> str:
    lines = [
        "# One-shot internal trajectory diagnostic",
        "",
        (
            "Frozen-Solver in-context transfer only; this is not persistent "
            "post-training capability expansion."
        ),
        f"- Shot presentation: `{summary.get('shot_presentation', 'unknown')}`",
        "",
        "| operator | condition | variant | targets | accuracy | mean StALT | "
        "path length | tokens | prompt cosine to no-shot |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        lines.append(
            "| {operator} | {condition} | {variant} | {n_targets} | "
            "{accuracy:.3f} | {mean_stalt:.6f} | {path:.6f} | "
            "{tokens:.1f} | {cosine:.6f} |".format(
                operator=row["operator"],
                condition=row["condition"],
                variant=row.get("family_variant") or "—",
                n_targets=row["n_targets"],
                accuracy=row["accuracy"],
                mean_stalt=row["mean_stalt"],
                path=row["mean_temporal_path_length"],
                tokens=row["mean_generated_tokens"],
                cosine=row["mean_prompt_last_cosine_to_no_shot"],
            )
        )
    lines.extend(["", "## Plain vs. reasoning contrasts", ""])
    for contrast in summary["contrasts"]:
        lines.extend(
            [
                f"### {contrast['operator']}",
                "",
                (
                    "- Accuracy (reasoning − plain): "
                    f"`{contrast['accuracy_reasoning_minus_plain']:.3f}`"
                ),
                (
                    "- StALT (reasoning − plain): "
                    f"`{contrast['stalt_reasoning_minus_plain']:.6f}`"
                ),
                (
                    "- Path length (reasoning − plain): "
                    f"`{contrast['path_length_reasoning_minus_plain']:.6f}`"
                ),
                "",
            ]
        )
    lines.extend(["## Shot identity audit", ""])
    for operator, rows in shots.items():
        same = rows["plain_shot"]["shot_hash"] == rows["reasoning_shot"]["shot_hash"]
        lines.append(
            f"- `{operator}` plain/reasoning worked examples identical: `{same}`"
        )
    if summary.get("skipped_operators"):
        lines.extend(["", "## Skipped operators", ""])
        for item in summary["skipped_operators"]:
            lines.append(
                f"- `{item['operator']}`: {item['reason']}"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 2:
        raise ValueError("--max-new-tokens must be >= 2")
    target_seeds = list(dict.fromkeys(int(seed) for seed in args.target_seeds))
    if not target_seeds:
        raise ValueError("--target-seeds must not be empty")
    if args.shot_seed in target_seeds:
        raise ValueError(
            "--shot-seed must not appear in --target-seeds; use a held-out seed"
        )
    comparison_root = args.comparison_root.expanduser().resolve()
    seed_program = args.seed_program.expanduser().resolve()
    if not (comparison_root / "manifest.json").is_file():
        raise FileNotFoundError(f"comparison manifest not found: {comparison_root}")
    requested_operators = (
        ("in_depth", "in_breadth")
        if args.operator == "both"
        else (args.operator,)
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else comparison_root / "shot_internal"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    parent = ProblemProgram.from_file(seed_program)
    targets = {}
    for seed in target_seeds:
        instance = parent.execute(seed)
        if instance is None:
            raise RuntimeError(f"parent failed at target seed {seed}")
        targets[seed] = instance
    shots, skipped_operators = _build_shots(
        comparison_root,
        requested_operators,
        args.shot_seed,
    )
    operators = tuple(shots)
    _write_json(
        output_dir / "preflight.json",
        {
            "requested_operators": list(requested_operators),
            "evaluated_operators": list(operators),
            "skipped_operators": skipped_operators,
        },
    )
    for item in skipped_operators:
        print(
            f"[shot-internal] skip {item['operator']}: {item['reason']}"
        )
    if not operators:
        empty_summary = {
            "rows": [],
            "contrasts": [],
            "requested_operators": list(requested_operators),
            "evaluated_operators": [],
            "skipped_operators": skipped_operators,
            "status": "skipped_no_paired_compiled_mutation",
            "shot_presentation": args.shot_presentation,
        }
        _write_json(output_dir / "summary.json", empty_summary)
        (output_dir / "REPORT.md").write_text(
            _report(empty_summary, {}),
            encoding="utf-8",
        )
        print(
            "[shot-internal] no paired compiled mutation; diagnostic skipped "
            f"without loading the model -> {output_dir / 'preflight.json'}"
        )
        return
    _write_json(output_dir / "shot_examples.json", shots)

    runner = FrozenSolverShotRunner(
        args.model,
        tokenizer_source=args.tokenizer,
        dtype=args.dtype,
        device=args.device,
        primary_layer_fraction=args.primary_layer_fraction,
        max_new_tokens=args.max_new_tokens,
        stalt_tau=args.stalt_tau,
        trust_remote_code=args.trust_remote_code,
    )

    cache: dict[str, dict[str, Any]] = {}
    baseline_vectors: dict[tuple[int, int], np.ndarray] = {}
    records: list[dict[str, Any]] = []
    vector_arrays: list[np.ndarray] = []
    vector_record_ids: list[str] = []

    # Phase 1: collect every distinct conversation, then run them in batches.
    # The record loop below is unchanged and simply reads the filled cache, so
    # batching cannot alter which prompts are executed or how they are paired.
    plan: list[tuple[str, list[dict[str, str]]]] = []
    planned: set[str] = set()
    for operator in operators:
        for condition in ("no_shot", "plain_shot", "reasoning_shot"):
            shot = shots[operator].get(condition)
            for target_seed in target_seeds:
                target = targets[target_seed]
                messages = build_solver_messages(
                    target.problem,
                    shot_problem=(shot["problem"] if shot else None),
                    shot_solution=(shot["solution"] if shot else None),
                    shot_presentation=args.shot_presentation,
                )
                prompt_hash = conversation_hash(messages)
                if prompt_hash not in planned:
                    planned.add(prompt_hash)
                    plan.append((prompt_hash, messages))

    batch_size = max(1, int(args.batch_size))
    print(
        f"[shot-internal] {len(plan)} unique prompts, batch size {batch_size}"
    )
    for start in range(0, len(plan), batch_size):
        chunk = plan[start : start + batch_size]
        outputs = runner.run_batch([messages for _, messages in chunk])
        for (prompt_hash, _messages), output in zip(chunk, outputs):
            cache[prompt_hash] = output
        print(
            f"[shot-internal] generated {min(start + batch_size, len(plan))}"
            f"/{len(plan)}"
        )

    if args.verify_batch_equivalence and batch_size > 1:
        checked = plan[: args.verify_batch_equivalence]
        mismatches = []
        for prompt_hash, messages in checked:
            single = runner.run(messages)
            batched = cache[prompt_hash]
            if single["response_hash"] != batched["response_hash"]:
                mismatches.append(prompt_hash)
        print(
            f"[shot-internal] batch-equivalence check: "
            f"{len(checked) - len(mismatches)}/{len(checked)} responses identical"
        )
        if mismatches:
            raise RuntimeError(
                "batched generation diverged from single-sequence generation on "
                f"{len(mismatches)} prompt(s); rerun with --batch-size 1"
            )

    # ``cache_hit`` keeps its original meaning: this record reuses a prompt an
    # earlier record already claimed, rather than "was prefilled in phase 1".
    consumed: set[str] = set()
    for operator in operators:
        for condition in ("no_shot", "plain_shot", "reasoning_shot"):
            shot = shots[operator].get(condition)
            for target_seed in target_seeds:
                target = targets[target_seed]
                messages = build_solver_messages(
                    target.problem,
                    shot_problem=(shot["problem"] if shot else None),
                    shot_solution=(shot["solution"] if shot else None),
                    shot_presentation=args.shot_presentation,
                )
                prompt_hash = conversation_hash(messages)
                cache_hit = prompt_hash in consumed
                consumed.add(prompt_hash)
                run = cache[prompt_hash]
                selected_vectors = run["prompt_vectors"]
                if condition == "no_shot":
                    for layer, vector in selected_vectors.items():
                        baseline_vectors[(target_seed, layer)] = vector
                cosines = [
                    cosine_similarity(
                        vector,
                        baseline_vectors[(target_seed, layer)],
                    )
                    for layer, vector in selected_vectors.items()
                ]
                predicted = extract_boxed(run["response"])
                correct = (
                    predicted is not None
                    and answers_match(predicted, target.answer)
                )
                record_id = (
                    f"{operator}:{condition}:target_seed={target_seed}"
                )
                metrics = _json_metrics(run["metrics"])
                record = {
                    "schema_version": SHOT_DIAGNOSTIC_SCHEMA_VERSION,
                    "record_id": record_id,
                    "operator": operator,
                    "condition": condition,
                    "shot_presentation": args.shot_presentation,
                    "target_seed": target_seed,
                    "target_problem": target.problem,
                    "target_answer": target.answer,
                    "generator_family": (
                        shot.get("generator_family") if shot else None
                    ),
                    "family_variant": (
                        shot.get("family_variant") if shot else None
                    ),
                    "shot_seed": args.shot_seed if shot else None,
                    "shot_hash": shot.get("shot_hash") if shot else None,
                    "conversation_hash": prompt_hash,
                    "cache_hit": cache_hit,
                    "prompt_tokens": run["prompt_tokens"],
                    "response": run["response"],
                    "response_hash": run["response_hash"],
                    "predicted_answer": predicted,
                    "correct": bool(correct),
                    "prompt_last_cosine_to_no_shot": float(np.mean(cosines)),
                    "selected_layers_zero_based": list(
                        runner.layer_selection.all_indices
                    ),
                    **metrics,
                }
                records.append(record)
                vector_arrays.append(
                    np.stack(
                        [
                            selected_vectors[layer]
                            for layer in runner.layer_selection.all_indices
                        ],
                        axis=0,
                    ).astype(np.float32)
                )
                vector_record_ids.append(record_id)
                print(
                    f"[shot-internal] {record_id} correct={correct} "
                    f"stalt={record['stalt']:.6f} cache_hit={cache_hit}"
                )

    summary = summarize_shot_records(records)
    summary["shot_presentation"] = args.shot_presentation
    summary["requested_operators"] = list(requested_operators)
    summary["evaluated_operators"] = list(operators)
    summary["skipped_operators"] = skipped_operators
    summary["shot_identity"] = {
        operator: {
            "plain_shot_hash": rows["plain_shot"]["shot_hash"],
            "reasoning_shot_hash": rows["reasoning_shot"]["shot_hash"],
            "identical": (
                rows["plain_shot"]["shot_hash"]
                == rows["reasoning_shot"]["shot_hash"]
            ),
        }
        for operator, rows in shots.items()
    }
    _write_jsonl(output_dir / "records.jsonl", records)
    _write_json(output_dir / "summary.json", summary)
    npz_path = output_dir / "prompt_last_vectors.npz"
    np.savez_compressed(
        npz_path,
        vectors=np.stack(vector_arrays, axis=0),
        record_ids=np.asarray(vector_record_ids),
        selected_layers=np.asarray(runner.layer_selection.all_indices),
    )
    report = _report(summary, shots)
    (output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    source_manifest = comparison_root / "manifest.json"
    manifest = {
        "schema_version": SHOT_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic": "frozen_solver_one_shot_internal_trajectory",
        "comparison_root": str(comparison_root),
        "comparison_manifest_sha256": _file_sha256(source_manifest),
        "diagnostic_script_sha256": _file_sha256(Path(__file__).resolve()),
        "shot_internal_source_sha256": _file_sha256(
            ROOT / "src" / "rq_evolve" / "shot_internal.py"
        ),
        "trajectory_source_sha256": _file_sha256(
            ROOT / "src" / "rq_evolve" / "expansion_trajectory.py"
        ),
        "seed_program": str(seed_program),
        "seed_program_sha256": _file_sha256(seed_program),
        "model": args.model,
        "tokenizer": args.tokenizer or args.model,
        "device": args.device,
        "dtype": args.dtype,
        "generation": {
            "do_sample": False,
            "max_new_tokens": args.max_new_tokens,
            # Recorded because batching changes padding, and padding can in
            # principle change a greedy decode even though decoding is
            # deterministic at any fixed batch size.
            "batch_size": batch_size,
            "padding_side": "left",
            "verified_batch_equivalence_prompts": (
                int(args.verify_batch_equivalence) if batch_size > 1 else None
            ),
        },
        "shot_seed": args.shot_seed,
        "shot_presentation": args.shot_presentation,
        "target_seeds": target_seeds,
        "requested_operators": list(requested_operators),
        "operators": list(operators),
        "skipped_operators": skipped_operators,
        "stalt_tau": args.stalt_tau,
        "layer_selection": runner.layer_selection.to_dict(),
        "unique_inference_prompts": len(cache),
        "records": len(records),
        "prompt_vectors_npz": str(npz_path),
        "prompt_vectors_npz_sha256": _file_sha256(npz_path),
        "interpretation": (
            "in-context elicitation diagnostic; no model weights were updated"
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    print(f"[shot-internal] saved to {output_dir}")


if __name__ == "__main__":
    main()
