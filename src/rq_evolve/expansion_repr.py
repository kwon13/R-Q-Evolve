"""Leakage-safe frozen-Solver representations for expansion experiments.

The module deliberately keeps ``torch`` and ``transformers`` imports inside
runtime methods.  Dataset/statistics tooling can therefore import the prompt,
layer-selection, hashing, and artifact helpers in a CPU-only environment.

Representation prompts contain exactly two messages:

* the repository's common Solver system instruction; and
* the final visible problem text.

There is no API parameter for an answer, reasoning trace, mutation plan,
generator source, or analysis metadata.  This makes accidental label/analysis
leakage difficult at the representation boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import numpy as np

from .prompts import SOLVER_SYSTEM_PROMPT


SCHEMA_VERSION = 1
POOLING_MODES = ("masked_mean", "last_prompt_token")


class LayerSelection(NamedTuple):
    """Pre-registered zero-based decoder-block selection."""

    primary: int
    adjacent: tuple[int, ...]

    @property
    def all_indices(self) -> tuple[int, ...]:
        return tuple(sorted({self.primary, *self.adjacent}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "indexing": "zero_based_decoder_block",
            "primary": self.primary,
            "adjacent": list(self.adjacent),
            "all": list(self.all_indices),
        }


@dataclass(slots=True)
class RepresentationArtifact:
    """In-memory representation arrays and their leakage-safe provenance."""

    arrays: dict[str, np.ndarray]
    metadata: dict[str, Any]


def render_representation_prompt(problem_text: str, tokenizer: Any) -> str:
    """Render only the common Solver instruction and visible problem text.

    ``answer`` and arbitrary metadata are intentionally not accepted by this
    function.  A tokenizer without an explicit chat template is rejected rather
    than silently switching to a different input format.
    """

    if not isinstance(problem_text, str):
        raise TypeError("problem_text must be a string")
    if not problem_text.strip():
        raise ValueError("problem_text must not be empty")
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("tokenizer must provide apply_chat_template")
    if getattr(tokenizer, "chat_template", None) in (None, ""):
        raise ValueError("tokenizer has no chat_template")

    messages = [
        {"role": "system", "content": SOLVER_SYSTEM_PROMPT},
        {"role": "user", "content": problem_text},
    ]
    rendered = apply_chat_template(
        messages,
        tokenize=False,
        # Match the prompt boundary used when the Solver actually starts its
        # response.  This adds only the template's assistant-start marker; it
        # does not add an answer or any analysis metadata.
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str) or not rendered:
        raise ValueError("tokenizer chat template returned no text")
    return rendered


def resolve_layers(
    num_hidden_layers: int,
    primary_fraction: float = 2 / 3,
) -> LayerSelection:
    """Resolve one upper-middle block and its immediate neighbors.

    The fraction is interpreted in one-based layer space and converted back to
    an explicit zero-based decoder-block index.  Consequently, 36 blocks at the
    default two-thirds fraction resolve to primary block 23 (the 24th block),
    with robustness blocks 22 and 24.
    """

    if isinstance(num_hidden_layers, bool) or int(num_hidden_layers) != num_hidden_layers:
        raise TypeError("num_hidden_layers must be an integer")
    num_hidden_layers = int(num_hidden_layers)
    if num_hidden_layers < 1:
        raise ValueError("num_hidden_layers must be >= 1")
    primary_fraction = float(primary_fraction)
    if not math.isfinite(primary_fraction) or not 0.0 < primary_fraction <= 1.0:
        raise ValueError("primary_fraction must be in (0, 1]")

    primary = math.ceil(num_hidden_layers * primary_fraction) - 1
    primary = min(num_hidden_layers - 1, max(0, primary))
    adjacent = tuple(
        index
        for index in (primary - 1, primary + 1)
        if 0 <= index < num_hidden_layers
    )
    return LayerSelection(primary=primary, adjacent=adjacent)


def l2_normalize(array: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize a batch of vectors in fp32."""

    values = np.asarray(array, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("representation array must have shape [batch, hidden]")
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if np.any(norms <= np.float32(eps)):
        raise ValueError("cannot normalize a zero prompt representation")
    return values / norms


class HFSelectedLayerExtractor:
    """Extract selected decoder-block states with forward hooks.

    Instantiate through :meth:`from_pretrained` for the confirmatory path.
    Direct construction is also supported for lightweight test doubles.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        checkpoint_source: str | Path | None = None,
        tokenizer_source: str | Path | None = None,
        layer_selection: LayerSelection | None = None,
        primary_fraction: float = 2 / 3,
        device: str | None = None,
        dtype: str | None = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.checkpoint_source = (
            str(Path(checkpoint_source).expanduser().resolve())
            if checkpoint_source is not None and Path(checkpoint_source).expanduser().exists()
            else (str(checkpoint_source) if checkpoint_source is not None else None)
        )
        self.tokenizer_source = (
            str(Path(tokenizer_source).expanduser().resolve())
            if tokenizer_source is not None and Path(tokenizer_source).expanduser().exists()
            else (
                str(tokenizer_source)
                if tokenizer_source is not None
                else self.checkpoint_source
            )
        )

        config = getattr(model, "config", None)
        num_hidden_layers = getattr(config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = len(self._decoder_layers(model))
        self.layer_selection = layer_selection or resolve_layers(
            int(num_hidden_layers),
            primary_fraction=primary_fraction,
        )
        invalid = [
            index
            for index in self.layer_selection.all_indices
            if index < 0 or index >= int(num_hidden_layers)
        ]
        if invalid:
            raise ValueError(
                f"layer indices outside [0, {int(num_hidden_layers) - 1}]: {invalid}"
            )

        self.device = device or self._infer_model_device(model)
        self.dtype = dtype or self._infer_model_dtype(model)
        self._provenance_cache: dict[str, Any] | None = None

        # Dropout must stay disabled for a frozen representation measurement.
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_source: str | Path,
        *,
        tokenizer_source: str | Path | None = None,
        primary_fraction: float = 2 / 3,
        layer_selection: LayerSelection | None = None,
        dtype: str = "bfloat16",
        device: str | None = None,
        trust_remote_code: bool = False,
    ) -> "HFSelectedLayerExtractor":
        """Load a Hugging Face causal LM with SDPA and no import-time GPU work."""

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        checkpoint_text = str(checkpoint_source)
        tokenizer_text = str(tokenizer_source or checkpoint_source)
        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype == "auto":
            torch_dtype: Any = "auto"
        else:
            torch_dtype = getattr(torch, str(dtype), None)
            if torch_dtype is None:
                raise ValueError(f"unknown torch dtype: {dtype!r}")

        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_text,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise ValueError("tokenizer has neither pad_token_id nor eos_token_id")
            tokenizer.pad_token = tokenizer.eos_token

        # One Qwen3-8B copy fits on one A100.  A single-device map also avoids a
        # transient full model copy when low_cpu_mem_usage is available.
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_text,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            attn_implementation="sdpa",
            low_cpu_mem_usage=True,
            device_map={"": resolved_device},
        )
        model.config.use_cache = False
        model.eval()
        return cls(
            model,
            tokenizer,
            checkpoint_source=checkpoint_source,
            tokenizer_source=tokenizer_source or checkpoint_source,
            layer_selection=layer_selection,
            primary_fraction=primary_fraction,
            device=resolved_device,
            dtype=dtype,
        )

    def extract(
        self,
        problem_texts: Sequence[str],
        *,
        batch_size: int = 8,
        max_length: int | None = None,
    ) -> RepresentationArtifact:
        """Extract normalized masked-mean and last-token states for each layer."""

        import torch

        if isinstance(problem_texts, (str, bytes)):
            raise TypeError("problem_texts must be a sequence of strings")
        problems = list(problem_texts)
        if not problems:
            raise ValueError("problem_texts must not be empty")
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        rendered_prompts = [
            render_representation_prompt(problem, self.tokenizer)
            for problem in problems
        ]
        prompt_hashes = [_sha256_text(prompt) for prompt in rendered_prompts]
        arrays: dict[str, list[np.ndarray]] = {
            _array_key(index, pooling): []
            for index in self.layer_selection.all_indices
            for pooling in POOLING_MODES
        }
        token_counts: list[int] = []
        token_id_hashes: list[str] = []

        layers = self._decoder_layers(self.model)
        backbone = self._backbone(self.model, layers)
        captured: dict[int, Any] = {}
        handles = []

        def make_hook(index: int):
            def hook(_module, _inputs, output):
                hidden = output[0] if isinstance(output, (tuple, list)) else output
                if not hasattr(hidden, "ndim") or hidden.ndim != 3:
                    raise RuntimeError(
                        f"decoder block {index} returned a non-[batch,seq,hidden] output"
                    )
                captured[index] = hidden

            return hook

        for index in self.layer_selection.all_indices:
            handles.append(layers[index].register_forward_hook(make_hook(index)))

        try:
            for start in range(0, len(rendered_prompts), int(batch_size)):
                prompt_batch = rendered_prompts[start : start + int(batch_size)]
                tokenized = self.tokenizer(
                    prompt_batch,
                    add_special_tokens=False,
                    padding=True,
                    # A confirmatory measurement must not silently remove part
                    # of a problem.  Tokenize in full and fail below if the
                    # preregistered prompt limit is exceeded.
                    truncation=False,
                    return_tensors="pt",
                )
                if "input_ids" not in tokenized or "attention_mask" not in tokenized:
                    raise ValueError("tokenizer must return input_ids and attention_mask")
                input_ids = tokenized["input_ids"]
                attention_mask = tokenized["attention_mask"]
                counts = attention_mask.sum(dim=1).to(dtype=torch.long)
                if bool((counts <= 0).any()):
                    raise ValueError("a rendered prompt tokenized to zero tokens")
                if max_length is not None and bool((counts > int(max_length)).any()):
                    largest = int(counts.max().item())
                    raise ValueError(
                        "a representation prompt exceeds max_length without "
                        f"truncation: largest={largest}, max_length={max_length}"
                    )
                token_counts.extend(int(value) for value in counts.tolist())
                token_id_hashes.extend(
                    _sha256_token_ids(
                        input_ids[row][attention_mask[row].to(dtype=torch.bool)].tolist()
                    )
                    for row in range(input_ids.shape[0])
                )

                model_inputs = {
                    key: value.to(self.device)
                    for key, value in tokenized.items()
                    if hasattr(value, "to")
                }
                mask = model_inputs["attention_mask"].to(dtype=torch.bool)
                captured.clear()
                with torch.inference_mode():
                    backbone(
                        **model_inputs,
                        use_cache=False,
                        return_dict=True,
                    )

                missing = set(self.layer_selection.all_indices) - set(captured)
                if missing:
                    raise RuntimeError(f"forward hooks did not capture layers: {sorted(missing)}")

                positions = torch.arange(
                    mask.shape[1],
                    device=mask.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand_as(mask)
                last_positions = positions.masked_fill(~mask, -1).argmax(dim=1)
                row_positions = torch.arange(mask.shape[0], device=mask.device)

                for index in self.layer_selection.all_indices:
                    hidden = captured[index].float()
                    float_mask = mask.unsqueeze(-1).to(dtype=hidden.dtype)
                    mean_pooled = (hidden * float_mask).sum(dim=1) / float_mask.sum(
                        dim=1
                    ).clamp_min(1.0)
                    last_pooled = hidden[row_positions, last_positions]
                    arrays[_array_key(index, "masked_mean")].append(
                        _torch_l2_normalize(mean_pooled).cpu().numpy().astype(
                            np.float32,
                            copy=False,
                        )
                    )
                    arrays[_array_key(index, "last_prompt_token")].append(
                        _torch_l2_normalize(last_pooled).cpu().numpy().astype(
                            np.float32,
                            copy=False,
                        )
                    )
        finally:
            for handle in handles:
                handle.remove()

        merged = {
            key: np.concatenate(chunks, axis=0)
            for key, chunks in arrays.items()
        }
        metadata: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "representation_definition": (
                "frozen_solver_selected_decoder_block_prompt_states"
            ),
            "checkpoint_source": self.checkpoint_source,
            "tokenizer_source": self.tokenizer_source,
            "attention_implementation": "sdpa",
            "model_mode": "eval",
            "use_cache": False,
            "dtype": self.dtype,
            "layer_indices": self.layer_selection.to_dict(),
            "pooling_modes": list(POOLING_MODES),
            "normalization": "fp32_l2",
            "num_problems": len(problems),
            "prompt_hashes_sha256": prompt_hashes,
            "token_id_hashes_sha256": token_id_hashes,
            "token_counts": token_counts,
            "stores_raw_prompts": False,
            "input_contract": "SOLVER_SYSTEM_PROMPT + visible problem_text only",
            "chat_generation_boundary": True,
        }
        return RepresentationArtifact(arrays=merged, metadata=metadata)

    def provenance(self) -> dict[str, Any]:
        """Return cached checkpoint/config/tokenizer content fingerprints."""

        if self._provenance_cache is None:
            self._provenance_cache = fingerprint_hf_sources(
                self.checkpoint_source,
                self.tokenizer_source,
            )
        return dict(self._provenance_cache)

    @staticmethod
    def _decoder_layers(model: Any) -> Sequence[Any]:
        candidates = [
            getattr(getattr(model, "model", None), "layers", None),
            getattr(model, "layers", None),
            getattr(getattr(model, "transformer", None), "h", None),
            getattr(getattr(model, "gpt_neox", None), "layers", None),
        ]
        for candidate in candidates:
            if candidate is not None and len(candidate) > 0:
                return candidate
        raise TypeError("could not locate decoder layers on the Hugging Face model")

    @staticmethod
    def _backbone(model: Any, layers: Sequence[Any]) -> Any:
        inner = getattr(model, "model", None)
        if inner is not None and getattr(inner, "layers", None) is layers:
            return inner
        return model

    @staticmethod
    def _infer_model_device(model: Any) -> str:
        try:
            return str(next(model.parameters()).device)
        except (AttributeError, StopIteration):
            return "cpu"

    @staticmethod
    def _infer_model_dtype(model: Any) -> str:
        try:
            return str(next(model.parameters()).dtype).replace("torch.", "")
        except (AttributeError, StopIteration):
            return "unknown"


def save_representation_artifact(
    path: str | Path,
    artifact: RepresentationArtifact,
    *,
    extractor: HFSelectedLayerExtractor | None = None,
    checkpoint_source: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
) -> tuple[Path, Path]:
    """Save arrays to ``.npz`` and provenance/shape metadata to adjacent JSON."""

    npz_path, json_path = _artifact_paths(path)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    arrays = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in artifact.arrays.items()
    }
    if not arrays:
        raise ValueError("artifact has no representation arrays")
    expected_rows = int(artifact.metadata.get("num_problems", -1))
    for key, value in arrays.items():
        if value.ndim != 2:
            raise ValueError(f"{key} must have shape [num_problems, hidden]")
        if expected_rows >= 0 and value.shape[0] != expected_rows:
            raise ValueError(
                f"{key} has {value.shape[0]} rows, expected {expected_rows}"
            )

    metadata = dict(artifact.metadata)
    if extractor is not None:
        provenance = extractor.provenance()
    else:
        provenance = fingerprint_hf_sources(
            checkpoint_source or metadata.get("checkpoint_source"),
            tokenizer_source or metadata.get("tokenizer_source"),
        )
    metadata.update(provenance)
    metadata["arrays"] = {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in arrays.items()
    }
    metadata["npz_sha256"] = None

    tmp_npz = npz_path.with_name(npz_path.name + ".tmp.npz")
    np.savez_compressed(tmp_npz, **arrays)
    os.replace(tmp_npz, npz_path)
    metadata["npz_sha256"] = _sha256_file(npz_path)

    tmp_json = json_path.with_name(json_path.name + ".tmp")
    tmp_json.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp_json, json_path)
    return npz_path, json_path


def load_representation_artifact(path: str | Path) -> RepresentationArtifact:
    """Load and integrity-check an ``.npz`` + adjacent JSON artifact pair."""

    npz_path, json_path = _artifact_paths(path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    if not json_path.is_file():
        raise FileNotFoundError(json_path)
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    expected_digest = metadata.get("npz_sha256")
    if expected_digest and _sha256_file(npz_path) != expected_digest:
        raise ValueError(f"representation NPZ checksum mismatch: {npz_path}")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {
            key: np.asarray(payload[key], dtype=np.float32)
            for key in payload.files
        }
    return RepresentationArtifact(arrays=arrays, metadata=metadata)


def fingerprint_hf_sources(
    checkpoint_source: str | Path | None,
    tokenizer_source: str | Path | None = None,
) -> dict[str, Any]:
    """Hash HF weights, config, and tokenizer sources as separate groups."""

    tokenizer_source = tokenizer_source or checkpoint_source
    checkpoint_hash, checkpoint_mode = _hash_source_group(
        checkpoint_source,
        patterns=(
            "*.safetensors",
            "pytorch_model*.bin",
            "model*.bin",
            "model*.pt",
        ),
    )
    config_hash, config_mode = _hash_source_group(
        checkpoint_source,
        patterns=("config.json",),
    )
    tokenizer_hash, tokenizer_mode = _hash_source_group(
        tokenizer_source,
        patterns=(
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "added_tokens.json",
            "chat_template.jinja",
            "vocab.json",
            "merges.txt",
            "*.model",
        ),
    )
    return {
        "hash_algorithm": "sha256",
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_hash_mode": checkpoint_mode,
        "config_sha256": config_hash,
        "config_hash_mode": config_mode,
        "tokenizer_sha256": tokenizer_hash,
        "tokenizer_hash_mode": tokenizer_mode,
    }


def _torch_l2_normalize(tensor: Any, eps: float = 1e-12) -> Any:
    norms = tensor.norm(p=2, dim=-1, keepdim=True).clamp_min(eps)
    return tensor / norms


def _array_key(layer_index: int, pooling: str) -> str:
    return f"layer_{int(layer_index):03d}__{pooling}"


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    base = Path(path).expanduser()
    if base.suffix in {".npz", ".json"}:
        base = base.with_suffix("")
    return base.with_suffix(".npz"), base.with_suffix(".json")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_token_ids(token_ids: Sequence[int]) -> str:
    canonical = ",".join(str(int(value)) for value in token_ids)
    return _sha256_text(canonical)


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_source_group(
    source: str | Path | None,
    *,
    patterns: Sequence[str],
) -> tuple[str, str]:
    """Hash a local file group, or hash the immutable source identifier."""

    if source is None:
        return _sha256_text("<unspecified>"), "unspecified_identifier"
    path = Path(source).expanduser()
    if path.is_file():
        return _sha256_file(path), "file_content"
    if not path.is_dir():
        return _sha256_text(str(source)), "source_identifier"

    files: dict[str, Path] = {}
    for pattern in patterns:
        for candidate in path.glob(pattern):
            if candidate.is_file():
                files[candidate.relative_to(path).as_posix()] = candidate
    if not files:
        return _sha256_text(str(path.resolve())), "empty_local_source_identifier"

    digest = hashlib.sha256()
    for relative, candidate in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(candidate.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        with candidate.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
        digest.update(b"\0")
    return digest.hexdigest(), "grouped_file_content"


__all__ = [
    "HFSelectedLayerExtractor",
    "LayerSelection",
    "RepresentationArtifact",
    "fingerprint_hf_sources",
    "l2_normalize",
    "load_representation_artifact",
    "render_representation_prompt",
    "resolve_layers",
    "save_representation_artifact",
]
