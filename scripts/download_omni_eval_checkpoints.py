#!/usr/bin/env python3
"""Download only the two approved RQ checkpoints, pinned and verified.

Existing final files are never overwritten: they must match the pinned Hub
metadata. Hugging Face's local-dir cache provides resumable partial downloads.
Each checkpoint receives a source plan and a manifest of actual file hashes.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import tempfile
import time
from typing import Any

# The public repositories support standard HTTP range downloads. Avoid an
# optional Xet runtime dependency and keep interrupted HTTP downloads resumable.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")

from huggingface_hub import HfApi, hf_hub_download, hf_hub_url  # noqa: E402
import requests  # noqa: E402


RUN_ROOT = Path(
    "/data1/yhoon113/R-Q-Evolve/rq_output/"
    "rq_evolve_8b_domain_type_35cell_8gpu"
)
APPROVED_STEPS = (256, 224)
SKIP_FILES = {".gitattributes", "README.md"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now()}] {message}", flush=True)


def save_json(path: Path, data: Any) -> None:
    """Atomically replace this script's own metadata, not checkpoint assets."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
    ) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def retry(operation: Any, description: str) -> Any:
    for attempt in range(1, 7):
        try:
            return operation()
        except Exception as exc:
            # Do not log exception strings: redirects may contain signed URLs.
            log(f"{description}: attempt {attempt}/6 failed ({type(exc).__name__})")
            if attempt == 6:
                raise RuntimeError(f"{description} failed after six attempts") from None
            time.sleep(min(5 * attempt, 20))


def source_plan(target: Path, repo_id: str, step: int) -> dict[str, Any]:
    plan_path = target / "download_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("repo_id") != repo_id or plan.get("step") != step:
            raise RuntimeError(f"Existing source plan does not match {repo_id}")
        return plan
    info = retry(
        lambda: HfApi().model_info(repo_id, files_metadata=True),
        f"Resolve {repo_id}",
    )
    files = []
    for sibling in info.siblings:
        name = sibling.rfilename
        if name in SKIP_FILES:
            continue
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts:
            raise RuntimeError("Unsafe repository filename")
        lfs = sibling.lfs
        files.append({
            "path": name,
            "size": sibling.size,
            "published_sha256": lfs.sha256 if lfs else None,
            "git_blob_sha1": sibling.blob_id if not lfs else None,
        })
    plan = {
        "created_at": now(), "repo_id": repo_id, "revision": info.sha,
        "step": step, "target": str(target), "files": files,
    }
    save_json(plan_path, plan)
    return plan


def verify_file(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    actual_size = path.stat().st_size
    if actual_size != item["size"]:
        raise RuntimeError(f"Size mismatch at {path}; existing asset preserved")
    sha256 = hashlib.sha256()
    git_sha1 = hashlib.sha1()
    git_sha1.update(f"blob {actual_size}\0".encode())
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            sha256.update(block)
            git_sha1.update(block)
    digest = sha256.hexdigest()
    if item.get("published_sha256") and digest != item["published_sha256"]:
        raise RuntimeError(f"Published SHA256 mismatch at {path}; asset preserved")
    if item.get("git_blob_sha1") and git_sha1.hexdigest() != item["git_blob_sha1"]:
        raise RuntimeError(f"Git blob hash mismatch at {path}; asset preserved")
    return {
        **item, "actual_size": actual_size, "actual_sha256": digest,
        "published_hash_verified": True,
    }


def parallel_http_download(
    target: Path, plan: dict[str, Any], item: dict[str, Any], workers: int,
) -> None:
    """Resume verified-size byte ranges when one long HTTP stream stalls.

    This is only used for the large public model file. Independent 32 MiB
    ranges are written to a script-owned sparse partial file, then the complete
    published digest is checked before the asset becomes visible at its final
    filename. Small assets use the Hugging Face downloader.
    """
    cache = target / ".cache" / "omni_eval_ranges"
    cache.mkdir(parents=True, exist_ok=True)
    part = cache / (item["path"].replace("/", "_") + ".partial")
    state_path = cache / (part.name + ".json")
    chunk_size = 32 * 1024 * 1024
    size = item["size"]
    identity = {"repo_id": plan["repo_id"], "revision": plan["revision"],
                "path": item["path"], "size": size, "chunk_size": chunk_size}
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if any(state.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Range cache identity mismatch; cache preserved")
        if not part.exists() or part.stat().st_size != size:
            raise RuntimeError("Range cache file size mismatch; cache preserved")
    else:
        if part.exists():
            raise RuntimeError("Unrecognized partial file; preserved")
        with part.open("xb") as handle:
            handle.truncate(size)
        state = {**identity, "completed": []}
        save_json(state_path, state)
    completed = set(state["completed"])
    count = (size + chunk_size - 1) // chunk_size
    url = hf_hub_url(plan["repo_id"], item["path"], revision=plan["revision"])
    descriptor = os.open(part, os.O_WRONLY)

    def transfer(index: int) -> int:
        start, end = index * chunk_size, min((index + 1) * chunk_size, size) - 1

        def attempt() -> None:
            with requests.get(url, headers={"Range": f"bytes={start}-{end}"},
                              stream=True, timeout=(20, 30)) as response:
                if response.status_code != 206:
                    raise RuntimeError("Server did not honor byte range")
                if response.headers.get("Content-Range") != f"bytes {start}-{end}/{size}":
                    raise RuntimeError("Unexpected Content-Range")
                position = start
                for block in response.iter_content(1024 * 1024):
                    if position + len(block) > end + 1:
                        raise RuntimeError("Range body exceeds promised length")
                    view = memoryview(block)
                    while view:
                        written = os.pwrite(descriptor, view, position)
                        position += written
                        view = view[written:]
                if position != end + 1:
                    raise RuntimeError("Range response was truncated")

        retry(attempt, f"Range {index + 1}/{count} of step {plan['step']}")
        return index

    try:
        log(f"Parallel ranges step {plan['step']}: {len(completed)}/{count} complete, workers={workers}")
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(transfer, i) for i in range(count) if i not in completed]
            for future in as_completed(futures):
                completed.add(future.result())
                os.fsync(descriptor)
                state["completed"] = sorted(completed)
                save_json(state_path, state)
                if len(completed) % 8 == 0 or len(completed) == count:
                    log(f"Range progress step {plan['step']}: {len(completed)}/{count} ({100 * len(completed) / count:.1f}%)")
    finally:
        os.close(descriptor)
    log(f"Verifying complete range download step {plan['step']}")
    verify_file(part, item)
    final = target / item["path"]
    # Hard-link publication fails if another process created the target: no
    # existing user asset can be overwritten. Keep cache as a resumable hardlink.
    os.link(part, final)


def validate_model(target: Path) -> dict[str, Any]:
    config = json.loads((target / "config.json").read_text())
    if config.get("model_type") != "qwen3":
        raise RuntimeError("Expected Qwen3 model configuration")
    if config.get("hidden_size") != 4096 or config.get("num_hidden_layers") != 36:
        raise RuntimeError("Expected Qwen3 8B dimensions")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        json.loads((target / filename).read_text())
    index_path = target / "model.safetensors.index.json"
    expected: dict[str, str] | None = None
    if index_path.exists():
        expected = json.loads(index_path.read_text())["weight_map"]
        shard_names = sorted(set(expected.values()))
    else:
        shard_names = ["model.safetensors"]
    found: dict[str, str] = {}
    parameters = 0
    dtype_bytes = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
                   "I64": 8, "I32": 4, "I16": 2, "I8": 1,
                   "U8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1}
    for name in shard_names:
        path = target / name
        with path.open("rb") as handle:
            header_size = struct.unpack("<Q", handle.read(8))[0]
            if header_size > 100 * 1024 * 1024:
                raise RuntimeError(f"Implausible safetensors header at {path}")
            header = json.loads(handle.read(header_size))
        tensors = {k: v for k, v in header.items() if k != "__metadata__"}
        payload_size = path.stat().st_size - 8 - header_size
        intervals = []
        for key, tensor in tensors.items():
            if key in found:
                raise RuntimeError(f"Duplicate tensor {key}")
            found[key] = name
            count = 1
            for dimension in tensor["shape"]:
                count *= dimension
            parameters += count
            start, end = tensor["data_offsets"]
            if not 0 <= start <= end <= payload_size:
                raise RuntimeError(f"Invalid tensor offsets {key}")
            if end - start != count * dtype_bytes[tensor["dtype"]]:
                raise RuntimeError(f"Tensor payload length mismatch {key}")
            intervals.append((start, end))
        previous_end = 0
        for start, end in sorted(intervals):
            if start != previous_end:
                raise RuntimeError(f"Gaps or overlaps in {name}")
            previous_end = end
        if previous_end != payload_size:
            raise RuntimeError(f"Truncated or trailing safetensors payload {name}")
    if expected is not None and found != expected:
        raise RuntimeError("Safetensors index does not match shard keys")
    return {
        "model_type": config["model_type"], "architectures": config["architectures"],
        "hidden_size": config["hidden_size"], "num_hidden_layers": config["num_hidden_layers"],
        "shards": shard_names, "tensor_count": len(found), "parameter_count": parameters,
        "safetensors_headers_validated": True, "tokenizer_json_validated": True,
    }


def download_step(step: int, http_workers: int = 1) -> None:
    if step not in APPROVED_STEPS:
        raise ValueError("Only steps 256 and 224 are approved")
    repo_id = f"fiveflow/rq_8b_{step}"
    target = RUN_ROOT / f"global_step_{step}" / "hf_merged"
    target.mkdir(parents=True, exist_ok=True)
    plan = source_plan(target, repo_id, step)
    log(f"Pinned {repo_id}@{plan['revision']} -> {target}")
    verified = []
    # Make tokenizers/configs available first so prompt parity can be inspected
    # while large weights are downloading.
    for item in sorted(plan["files"], key=lambda entry: entry["size"] >= 1024 ** 3):
        path = target / item["path"]
        if path.exists():
            log(f"Validate existing step {step}: {item['path']}")
        else:
            log(f"Download step {step}: {item['path']} ({item['size']} bytes)")
            if item["size"] >= 1024 ** 3 and http_workers > 1:
                parallel_http_download(target, plan, item, http_workers)
            else:
                retry(
                    lambda: hf_hub_download(
                        repo_id=repo_id, filename=item["path"], revision=plan["revision"],
                        local_dir=target,
                    ),
                    f"Download {repo_id}/{item['path']}",
                )
        verified.append(verify_file(path, item))
        log(f"Verified step {step}: {item['path']}")
    validation = validate_model(target)
    save_json(target / "download_manifest.json", {
        **{k: v for k, v in plan.items() if k != "files"},
        "completed_at": now(), "files": verified, "validation": validation,
        "status": "complete",
    })
    log(f"COMPLETE step {step}: {validation['tensor_count']} tensors, all published hashes verified")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", nargs="+", type=int, choices=APPROVED_STEPS,
                        default=list(APPROVED_STEPS))
    parser.add_argument("--http-workers", type=int, choices=range(1, 17), default=1,
                        help="Use up to sixteen resumable HTTP ranges for large model assets")
    args = parser.parse_args()
    for step in args.steps:
        download_step(step, args.http_workers)


if __name__ == "__main__":
    main()
