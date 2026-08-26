"""Freeze the effective method contract before a structural-inspiration run.

The run config uses ``extends`` and the prompts live outside YAML.  Without a
snapshot, editing the base config or one prompt and then using ``resume_mode:
auto`` silently combines two methods under one checkpoint directory.  This
module writes the fully resolved, secret-redacted contract on first launch and
refuses a mismatched resume before Ray or any GPU worker starts.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


SCHEMA_VERSION = 1
CONTRACT_DIR = "run_contract"
MANIFEST_FILE = "run_manifest.json"

_SECRET_KEYS = {
    "key",
    "token",
    "api_key",
    "openai_api_key",
    "wandb_api_key",
    "hf_token",
    "access_token",
    "password",
    "secret",
    "client_secret",
}


class RunContractMismatch(RuntimeError):
    """The requested resume does not match the method already in its directory."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bundle_hash(files: dict[str, str]) -> str:
    payload = json.dumps(
        files, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(payload)


def _config_chain(path: Path) -> list[Path]:
    """Return base-to-child YAML paths without resolving any interpolation."""
    path = path.expanduser().resolve()
    chain: list[Path] = []
    seen: set[Path] = set()
    while True:
        if path in seen:
            raise ValueError(f"cyclic config extends chain at {path}")
        seen.add(path)
        chain.append(path)
        raw = OmegaConf.load(path)
        parent = raw.get("extends") if hasattr(raw, "get") else None
        if not parent:
            break
        parent_path = Path(str(parent)).expanduser()
        path = (
            parent_path if parent_path.is_absolute() else path.parent / parent_path
        ).resolve()
    return list(reversed(chain))


def _tree_files(root: Path, pattern: str) -> dict[str, str]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): _sha256_file(path)
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    }


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            name = str(key).lower()
            sensitive = name in _SECRET_KEYS or name.endswith(
                (
                    "_api_key",
                    "_access_token",
                    "_auth_token",
                    "_password",
                    "_secret",
                )
            )
            out[str(key)] = "<redacted>" if sensitive else _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _git_head(project_root: Path) -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            or None
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _snapshot_tree(source: Path, destination: Path, pattern: str) -> None:
    for path in sorted(source.rglob(pattern)):
        if path.is_file():
            relative = path.relative_to(source)
            _atomic_write(
                destination / relative,
                path.read_text(encoding="utf-8"),
            )


def _implementation_hashes(project_root: Path) -> dict[str, str]:
    """Hash the live training implementation, including the installed-patch source."""
    paths = list((project_root / "src" / "rq_evolve").rglob("*.py"))
    paths.extend((project_root / "patches").rglob("*.py"))
    paths.extend(
        path
        for path in (
            project_root / "scripts" / "train_with_verl.py",
            project_root / "scripts" / "preflight_check.py",
            project_root / "launch_4b_train.sh",
        )
        if path.is_file()
    )
    return {
        path.relative_to(project_root).as_posix(): _sha256_file(path)
        for path in sorted(set(paths))
        if path.is_file()
    }


def freeze_evolution_run_contract(
    *,
    contract_name: str,
    config_path: str | Path,
    resolved_config,
    output_dir: str | Path,
    project_root: str | Path,
    prompt_dir: str | Path,
) -> Path:
    """Write or validate the immutable run contract; return its manifest path."""
    project_root = Path(project_root).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    prompt_dir = Path(prompt_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser()
    if not output_dir.is_absolute():
        # Matches verl's relative default_local_dir behaviour.
        output_dir = (Path.cwd() / output_dir).resolve()

    config_chain = _config_chain(config_path)
    config_files = {str(path): _sha256_file(path) for path in config_chain}
    resolved_container = OmegaConf.to_container(resolved_config, resolve=True)
    resolved_json = json.dumps(
        resolved_container,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    resolved_sha = _sha256_bytes(resolved_json.encode("utf-8"))

    prompt_files = _tree_files(prompt_dir, "*.txt")
    evolution = (resolved_container or {}).get("evolution", {})
    seed_dir = Path(str(evolution.get("seed_programs_dir", "seed_programs")))
    if not seed_dir.is_absolute():
        seed_dir = project_root / seed_dir
    seed_dir = seed_dir.resolve()
    seed_files = _tree_files(seed_dir, "*.py")
    implementation_files = _implementation_hashes(project_root)

    contract_components = {
        "contract_name": str(contract_name),
        "config_chain": config_files,
        "resolved_config_sha256": resolved_sha,
        "prompt_bundle_sha256": _bundle_hash(prompt_files),
        "seed_bundle_sha256": _bundle_hash(seed_files),
        "implementation_bundle_sha256": _bundle_hash(implementation_files),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "contract_name": str(contract_name),
        "contract_sha256": _bundle_hash(contract_components),
        "source_config": str(config_path),
        **contract_components,
        "prompt_files": prompt_files,
        "seed_files": seed_files,
        "implementation_files": implementation_files,
        "git_head": _git_head(project_root),
    }

    contract_dir = output_dir / CONTRACT_DIR
    manifest_path = contract_dir / MANIFEST_FILE
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("contract_sha256") != manifest["contract_sha256"]:
            changed = [
                key
                for key in contract_components
                if existing.get(key) != manifest.get(key)
            ]
            raise RunContractMismatch(
                "evolution resume contract changed in "
                f"{', '.join(changed) or 'unknown inputs'}; existing run: "
                f"{existing.get('contract_sha256')}, requested: "
                f"{manifest['contract_sha256']}. Use a new trainer.default_local_dir "
                "instead of mixing methods in one checkpoint directory."
            )
        return manifest_path

    # A checkpoint without a contract cannot be proven compatible. Empty/new
    # directories and log-only directories are safe to initialize.
    has_training_state = (
        any(output_dir.glob("global_step_*"))
        or (output_dir / "rq_archive" / "archive.json").exists()
    )
    if has_training_state:
        raise RunContractMismatch(
            f"{output_dir} contains training state but no {manifest_path}; "
            "use a fresh output directory for this experiment"
        )

    redacted = _redact(resolved_container)
    _atomic_write(
        contract_dir / "config_resolved.yaml",
        OmegaConf.to_yaml(OmegaConf.create(redacted), resolve=False),
    )
    for index, path in enumerate(config_chain):
        source_payload = OmegaConf.to_container(OmegaConf.load(path), resolve=False)
        _atomic_write(
            contract_dir / "config_sources" / f"{index:02d}_{path.name}",
            OmegaConf.to_yaml(OmegaConf.create(_redact(source_payload)), resolve=False),
        )
    _snapshot_tree(prompt_dir, contract_dir / "prompt_templates", "*.txt")
    _snapshot_tree(seed_dir, contract_dir / "seed_programs", "*.py")
    _atomic_write(
        manifest_path,
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )
    return manifest_path
