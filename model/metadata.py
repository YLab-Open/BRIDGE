"""Unified reproducibility artifacts for local and commercial API inference."""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "authorization",
    "azure_endpoint",
    "endpoint",
    "key",
    "password",
    "secret",
    "token",
}


def result_metadata_paths(result_path: str | Path) -> dict[str, Path]:
    """Return standardized metadata and effective-config snapshot paths."""
    result_path = Path(result_path)
    if result_path.name.endswith(".result.json"):
        stem = result_path.name[: -len(".result.json")]
        prefix = result_path.with_name(f"{stem}.result")
    else:
        prefix = result_path
    return {
        "metadata": Path(f"{prefix}.metadata.json"),
        "config": Path(f"{prefix}.config.yaml"),
    }


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted, JSON/YAML-safe configuration."""
    if isinstance(value, Mapping):
        return {
            str(key): (
                "***"
                if str(key).lower() in _SECRET_KEYS
                else redact_secrets(child)
            )
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def file_manifest(
    path: str | Path | None,
    *,
    include_text: bool = False,
) -> dict[str, Any] | None:
    """Return size, hash, and optional text for a local file."""
    if not path:
        return None
    file_path = Path(path)
    manifest: dict[str, Any] = {
        "path": str(file_path),
        "exists": file_path.is_file(),
    }
    if not file_path.is_file():
        return manifest
    data = file_path.read_bytes()
    manifest.update(
        {
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "mtime": datetime.datetime.fromtimestamp(
                file_path.stat().st_mtime
            ).isoformat(),
        }
    )
    if include_text:
        try:
            manifest["text"] = data.decode("utf-8")
        except UnicodeDecodeError:
            manifest["text_decode_error"] = "not UTF-8 text"
    return manifest


def _git_manifest() -> dict[str, Any]:
    """Return best-effort repository state without storing the full diff."""

    def run_git(*args: str) -> str | None:
        try:
            process = subprocess.run(
                ["git", *args],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except OSError:
            return None
        return process.stdout.strip() if process.returncode == 0 else None

    status = run_git("status", "--short")
    return {
        "commit": run_git("rev-parse", "HEAD"),
        "branch": run_git("rev-parse", "--abbrev-ref", "HEAD"),
        "is_dirty": bool(status),
        "status_short": status,
    }


def _runtime_versions() -> dict[str, Any]:
    """Return Python and core package versions without importing heavy modules."""
    versions: dict[str, Any] = {"python": sys.version}
    for package in (
        "torch",
        "vllm",
        "transformers",
        "openai",
        "anthropic",
        "google-genai",
    ):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_config_snapshot(
    path: Path,
    *,
    effective_config: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            redact_secrets(effective_config),
            handle,
            allow_unicode=True,
            sort_keys=False,
        )


def write_result_metadata(
    *,
    result_path: str | Path,
    run_type: str,
    n_records: int,
    effective_config: Mapping[str, Any],
    source_files: Mapping[str, str | Path | None] | None = None,
    related_files: Mapping[str, str | Path | None] | None = None,
    experiment: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    """Write a config snapshot plus one common metadata manifest."""
    result_path = Path(result_path)
    paths = result_metadata_paths(result_path)
    _write_config_snapshot(paths["config"], effective_config=effective_config)

    source_manifests = {
        name: file_manifest(path)
        for name, path in (source_files or {}).items()
    }
    related_manifests = {
        "result": file_manifest(result_path),
        "config_snapshot": file_manifest(paths["config"]),
        **{
            name: file_manifest(path)
            for name, path in (related_files or {}).items()
        },
    }
    payload = {
        "schema_version": 3,
        "created_at": datetime.datetime.now().astimezone().isoformat(),
        "run_type": run_type,
        "n_records": int(n_records),
        "experiment": redact_secrets(experiment or {}),
        "effective_config": redact_secrets(effective_config),
        "run": {
            "cwd": os.getcwd(),
            "argv": sys.argv,
            "entrypoint": sys.argv[0] if sys.argv else None,
        },
        "files": {
            "source": source_manifests,
            "artifacts": related_manifests,
        },
        "git": _git_manifest(),
        "environment": _runtime_versions(),
        "extra": redact_secrets(extra or {}),
    }
    paths["metadata"].parent.mkdir(parents=True, exist_ok=True)
    with paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return paths
