"""Helpers for local-only configuration files.

This module centralizes access to files that normally stay outside git, such as
API keys and local model-path mappings. Callers get validation and clear error
messages without printing secrets or machine-specific paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from util.runtime import load_yaml_config


DEFAULT_CONFIG_DIR = Path("configs")
DEFAULT_API_KEY_PATH = DEFAULT_CONFIG_DIR / "API.key.yaml"
DEFAULT_MODEL_MAPPING_PATH = DEFAULT_CONFIG_DIR / "dict_model_path.json"


def _expand_path(path_file: str | Path) -> Path:
    """Return a user-expanded Path while preserving relative paths."""
    return Path(path_file).expanduser()


def _resolve_with_legacy_fallback(path_file: str | Path, legacy_name: str) -> Path:
    """
    Resolve a local config path and support the pre-`configs/` root location.

    New code should use `configs/<name>`. If that file is missing but an old
    root-level local file exists, the old file is used to keep upgraded checkouts
    working until users move their private config files.
    """
    path = _expand_path(path_file)
    if path.exists():
        return path

    if path.name == legacy_name and path.parent.name == DEFAULT_CONFIG_DIR.name:
        legacy_path = path.parent.parent / legacy_name
        if legacy_path.exists():
            return legacy_path
    return path


def _require_mapping(value: Any, source: str) -> dict:
    """Validate that a loaded config object is a mapping."""
    if not isinstance(value, dict):
        raise ValueError(f"{source} must contain a mapping object.")
    return value


def _default_required_api_fields(provider: str) -> tuple[str, ...]:
    """Return required credential fields for a known provider block."""
    provider_upper = provider.upper()
    if provider_upper == "AZURE-OPENAI":
        return ("endpoint", "api_version", "api_key")
    if provider_upper == "OPENAI":
        return ("api_key",)
    return ()


def load_local_yaml(path_file: str | Path, required: bool = True) -> dict:
    """
    Load a local YAML config file with duplicate-key checking.

    Args:
        path_file: YAML file path, usually ignored by git.
        required: When false, a missing file returns an empty dict.

    Returns:
        Parsed YAML mapping.
    """
    path = _expand_path(path_file)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Local config file not found: {path}")
        return {}
    return _require_mapping(load_yaml_config(str(path)), str(path))


def load_api_key_config(
    path_file: str | Path = DEFAULT_API_KEY_PATH,
    provider: str = "AZURE-OPENAI",
    required_fields: tuple[str, ...] | None = None,
) -> dict:
    """
    Load and validate one provider credential block from API.key.yaml.

    The function never logs or prints credential values. `AZURE-OPENAI` expects
    `endpoint`, `api_version`, and `api_key`; `OPENAI` expects `api_key`.
    The default location is `configs/API.key.yaml`, with a fallback to the old
    root-level `API.key.yaml` if present.
    """
    path_file = _resolve_with_legacy_fallback(path_file, "API.key.yaml")
    config = load_local_yaml(path_file)
    if provider not in config:
        available = ", ".join(sorted(str(key) for key in config))
        raise KeyError(
            f"Provider {provider!r} not found in {path_file}. "
            f"Available provider blocks: {available or 'none'}"
        )

    provider_config = _require_mapping(config[provider], f"{path_file}:{provider}")
    required = (
        _default_required_api_fields(provider)
        if required_fields is None
        else tuple(required_fields)
    )
    missing = [
        field
        for field in required
        if field not in provider_config or str(provider_config[field]).strip() == ""
    ]
    if missing:
        raise ValueError(
            f"Provider {provider!r} in {path_file} is missing required field(s): {missing}"
        )
    return dict(provider_config)


def load_model_path_mapping(
    path_file: str | Path = DEFAULT_MODEL_MAPPING_PATH,
) -> dict:
    """
    Load a model-name to local-path mapping from dict_model_path.json.

    The mapping file is intentionally local-only because it usually contains
    machine-specific checkpoint paths. The default location is
    `configs/dict_model_path.json`, with a fallback to the old root-level
    `dict_model_path.json` if present.
    """
    path = _resolve_with_legacy_fallback(path_file, "dict_model_path.json")
    if not path.exists():
        raise FileNotFoundError(f"Model mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    _require_mapping(mapping, str(path))

    invalid = [
        key
        for key, value in mapping.items()
        if not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
    ]
    if invalid:
        raise ValueError(
            "Model mapping entries must be non-empty string keys and values. "
            f"Invalid model key(s): {invalid}"
        )
    return mapping


def resolve_model_path(
    model_name: str,
    path_file: str | Path = DEFAULT_MODEL_MAPPING_PATH,
    must_exist: bool = True,
) -> str:
    """
    Resolve one model name to its local checkpoint path.

    Args:
        model_name: Key in the model-path mapping file.
        path_file: JSON mapping path.
        must_exist: When true, fail if the resolved path does not exist.

    Returns:
        Local checkpoint path as a string.
    """
    mapping = load_model_path_mapping(path_file)
    if model_name not in mapping:
        raise KeyError(f"Model {model_name!r} not found in {path_file}")

    model_path = _expand_path(mapping[model_name])
    if must_exist and not model_path.exists():
        raise FileNotFoundError(
            f"Resolved model path does not exist for {model_name}: {model_path}"
        )
    return str(model_path)
