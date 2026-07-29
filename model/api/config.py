from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from model.api.common import deep_merge_dict
from model.metadata import redact_secrets
from util.runtime import parse_prompt_mode


def load_yaml(path: str | Path | None, required: bool = False) -> dict[str, Any]:
    """Load YAML as a mapping. Missing optional files become empty mappings."""
    if not path:
        if required:
            raise ValueError("YAML path is required.")
        return {}
    path = Path(path)
    if not path.exists():
        if required:
            raise FileNotFoundError(f"YAML file not found: {path}")
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return data


def parse_numeric(value: Any, default: int | None = None) -> Any:
    """Parse integer values and simple expressions such as `1024*32`."""
    if value is None:
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        expr = value.replace(" ", "")
        if expr.isdigit():
            return int(expr)
        if "*" in expr:
            parts = expr.split("*")
            if all(part.isdigit() for part in parts):
                out = 1
                for part in parts:
                    out *= int(part)
                return out
    return value


def redact(value: Any) -> Any:
    """Return a JSON-safe copy with credentials redacted."""
    return redact_secrets(value)


def provider_section(data: dict[str, Any], provider: str) -> dict[str, Any]:
    """Return a case-insensitive provider section from a config mapping."""
    provider_norm = provider.lower().replace("_", "-")
    aliases = {
        "azure": {"azure", "azure-openai", "azure_openai", "azURE-openai".lower()},
        "claude": {"claude", "anthropic"},
        "gemini": {"gemini", "google", "google-gemini"},
    }.get(provider_norm, {provider_norm})
    for key, value in data.items():
        if str(key).lower().replace("_", "-") in aliases:
            return value or {}
    return {}


def model_section(api_model_cfg: dict[str, Any], provider: str, model_name: str) -> dict[str, Any]:
    """Return provider/model capabilities from configs/API.model.yaml."""
    section = provider_section(api_model_cfg, provider)
    for key, value in section.items():
        if str(key).lower() == model_name.lower():
            return value or {}
    return {}


def require_model_section(
    api_model_cfg: dict[str, Any],
    provider: str,
    model_name: str,
    require_registered: bool = True,
) -> dict[str, Any]:
    """Return one model config and fail clearly when the registry is incomplete."""
    model_cfg = model_section(api_model_cfg, provider, model_name)
    if model_cfg or not require_registered:
        return model_cfg

    provider_cfg = provider_section(api_model_cfg, provider)
    if not provider_cfg:
        raise ValueError(
            f"Provider `{provider}` is missing from the API model registry. "
            "Check --api_model_config, usually configs/API.model.yaml."
        )
    available = ", ".join(sorted(str(key) for key in provider_cfg)) or "<none>"
    raise ValueError(
        f"Model `{model_name}` is not registered under provider `{provider}`. "
        f"Available models: {available}. Add it to configs/API.model.yaml or set "
        "`require_model_registered: false` in the run YAML for ad hoc tests."
    )


def resolve_api_settings(
    provider: str,
    run_cfg: dict[str, Any],
    api_key_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Merge API settings from the run YAML, key YAML, and environment."""
    settings = dict(run_cfg.get("api") or {})
    central = provider_section(api_key_cfg, provider)
    for key, value in central.items():
        if value not in (None, ""):
            settings[key] = value

    if provider == "azure":
        settings.setdefault("api_key_env", "AZURE_OPENAI_API_KEY")
        settings.setdefault("endpoint_env", "AZURE_OPENAI_ENDPOINT")
        settings.setdefault("api_version", "2025-04-01-preview")
        if not settings.get("endpoint"):
            settings["endpoint"] = os.getenv(str(settings["endpoint_env"]), "")
        if not settings.get("api_key"):
            settings["api_key"] = os.getenv(str(settings["api_key_env"]), "")
    elif provider == "claude":
        settings.setdefault("api_key_env", "ANTHROPIC_API_KEY")
        if not settings.get("api_key"):
            settings["api_key"] = os.getenv(str(settings["api_key_env"]), "")
    elif provider == "gemini":
        settings.setdefault("api_key_env", "GEMINI_API_KEY")
        if not settings.get("api_key"):
            settings["api_key"] = os.getenv(str(settings["api_key_env"]), "")

    return settings


def as_bool(value: Any, default: bool = False) -> bool:
    """Parse a config value as a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def models_from_config(run_cfg: dict[str, Any]) -> list[str]:
    """Return configured model names as a non-empty list."""
    models = run_cfg.get("models")
    if isinstance(models, str):
        models = [item.strip() for item in models.split(",") if item.strip()]
    if isinstance(models, list) and models:
        parsed = [str(model).strip() for model in models if str(model).strip()]
        if parsed:
            return parsed
    raise ValueError("`models` must be set in the config, or pass --model.")


def validate_api_run_config(run_cfg: dict[str, Any]) -> None:
    """Validate fields shared by CLI, Python, and notebook API runs."""
    required = ["tasks", "experiments", "mode", "dir_dataset", "dir_result"]
    missing = [key for key in required if key not in run_cfg]
    if missing:
        raise ValueError(f"Missing required API config field(s): {missing}")
    if not isinstance(run_cfg["tasks"], list) or not run_cfg["tasks"]:
        raise ValueError("tasks must be a non-empty list.")
    if not isinstance(run_cfg["experiments"], list) or not run_cfg["experiments"]:
        raise ValueError("experiments must be a non-empty list.")
    for idx, item in enumerate(run_cfg["experiments"]):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(f"experiments[{idx}] must be [prompt_mode, decoding_strategy, seed].")
        parse_prompt_mode(str(item[0]))
        if str(item[1]).lower() not in {"greedy", "sampling"}:
            raise ValueError(f"experiments[{idx}] has invalid decoding strategy: {item[1]}")
        int(item[2])


def _merge_if_present(target: dict[str, Any], source: dict[str, Any], key: str) -> bool:
    """Merge source[key] into target when present and return whether it existed."""
    value = source.get(key)
    if isinstance(value, dict):
        deep_merge_dict(target, value)
        return True
    return False


def effective_run_config(
    run_cfg: dict[str, Any],
    provider: str,
    decoding_strategy: str,
    seed: int,
) -> dict[str, Any]:
    """
    Return a per-experiment config where decoding_strategy/seed affect requests.

    For sampling, the YAML must include provider-specific sampling overrides so
    result filenames cannot silently describe the same request twice.
    """
    provider = provider.lower()
    decoding_strategy = decoding_strategy.lower()
    cfg = copy.deepcopy(run_cfg)
    cfg["_experiment"] = {
        "decoding_strategy": decoding_strategy,
        "seed": int(seed),
    }

    if provider == "azure":
        chat_params = dict(cfg.get("chat_params") or {})
        chat_params["seed"] = int(seed)
        if decoding_strategy == "greedy":
            chat_params["temperature"] = 0
            chat_params["top_p"] = 1
        else:
            if not _merge_if_present(chat_params, cfg, "sampling_chat_params"):
                raise ValueError(
                    "Azure sampling experiment requires `sampling_chat_params` in the run YAML."
                )
        cfg["chat_params"] = chat_params
        if decoding_strategy == "sampling":
            responses_params = dict(cfg.get("responses_params") or {})
            _merge_if_present(responses_params, cfg, "sampling_responses_params")
            cfg["responses_params"] = responses_params

    elif provider == "claude":
        params = dict(cfg.get("params") or {})
        params.setdefault("max_tokens", parse_numeric(cfg.get("max_token_output"), 1024 * 16))
        if decoding_strategy == "sampling":
            if not _merge_if_present(params, cfg, "sampling_params"):
                raise ValueError(
                    "Claude sampling experiment requires `sampling_params` in the run YAML."
                )
        cfg["params"] = params

    elif provider == "gemini":
        generation_config = dict(cfg.get("generation_config") or {})
        if decoding_strategy == "greedy":
            generation_config["temperature"] = 0
            generation_config["top_p"] = 1
        else:
            if not _merge_if_present(generation_config, cfg, "sampling_generation_config"):
                raise ValueError(
                    "Gemini sampling experiment requires `sampling_generation_config` in the run YAML."
                )
        cfg["generation_config"] = generation_config

    else:
        raise ValueError(f"Unsupported provider: {provider}")
    return cfg
