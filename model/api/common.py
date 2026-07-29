"""Small provider-neutral helpers shared by commercial API implementations."""

from __future__ import annotations

from typing import Any


def deep_merge_dict(
    base: dict[str, Any],
    override: dict[str, Any] | None,
) -> dict[str, Any]:
    """Recursively merge an optional mapping into a mutable base mapping."""
    if not override:
        return base
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            deep_merge_dict(base[key], value)
        else:
            base[key] = value
    return base


def safe_model_dump(response: Any) -> dict[str, Any]:
    """Return a dict from an SDK response object or an existing mapping."""
    if hasattr(response, "model_dump"):
        dumped = response.model_dump()
        if isinstance(dumped, dict):
            return dumped
    return response if isinstance(response, dict) else {}
