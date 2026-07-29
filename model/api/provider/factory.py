from __future__ import annotations

from typing import Any

from .azure import AzureProvider
from .claude import ClaudeProvider
from .gemini import GeminiProvider


def make_provider(
    provider: str,
    model_name: str,
    model_cfg: dict[str, Any],
    run_cfg: dict[str, Any],
    api_settings: dict[str, Any],
):
    """Create a commercial API provider object."""
    provider = provider.lower()
    if provider == "azure":
        return AzureProvider(model_name, model_cfg, run_cfg, api_settings)
    if provider == "claude":
        return ClaudeProvider(model_name, model_cfg, run_cfg, api_settings)
    if provider == "gemini":
        return GeminiProvider(model_name, model_cfg, run_cfg, api_settings)
    raise ValueError(f"Unsupported provider: {provider}")
