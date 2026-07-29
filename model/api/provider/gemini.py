from __future__ import annotations

import time
from typing import Any

from model.api.prompts import gemini_contents_from_messages
from model.api.results import QueryResult


class GeminiProvider:
    """Google Gemini provider using single-query generate_content calls."""

    provider = "gemini"

    def __init__(
        self,
        model_name: str,
        model_cfg: dict[str, Any],
        run_cfg: dict[str, Any],
        api_settings: dict[str, Any],
    ):
        self.model_name = model_name
        self.model_cfg = model_cfg
        self.run_cfg = run_cfg
        self.api_settings = api_settings
        self.model_id = model_cfg.get("model_id") or model_cfg.get("deploy_name") or model_name

    def query_single(
        self,
        messages: list[dict[str, str]],
        cid: str,
        max_token_output: int,
    ) -> QueryResult:
        """Run one Gemini request."""
        t0 = time.time()
        try:
            from google import genai
            from google.genai import types
        except Exception as exc:
            return QueryResult(custom_id=cid, error=f"ImportError: {exc}")

        if not self.api_settings.get("api_key"):
            return QueryResult(custom_id=cid, error="GEMINI_API_KEY is missing")

        system, contents = gemini_contents_from_messages(messages)
        gen_cfg = dict(self.run_cfg.get("generation_config") or {})
        thinking_cfg = dict(self.run_cfg.get("thinking_config") or {})
        gen_cfg.setdefault("max_output_tokens", max_token_output)
        cfg_kwargs = {}
        if system:
            cfg_kwargs["system_instruction"] = system
        for key in ("temperature", "top_p", "top_k", "max_output_tokens"):
            if key in gen_cfg:
                cfg_kwargs[key] = gen_cfg[key]
        if thinking_cfg:
            try:
                thinking_kwargs = {}
                uses_level = False
                if "thinking_level" in thinking_cfg:
                    thinking_kwargs["thinking_level"] = thinking_cfg["thinking_level"]
                    uses_level = True
                if "thinking_budget" in thinking_cfg and not (
                    uses_level and "gemini-3" in self.model_id.lower()
                ):
                    thinking_kwargs["thinking_budget"] = thinking_cfg["thinking_budget"]
                if "include_thoughts" in thinking_cfg:
                    thinking_kwargs["include_thoughts"] = bool(thinking_cfg["include_thoughts"])
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(**thinking_kwargs)
            except Exception:
                pass

        try:
            client = genai.Client(api_key=str(self.api_settings["api_key"]))
            response = client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=types.GenerateContentConfig(**cfg_kwargs) if cfg_kwargs else None,
            )
            raw = gemini_raw_response(response)
            usage = getattr(response, "usage_metadata", None)
            return QueryResult(
                custom_id=cid,
                text=getattr(response, "text", "") or raw.get("text", ""),
                raw=raw,
                latency_ms=int((time.time() - t0) * 1000),
                tokens_in=int(getattr(usage, "prompt_token_count", 0) or 0),
                tokens_out=int(getattr(usage, "candidates_token_count", 0) or 0),
            )
        except Exception as exc:
            return QueryResult(
                custom_id=cid,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.time() - t0) * 1000),
            )


def gemini_raw_response(response: Any) -> dict[str, Any]:
    """Return a JSON-safe Gemini response trace."""
    if hasattr(response, "model_dump"):
        try:
            dumped = response.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    raw: dict[str, Any] = {"text": getattr(response, "text", "") or ""}
    candidates = []
    for candidate in getattr(response, "candidates", []) or []:
        if hasattr(candidate, "to_json_dict"):
            try:
                candidates.append(candidate.to_json_dict())
                continue
            except Exception:
                pass
        candidates.append(str(candidate))
    raw["candidates"] = candidates
    usage = getattr(response, "usage_metadata", None)
    if usage:
        raw["usage_metadata"] = {
            "prompt_token_count": getattr(usage, "prompt_token_count", 0),
            "candidates_token_count": getattr(usage, "candidates_token_count", 0),
            "total_token_count": getattr(usage, "total_token_count", 0),
        }
    return raw
