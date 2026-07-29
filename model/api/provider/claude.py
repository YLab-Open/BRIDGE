from __future__ import annotations

import time
from typing import Any

from model.api.common import safe_model_dump
from model.api.prompts import split_system_messages
from model.api.results import QueryResult


class ClaudeProvider:
    """Anthropic Claude provider using single-query Messages calls."""

    provider = "claude"

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
        """Run one Claude request."""
        t0 = time.time()
        try:
            import anthropic
        except Exception as exc:
            return QueryResult(custom_id=cid, error=f"ImportError: {exc}")

        if not self.api_settings.get("api_key"):
            return QueryResult(custom_id=cid, error="ANTHROPIC_API_KEY is missing")

        system, dialog = split_system_messages(messages)
        params = dict(self.run_cfg.get("params") or {})
        params.setdefault("max_tokens", max_token_output)
        timeout = float(self.api_settings.get("timeout", 900) or 900)
        max_retries = int(self.api_settings.get("sdk_max_retries", 2) or 0)
        stream = bool(self.api_settings.get("stream", True))

        try:
            client = anthropic.Anthropic(
                api_key=str(self.api_settings["api_key"]),
                timeout=timeout,
                max_retries=max_retries,
            )
            kwargs = {"model": self.model_id, "messages": dialog, **params}
            if system:
                kwargs["system"] = system
            if stream:
                text, raw, tokens_in, tokens_out = anthropic_stream_message(client, kwargs)
            else:
                response = client.messages.create(**kwargs)
                raw = safe_model_dump(response)
                text = anthropic_text_from_raw(raw)
                usage = raw.get("usage") or {}
                tokens_in = int(usage.get("input_tokens") or 0)
                tokens_out = int(usage.get("output_tokens") or 0)
            return QueryResult(
                custom_id=cid,
                text=text,
                raw=raw,
                latency_ms=int((time.time() - t0) * 1000),
                tokens_in=tokens_in,
                tokens_out=tokens_out,
            )
        except Exception as exc:
            return QueryResult(
                custom_id=cid,
                error=f"{type(exc).__name__}: {exc}",
                latency_ms=int((time.time() - t0) * 1000),
            )


def anthropic_text_from_raw(raw: dict[str, Any]) -> str:
    """Extract text from a non-streaming Anthropic response dict."""
    parts = []
    for block in raw.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(block.get("text", ""))
    return "".join(parts).strip()


def anthropic_stream_message(client: Any, kwargs: dict[str, Any]) -> tuple[str, dict[str, Any], int, int]:
    """Run an Anthropic Messages streaming call and retain all stream events."""
    text_parts: list[str] = []
    raw_events: list[dict[str, Any]] = []
    tokens_in = 0
    tokens_out = 0
    stream = client.messages.create(stream=True, **kwargs)
    for event in stream:
        event_dict = safe_model_dump(event)
        if not event_dict:
            event_dict = {"type": getattr(event, "type", "")}
        raw_events.append(event_dict)
        event_type = getattr(event, "type", "") or event_dict.get("type", "")
        if event_type == "message_start":
            message = getattr(event, "message", None)
            usage = getattr(message, "usage", None)
            if usage:
                tokens_in = int(getattr(usage, "input_tokens", tokens_in) or 0)
                tokens_out = int(getattr(usage, "output_tokens", tokens_out) or 0)
        elif event_type == "content_block_delta":
            delta = getattr(event, "delta", None)
            if getattr(delta, "type", "") == "text_delta":
                text_parts.append(getattr(delta, "text", "") or "")
        elif event_type == "message_delta":
            usage = getattr(event, "usage", None)
            if usage:
                tokens_out = int(getattr(usage, "output_tokens", tokens_out) or 0)
    return "".join(text_parts).strip(), {"streamed": True, "events": raw_events}, tokens_in, tokens_out
