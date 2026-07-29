"""Shared Azure OpenAI request, response, and cost helpers."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from model.api.common import deep_merge_dict

try:
    import tiktoken
except ImportError:  # pragma: no cover - only for minimal test environments.
    tiktoken = None


MODEL_PRICE_PER_1M = {
    "gpt-5.4": (2.5, 15.0),
    "gpt-5.2": (1.75, 14.0),
    "gpt-5.1": (1.25, 10.0),
    "gpt-5": (1.25, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
    "o1": (15.0, 60.0),
    "o3-mini": (1.1, 4.4),
    "o3": (2.0, 8.0),
    "o4-mini": (1.1, 4.4),
}


def is_reasoning_model(model_name: str) -> bool:
    """
    Return whether a model name belongs to a reasoning-model family.

    Args:
        model_name: Azure deployment or model name.
    """
    if not model_name:
        return False
    name = model_name.lower()
    if name.startswith("gpt-5") or "/gpt-5" in name or "-gpt-5" in name:
        return True
    if re.match(r"^(o1|o3|o4)([-._]|$)", name):
        return True
    markers = ("-o1-", "-o3-", "-o4-", "o1-", "o3-", "o4-", "-o1", "-o3", "-o4")
    return any(marker in name for marker in markers)


def supports_chat_reasoning_effort(model_name: str) -> bool:
    """Return whether `reasoning_effort` can be sent to Chat Completions."""
    return is_reasoning_model(model_name) and "o1-mini" not in model_name.lower()


def default_query_mode_for_model(model_name: str) -> str:
    """Return `response` for reasoning models and `chat` for conventional models."""
    return "response" if is_reasoning_model(model_name) else "chat"


def apply_output_token_key(
    config: Dict[str, Any],
    model_name: str,
    mode: str,
    max_token_output: int,
) -> Dict[str, Any]:
    """
    Normalize Azure output-token keys for Chat Completions or Responses API.

    Args:
        config: Request config dictionary.
        model_name: Azure deployment/model name.
        mode: `chat` or `response`.
        max_token_output: Unified output-token cap.

    Returns:
        A copied config containing only the Azure-compatible output-token key.
    """
    normalized = dict(config)
    for key in ("max_tokens", "max_completion_tokens", "max_output_tokens"):
        normalized.pop(key, None)

    if mode == "response" and is_reasoning_model(model_name):
        normalized["max_output_tokens"] = max_token_output
    else:
        normalized["max_completion_tokens"] = max_token_output
    return normalized


def resolve_tokenizer_model(model_name: str) -> str:
    """Return a tiktoken model name suitable for token counting."""
    name = model_name.lower()
    if "gpt-5" in name:
        return "gpt-5"
    if "gpt-4o" in name:
        return "gpt-4o"
    if "gpt-35-turbo" in name:
        return "gpt-35-turbo"
    return "gpt-4o"


def parse_custom_id(custom_id: str) -> Optional[Tuple[str, str, str, str, str]]:
    """
    Parse a provider custom id.

    Expected format is `task|model_name|prompt_mode|split|sample_id`. The final
    sample id may itself contain separators before the last split point.
    """
    if not custom_id or custom_id.count("|") < 4:
        return None
    prefix, sample_id = custom_id.rsplit("|", 1)
    task, model_name, prompt_mode, split = prefix.split("|", 3)
    return task, model_name, prompt_mode, split, sample_id


def build_messages(
    input_data: Any,
    examples: Optional[Sequence[Dict[str, str]]] = None,
) -> List[Dict[str, str]]:
    """
    Build chat-format messages from a raw input payload and optional examples.

    Args:
        input_data: One of:
            - dict with `instruction` and `input`
            - existing list of chat messages
            - plain text prompt
        examples: Optional examples with `input` and `output` fields.

    Returns:
        Messages ordered as system instruction, examples, final user input.
    """
    example_messages: List[Dict[str, str]] = []
    for example in examples or []:
        example_messages.append({"role": "user", "content": example["input"]})
        example_messages.append({"role": "assistant", "content": example["output"]})

    if isinstance(input_data, dict):
        if "instruction" not in input_data or "input" not in input_data:
            raise KeyError(
                "When `input_data` is dict, keys `instruction` and `input` are required."
            )
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": input_data["instruction"]},
            *example_messages,
            {"role": "user", "content": input_data["input"]},
        ]
    elif isinstance(input_data, list):
        messages = list(input_data)
        insert_idx = 1 if messages and messages[0].get("role") == "system" else 0
        messages[insert_idx:insert_idx] = example_messages
    else:
        messages = [*example_messages, {"role": "user", "content": str(input_data)}]

    return messages


def default_chat_config(
    model_name: str,
    max_completion_tokens: int = 16384,
    seed: int = 42,
) -> Dict[str, Any]:
    """Return default Chat Completions decoding config for Azure batch/single calls."""
    if is_reasoning_model(model_name):
        config: Dict[str, Any] = {
            "model": model_name,
            "seed": seed,
            "max_completion_tokens": max_completion_tokens,
        }
        if supports_chat_reasoning_effort(model_name):
            config["reasoning_effort"] = "medium"
        return config

    return {
        "model": model_name,
        "seed": seed,
        "n": 1,
        "temperature": 0,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
        "max_completion_tokens": max_completion_tokens,
    }


def default_response_config(
    model_name: str,
    max_output_tokens: int = 16384,
) -> Dict[str, Any]:
    """Return default Responses API config for Azure reasoning models."""
    return {
        "model": model_name,
        "max_output_tokens": max_output_tokens,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"verbosity": "medium"},
    }


def build_chat_completion_batch_body(
    model_name: str,
    messages: List[Dict[str, str]],
    max_token_output: int = 8192,
    dict_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a request body for Azure batch `POST /chat/completions`."""
    if is_reasoning_model(model_name):
        body: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "seed": 42,
        }
        if supports_chat_reasoning_effort(model_name):
            body["reasoning_effort"] = "medium"
    else:
        body = {
            "model": model_name,
            "messages": messages,
            "seed": 42,
        }

    deep_merge_dict(body, dict_config)
    return apply_output_token_key(
        config=body,
        model_name=model_name,
        mode="chat",
        max_token_output=max_token_output,
    )


def build_response_batch_body(
    model_name: str,
    messages: List[Dict[str, str]],
    max_token_output: int = 8192,
    dict_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a request body for Azure batch `POST /responses`."""
    body: Dict[str, Any] = {
        "model": model_name,
        "input": messages,
        "max_output_tokens": max_token_output,
        "reasoning": {"effort": "medium", "summary": "auto"},
        "text": {"verbosity": "medium"},
    }
    deep_merge_dict(body, dict_config)
    return apply_output_token_key(
        config=body,
        model_name=model_name,
        mode="response",
        max_token_output=max_token_output,
    )


def extract_text_from_responses_api_output(body: Dict[str, Any]) -> str:
    """Extract final assistant text from a Responses API response body."""
    if not body:
        print(f"Error extracting text from responses api output: {body}")
        return ""
    text_direct = body.get("output_text")
    if isinstance(text_direct, str) and text_direct.strip():
        return text_direct

    output_items = body.get("output") or []
    parts: List[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "message":
            continue
        for block in item.get("content") or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") in {"output_text", "text"}:
                text = block.get("text")
                if isinstance(text, str) and text:
                    parts.append(text)
    if parts:
        return "".join(parts)

    print(f"Error extracting text from responses api output: {body}")
    return ""


def extract_text_from_chat_completion_body(body: Dict[str, Any]) -> str:
    """Extract assistant text from a Chat Completions response body."""
    try:
        content = body["choices"][0]["message"]["content"]
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str) and text:
                        parts.append(text)
            return "".join(parts)
        return str(content)
    except (KeyError, IndexError, TypeError):
        print(f"Error extracting text from chat completion body: {body}")
        return ""


def extract_summary_from_responses_api_output(body: Dict[str, Any]) -> str:
    """Extract reasoning summary text from a Responses API response body."""
    if not body:
        return ""
    output_items = body.get("output") or []
    summary_parts: List[str] = []
    for item in output_items:
        if not isinstance(item, dict) or item.get("type") != "reasoning":
            continue
        for block in item.get("summary") or []:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str) and text:
                    summary_parts.append(text)
    return "\n\n".join(summary_parts)


def get_azure_price(model_name: str, flag_batch: bool = False) -> Tuple[float, float]:
    """
    Return Azure input/output price in USD per 1M tokens.

    Args:
        model_name: Azure deployment/model name. `-batch` suffix is ignored.
        flag_batch: If true, apply the standard batch discount.
    """
    normalized_name = model_name.lower().replace("-batch", "")

    if normalized_name in MODEL_PRICE_PER_1M:
        cost_input_1m, cost_output_1m = MODEL_PRICE_PER_1M[normalized_name]
    else:
        matched_key = None
        for key in sorted(MODEL_PRICE_PER_1M.keys(), key=len, reverse=True):
            if normalized_name.startswith(key):
                matched_key = key
                break
        if not matched_key:
            raise KeyError(
                f"Price not found for model `{model_name}`. "
                f"Known keys: {sorted(MODEL_PRICE_PER_1M.keys())}"
            )
        cost_input_1m, cost_output_1m = MODEL_PRICE_PER_1M[matched_key]

    if flag_batch:
        return cost_input_1m / 2.0, cost_output_1m / 2.0
    return cost_input_1m, cost_output_1m


def cost_calculation_token(
    num_token_input_all: int,
    num_token_output_all: int,
    model_name: str,
    flag_batch: bool = False,
) -> float:
    """Calculate total Azure cost from aggregate input/output token counts."""
    cost_input_1m, cost_output_1m = get_azure_price(model_name, flag_batch)
    cost_input = num_token_input_all * cost_input_1m / 10**6
    cost_output = num_token_output_all * cost_output_1m / 10**6
    return cost_input + cost_output


def cost_calculation_word(
    list_input: Sequence[str],
    list_output: Sequence[str],
    flag_batch: bool = False,
    model_name: str = "gpt-4",
) -> float:
    """Estimate Azure cost by tokenizing concrete input and output strings."""
    if tiktoken is None:
        raise ImportError("tiktoken is required for word-based Azure cost estimation.")
    tokenizer = tiktoken.encoding_for_model(resolve_tokenizer_model(model_name))
    token_input_each = tokenizer.encode_batch(list_input)
    token_output_each = tokenizer.encode_batch(list_output)
    num_token_input_all = sum(len(tokens) for tokens in token_input_each)
    num_token_output_all = sum(len(tokens) for tokens in token_output_each)
    return cost_calculation_token(
        num_token_input_all=num_token_input_all,
        num_token_output_all=num_token_output_all,
        model_name=model_name,
        flag_batch=flag_batch,
    )


def cost_estimation_word(
    list_input: Sequence[str],
    word_output_each: int,
    num_sample: int,
    model_name: str,
    flag_batch: bool = False,
    ratio_word_token: int = 2,
) -> float:
    """Estimate Azure cost from input strings and expected output word count."""
    if tiktoken is None:
        raise ImportError("tiktoken is required for word-based Azure cost estimation.")
    tokenizer = tiktoken.encoding_for_model(resolve_tokenizer_model(model_name))
    token_input_each = tokenizer.encode_batch(list_input)
    num_token_input_all = sum(len(tokens) for tokens in token_input_each)
    num_token_output_all = word_output_each * ratio_word_token * num_sample
    return cost_calculation_token(
        num_token_input_all=num_token_input_all,
        num_token_output_all=num_token_output_all,
        model_name=model_name,
        flag_batch=flag_batch,
    )
