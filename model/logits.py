from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from evaluation.probability import (
    decode_token_cached,
    encode_without_special_tokens,
    extract_label_from_text,
    get_label_token_ids,
    score_label_groups_from_logits,
    score_vllm_probe_output,
    top_token_rows_from_logits,
)

DEFAULT_LOGITS_CONFIG: dict[str, Any] = {
    "enabled": False,
    "vllm": {
        "max_logprobs": -1,
        "logprobs_mode": "raw_logits",
    },
    "generation_trace": {
        "enabled": True,
        "top_k_logprobs_for_all_output_tokens": 0,
    },
    "label_probability": {
        "enabled": False,
        "mode": "generation_logprobs",
        "generation_logprobs_top_k": 5000,
        "fallback_to_exact_probe": True,
        "probe_batch_size": 64,
        "top_k_logits": 1000,
        "save_top_logits": True,
        "label_token_prefixes": "auto",
        "extra_label_token_prefixes": [],
        "scan_decoded_vocab": True,
        "fail_on_error": True,
        "probe_prompt_mode": "token_ids",
        "model_label_token_id_overrides": {},
    },
    "tasks": {},
}


def deep_update(base: dict[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    """Recursively merge a config override into a copy of base."""
    out = copy.deepcopy(base)
    if not override:
        return out
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def normalize_logits_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Normalize legacy aliases before merging with defaults."""
    normalized = copy.deepcopy(raw)
    trace_cfg = normalized.get("generation_trace")
    if isinstance(trace_cfg, dict):
        if (
            "top_k_logprobs" in trace_cfg
            and "top_k_logprobs_for_all_output_tokens" not in trace_cfg
        ):
            trace_cfg["top_k_logprobs_for_all_output_tokens"] = trace_cfg[
                "top_k_logprobs"
            ]
    return normalized


def load_logits_config(path: str | Path | None) -> dict[str, Any]:
    """Load the optional logits YAML config."""
    if not path:
        return copy.deepcopy(DEFAULT_LOGITS_CONFIG)
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Logits config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Logits config must contain a YAML mapping: {path}")
    raw = normalize_logits_config(raw)
    cfg = deep_update(DEFAULT_LOGITS_CONFIG, raw)
    cfg["config_path"] = str(path)
    return cfg


def configure_logits_from_args(args) -> dict[str, Any]:
    """Attach normalized logits config to args before model initialization."""
    path = getattr(args, "logits_config", None) or getattr(args, "logit_config", None)
    cfg = load_logits_config(path)
    args.logits = cfg
    args.logits_enabled = bool(cfg.get("enabled", False))
    if args.logits_enabled:
        vllm_cfg = cfg.get("vllm") or {}
        args.vllm_max_logprobs = int(vllm_cfg.get("max_logprobs", -1))
        args.vllm_logprobs_mode = str(vllm_cfg.get("logprobs_mode", "raw_logits"))
    return cfg


def logits_enabled(args) -> bool:
    return bool(getattr(args, "logits_enabled", False))


def get_trace_top_k_for_all_output_tokens(trace_cfg: Mapping[str, Any] | None) -> int:
    """Return generation-trace top-K for every generated output token."""
    trace_cfg = trace_cfg or {}
    value = trace_cfg.get("top_k_logprobs_for_all_output_tokens")
    if value is None:
        # Backward-compatible read for older local configs; new docs use the
        # explicit all-output-token name.
        value = trace_cfg.get("top_k_logprobs", 0)
    return int(value or 0)


def get_generation_trace_top_k(args) -> int:
    if not logits_enabled(args):
        return 0
    cfg = getattr(args, "logits", {}) or {}
    trace_cfg = cfg.get("generation_trace") or {}
    lp_cfg = cfg.get("label_probability") or {}
    if not trace_cfg.get("enabled", True):
        trace_top_k = 0
    else:
        trace_top_k = get_trace_top_k_for_all_output_tokens(trace_cfg)

    mode = str(lp_cfg.get("mode", "exact_output_token_probe"))
    if lp_cfg.get("enabled", False) and mode == "generation_logprobs":
        label_top_k = int(
            lp_cfg.get("generation_logprobs_top_k")
            or lp_cfg.get("top_k_logits", 0)
            or 0
        )
        if label_top_k == -1 or trace_top_k == -1:
            return -1
        return max(trace_top_k, label_top_k)
    return trace_top_k


def get_task_target_config(args, task_name: str | None = None) -> dict[str, Any] | None:
    """Return task-specific label-probability config, if present."""
    if not logits_enabled(args):
        return None
    cfg = getattr(args, "logits", {}) or {}
    task_map = cfg.get("tasks") or {}
    task_name = str(task_name or getattr(args, "task_name", ""))
    target = task_map.get(task_name)
    if target is None:
        return None
    if not isinstance(target, dict):
        raise ValueError(f"logits.tasks.{task_name} must be a mapping.")
    return target


def logit_output_paths(path_file_result: str | Path) -> dict[str, Path]:
    """Return default sidecar paths for one result JSON path."""
    result_path = Path(path_file_result)
    name = result_path.name
    if name.endswith(".result.json"):
        stem = name[: -len(".result.json")]
        logit_base = result_path.with_name(f"{stem}.result.logit")
        metadata_path = result_path.with_name(f"{stem}.result.metadata.json")
    else:
        logit_base = result_path.with_name(f"{name}.logit")
        metadata_path = result_path.with_name(f"{name}.metadata.json")
    return {
        "logit_json": Path(f"{logit_base}.json"),
        "top_logits": Path(f"{logit_base}.top_logits.jsonl.gz"),
        "metadata": metadata_path,
    }


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def _repair_truncated_gzip_jsonl(path: Path) -> int:
    """Atomically retain complete JSONL rows from an interrupted gzip stream."""
    temporary_path = path.with_name(f".{path.name}.repair-{os.getpid()}")
    line_count = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8") as source, gzip.open(
            temporary_path, "wt", encoding="utf-8"
        ) as destination:
            while True:
                try:
                    line = source.readline()
                except EOFError:
                    break
                if not line:
                    break
                if not line.endswith("\n"):
                    break
                if not line.strip():
                    continue
                json.loads(line)
                destination.write(line)
                line_count += 1
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    return line_count


class JsonlSidecarWriter:
    """Write compact JSONL or JSONL.GZ sidecars and return stable line refs."""

    def __init__(self, path: str | Path | None, *, append: bool = False):
        self.path = Path(path) if path else None
        self.append = bool(append)
        self.line = 0
        self._fh = None

    def __enter__(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            if self.append and self.path.exists():
                opener = gzip.open if str(self.path).endswith(".gz") else open
                try:
                    with opener(self.path, "rt", encoding="utf-8") as handle:
                        self.line = sum(1 for line in handle if line.strip())
                except EOFError:
                    if not str(self.path).endswith(".gz"):
                        raise
                    self.line = _repair_truncated_gzip_jsonl(self.path)
            if str(self.path).endswith(".gz"):
                mode = "at" if self.append else "wt"
                self._fh = gzip.open(self.path, mode, encoding="utf-8")
            else:
                mode = "a" if self.append else "w"
                self._fh = self.path.open(mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh is not None:
            self._fh.close()
        self._fh = None

    def write(self, payload: Mapping[str, Any], top_k: int | None = None) -> dict[str, Any] | None:
        if self._fh is None or self.path is None:
            return None
        self.line += 1
        row = {"line": self.line, **dict(payload)}
        self._fh.write(json.dumps(row, ensure_ascii=False, default=_json_default) + "\n")
        self._fh.flush()
        return {
            "path": str(self.path),
            "format": "jsonl.gz" if str(self.path).endswith(".gz") else "jsonl",
            "line": self.line,
            "top_k": int(top_k) if top_k is not None else None,
        }


def token_spans_from_token_ids(
    tokenizer,
    token_ids: Sequence[int],
    text: str | None = None,
    token_text_cache: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Return decoded-token spans from the actual generated token ids."""
    spans: list[dict[str, Any]] = []
    cursor = 0
    parts = []
    for idx, token_id in enumerate(token_ids):
        token_text = decode_token_cached(tokenizer, int(token_id), cache=token_text_cache)
        parts.append(token_text)
        start = cursor
        end = start + len(token_text)
        spans.append(
            {
                "index": idx,
                "token_id": int(token_id),
                "token_text": token_text,
                "start": start,
                "end": end,
            }
        )
        cursor = end
    decoded = "".join(parts)
    if text is not None and decoded != text:
        # Some tokenizers treat special tokens differently in one-token decode.
        # Fall back to cumulative decode when it exactly matches output text.
        decoded_full = tokenizer.decode([int(token_id) for token_id in token_ids])
        if decoded_full != text:
            return spans
        spans = []
        previous = ""
        for idx, token_id in enumerate(token_ids):
            current = tokenizer.decode([int(item) for item in token_ids[: idx + 1]])
            token_text = current[len(previous) :]
            spans.append(
                {
                    "index": idx,
                    "token_id": int(token_id),
                    "token_text": token_text,
                    "start": len(previous),
                    "end": len(current),
                }
            )
            previous = current
    return spans


def infer_label_target_from_generated_tokens(
    text: str,
    token_ids: Sequence[int],
    tokenizer,
    label_values: Sequence[str],
    answer_prefix: str | None = None,
    token_text_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Find the actual generated token that contains the configured label."""
    pred_info = extract_label_from_text(
        text=text,
        label_values=label_values,
        answer_prefix=answer_prefix,
    )
    spans = token_spans_from_token_ids(
        tokenizer=tokenizer,
        token_ids=token_ids,
        text=text,
        token_text_cache=token_text_cache,
    )
    label_start = int(pred_info["label_start"])
    target_span = None
    for span in spans:
        if int(span["end"]) > label_start:
            target_span = span
            break
    if target_span is None:
        raise ValueError(f"Could not locate label token in generated text: {text!r}")

    token_start = int(target_span["start"])
    token_end = int(target_span["end"])
    if token_start > label_start:
        raise ValueError(
            "Generated token boundary starts after label start; cannot build "
            f"one-token label probe for text={text!r}, span={target_span!r}"
        )
    return {
        **pred_info,
        "prefix_before_target_token": text[:token_start],
        "target_token_start": token_start,
        "target_token_end": token_end,
        "target_token_id": int(target_span["token_id"]),
        "target_token_text": target_span["token_text"],
        "target_token_index": int(target_span["index"]),
        "target_token_prefix": text[token_start:label_start],
        "target_token_label_text": text[label_start:min(int(pred_info["label_end"]), token_end)],
        "generated_token_spans": spans,
        "prefix_token_ids": [int(token_id) for token_id in token_ids[: int(target_span["index"])]],
    }


def serialize_top_logprobs_at_position(
    logits_at_position: Mapping[int, Any],
    tokenizer,
    top_k: int,
    token_text_cache: dict[int, str] | None = None,
) -> list[dict[str, Any]]:
    """Serialize top-k rows from a vLLM logprobs dict."""
    if not logits_at_position or top_k <= 0:
        return []
    rows = []
    items = []
    for token_id, logprob_obj in logits_at_position.items():
        value = float(getattr(logprob_obj, "logprob", logprob_obj))
        rank = getattr(logprob_obj, "rank", None)
        items.append((int(token_id), value, rank))
    items.sort(key=lambda item: item[1], reverse=True)
    for rank_idx, (token_id, value, rank) in enumerate(items[: int(top_k)], start=1):
        rows.append(
            {
                "rank": int(rank or rank_idx),
                "token_id": int(token_id),
                "token_text": decode_token_cached(tokenizer, token_id, cache=token_text_cache),
                "value": value,
            }
        )
    return rows


def serialize_vllm_generation(
    request_output,
    tokenizer,
    prompt_text: str | None,
    cfg: Mapping[str, Any],
    token_text_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Serialize exact generated token ids and optional top-k generation logits."""
    trace_cfg = cfg.get("generation_trace") or {}
    top_k = get_trace_top_k_for_all_output_tokens(trace_cfg)

    row: dict[str, Any] = {
        "request_id": getattr(request_output, "request_id", None),
        "prompt_length": len(prompt_text) if prompt_text is not None else None,
        "outputs": [],
    }
    if prompt_text is not None:
        row["prompt_hash"] = prompt_hash(prompt_text)

    for output_idx, output in enumerate(request_output.outputs):
        token_ids = [int(token_id) for token_id in (getattr(output, "token_ids", None) or [])]
        out = {
            "index": output_idx,
            "text": getattr(output, "text", ""),
            "finish_reason": getattr(output, "finish_reason", None),
            "stop_reason": getattr(output, "stop_reason", None),
            "cumulative_logprob": getattr(output, "cumulative_logprob", None),
            "token_ids": token_ids,
            "token_texts": [
                decode_token_cached(tokenizer, token_id, cache=token_text_cache)
                for token_id in token_ids
            ],
        }
        if top_k and getattr(output, "logprobs", None):
            out["top_logprobs"] = [
                serialize_top_logprobs_at_position(
                    logits_at_position=position,
                    tokenizer=tokenizer,
                    top_k=top_k,
                    token_text_cache=token_text_cache,
                )
                for position in output.logprobs
            ]
        row["outputs"].append(out)
    return row


def collect_vllm_text_outputs(args, request_outputs) -> list[Any]:
    """Convert vLLM outputs to the historical pred field shape."""
    list_response = []
    for request_output in request_outputs:
        if args.decoding_strategy != "greedy" and getattr(args, "sample_size", 1) > 1:
            list_response.append([output.text for output in request_output.outputs])
        else:
            list_response.append(request_output.outputs[0].text)
    return list_response


def _build_probe_prompt(
    tokenizer,
    prompt_text: str,
    prefix_token_ids: Sequence[int],
    prefix_text: str,
    mode: str,
) -> Any:
    if mode == "token_ids":
        prompt_token_ids = encode_without_special_tokens(tokenizer, prompt_text)
        return {"prompt_token_ids": [int(x) for x in prompt_token_ids] + [int(x) for x in prefix_token_ids]}
    if mode == "text":
        return prompt_text + prefix_text
    raise ValueError(f"Unsupported probe_prompt_mode: {mode}")


def _model_name_candidates(model_name: str | None) -> list[str]:
    """Return exact and common base-name variants for config lookup."""
    if not model_name:
        return []
    names = [str(model_name)]
    for suffix in ("-Non-Thinking", "-Thinking", "-low", "-medium", "-high"):
        if names[0].endswith(suffix):
            names.append(names[0][: -len(suffix)])
    out = []
    for name in names:
        if name and name not in out:
            out.append(name)
    return out


def _normalize_label_key(label: Any) -> str:
    """Normalize YAML label keys while preserving yes/no labels."""
    if isinstance(label, bool):
        return "Yes" if label else "No"
    return str(label)


def _configured_label_token_id_overrides(
    *,
    target_cfg: Mapping[str, Any],
    label_probability_cfg: Mapping[str, Any],
    model_name: str | None,
    target_prefix: str,
) -> dict[str, list[int]]:
    """
    Return task/model-specific token-id additions for label scoring.

    This is intentionally a scoring-only hook. It handles tokenizer edge cases
    where the model's vLLM tokenizer and the local HF tokenizer disagree about a
    single label string, without changing the actual generation path.
    """
    merged: dict[str, set[int]] = {}
    candidate_model_names = _model_name_candidates(model_name)
    if not candidate_model_names:
        return {}

    override_maps = [
        label_probability_cfg.get("model_label_token_id_overrides") or {},
        target_cfg.get("model_label_token_id_overrides") or {},
    ]
    for override_map in override_maps:
        if not isinstance(override_map, Mapping):
            continue
        for model_key in [*candidate_model_names, "*"]:
            model_cfg = override_map.get(model_key)
            if not isinstance(model_cfg, Mapping):
                continue

            label_maps = []
            direct = model_cfg.get("label_token_ids") or model_cfg.get("labels")
            if isinstance(direct, Mapping):
                label_maps.append(direct)

            prefixes = model_cfg.get("prefixes") or {}
            if isinstance(prefixes, Mapping):
                for prefix_key in (target_prefix, "*"):
                    prefix_label_map = prefixes.get(prefix_key)
                    if isinstance(prefix_label_map, Mapping):
                        label_maps.append(prefix_label_map)

            for label_map in label_maps:
                for label, token_ids in label_map.items():
                    if token_ids is None:
                        continue
                    if isinstance(token_ids, (str, bytes)):
                        token_id_list = [token_ids]
                    else:
                        token_id_list = token_ids
                    label_key = _normalize_label_key(label)
                    merged.setdefault(label_key, set())
                    for token_id in token_id_list:
                        merged[label_key].add(int(token_id))

    return {label: sorted(token_ids) for label, token_ids in merged.items()}


def _label_token_ids_for_prefix(
    tokenizer,
    target_cfg: Mapping[str, Any],
    label_probability_cfg: Mapping[str, Any],
    target_prefix: str,
    model_name: str | None = None,
) -> dict[str, list[int]]:
    label_values = [str(label) for label in target_cfg["label_values"]]
    configured_prefixes = label_probability_cfg.get("label_token_prefixes", "auto")
    if configured_prefixes in (None, "auto"):
        prefixes = [str(target_prefix)]
    else:
        prefixes = [str(prefix) for prefix in configured_prefixes]
    for prefix in label_probability_cfg.get("extra_label_token_prefixes", []) or []:
        prefix = str(prefix)
        if prefix not in prefixes:
            prefixes.append(prefix)
    label_token_ids = get_label_token_ids(
        tokenizer=tokenizer,
        label_values=label_values,
        candidate_prefixes=prefixes,
        scan_decoded_vocab=bool(label_probability_cfg.get("scan_decoded_vocab", True)),
    )
    overrides = _configured_label_token_id_overrides(
        target_cfg=target_cfg,
        label_probability_cfg=label_probability_cfg,
        model_name=model_name,
        target_prefix=str(target_prefix),
    )
    for label, token_ids in overrides.items():
        if label not in label_token_ids:
            continue
        merged = set(int(token_id) for token_id in label_token_ids[label])
        merged.update(int(token_id) for token_id in token_ids)
        label_token_ids[label] = sorted(merged)
    return label_token_ids


def _top_token_matches_target(
    top_logits: Sequence[Mapping[str, Any]] | None,
    target_token_id: int,
    target_token_text: str,
) -> dict[str, Any]:
    if not top_logits:
        return {
            "exact_token_match": None,
            "token_text_match": None,
            "stripped_label_match": None,
        }
    top = top_logits[0]
    return {
        "exact_token_match": int(top.get("token_id")) == int(target_token_id),
        "token_text_match": top.get("token_text") == target_token_text,
        "stripped_label_match": _normalize_decoded_label_text(top.get("token_text"))
        == _normalize_decoded_label_text(target_token_text),
    }


def _normalize_decoded_label_text(text: Any) -> str:
    """Normalize decoded one-token label text for audit-only comparisons."""
    return str(text or "").replace("Ġ", " ").replace("▁", " ").strip()


def _flatten_label_token_ids(label_token_ids: Mapping[str, Sequence[int]]) -> list[int]:
    """Return unique label token ids in stable label/key order."""
    out: list[int] = []
    seen = set()
    for token_ids in label_token_ids.values():
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id not in seen:
                out.append(token_id)
                seen.add(token_id)
    return out


def _logit_block_has_error(block: Any) -> bool:
    if not isinstance(block, Mapping):
        return False
    if block.get("error"):
        return True
    outputs = block.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        return any(_logit_block_has_error(output) for output in outputs)
    return False


def _logit_block_errors(block: Any) -> list[str]:
    if not isinstance(block, Mapping):
        return []
    errors = []
    if block.get("error"):
        errors.append(str(block["error"]))
    outputs = block.get("outputs")
    if isinstance(outputs, Sequence) and not isinstance(outputs, (str, bytes)):
        for output in outputs:
            errors.extend(_logit_block_errors(output))
    return errors


def _mark_exact_probe_fallback(block: Any, generation_errors: Sequence[str]) -> Any:
    if not isinstance(block, Mapping):
        return block
    marked = copy.deepcopy(block)
    if marked.get("mode") == "multi_output_label_probability":
        for output in marked.get("outputs") or []:
            if isinstance(output, dict) and not output.get("error"):
                output["source"] = "exact_output_token_probe_fallback"
                output["mode"] = "exact_output_token_probe_fallback"
                output["fallback_from_generation_logprobs_errors"] = list(
                    generation_errors
                )
        return marked
    if not marked.get("error"):
        marked["source"] = "exact_output_token_probe_fallback"
        marked["mode"] = "exact_output_token_probe_fallback"
        marked["fallback_from_generation_logprobs_errors"] = list(generation_errors)
    return marked


def _run_vllm_label_probability_from_generation_logprobs(
    *,
    args,
    tokenizer,
    request_outputs,
    target_cfg: Mapping[str, Any],
    sidecar_writer: JsonlSidecarWriter | None = None,
    token_text_cache: dict[int, str] | None = None,
) -> list[dict[str, Any] | None]:
    """Score label groups from the exact logprobs returned during generation."""
    cfg = getattr(args, "logits", {}) or {}
    lp_cfg = cfg.get("label_probability") or {}

    label_values = [str(label) for label in target_cfg["label_values"]]
    label_groups = target_cfg["label_groups"]
    answer_prefix = target_cfg.get("answer_prefix")
    top_k_logits = int(lp_cfg.get("top_k_logits", 0) or 0)
    save_top_logits = bool(lp_cfg.get("save_top_logits", True))
    fail_on_error = bool(lp_cfg.get("fail_on_error", False))
    allow_fallback = bool(lp_cfg.get("fallback_to_exact_probe", False))
    requested_logprobs = get_generation_trace_top_k(args)

    flattened: list[dict[str, Any] | None] = []
    for row_idx, request_output in enumerate(request_outputs):
        output_blocks = []
        for output_idx, output in enumerate(request_output.outputs):
            text = str(getattr(output, "text", "") or "")
            token_ids = [
                int(token_id)
                for token_id in (getattr(output, "token_ids", None) or [])
            ]
            try:
                target_info = infer_label_target_from_generated_tokens(
                    text=text,
                    token_ids=token_ids,
                    tokenizer=tokenizer,
                    label_values=label_values,
                    answer_prefix=answer_prefix,
                    token_text_cache=token_text_cache,
                )
                output_logprobs = getattr(output, "logprobs", None)
                if output_logprobs is None:
                    raise ValueError(
                        "Generation output.logprobs is None; set "
                        "label_probability.mode=generation_logprobs and "
                        "top_k_logits > 0 so main SamplingParams requests logprobs."
                    )
                target_token_index = int(target_info["target_token_index"])
                if target_token_index >= len(output_logprobs):
                    raise IndexError(
                        "Target token index is outside output.logprobs: "
                        f"index={target_token_index}, len={len(output_logprobs)}"
                    )
                logits_at_position = output_logprobs[target_token_index]
                if not logits_at_position:
                    raise ValueError(
                        f"Generation logprobs at target index {target_token_index} "
                        "are empty."
                    )
                label_token_ids = _label_token_ids_for_prefix(
                    tokenizer=tokenizer,
                    target_cfg=target_cfg,
                    label_probability_cfg=lp_cfg,
                    target_prefix=target_info["target_token_prefix"],
                    model_name=getattr(args, "model_name", None),
                )
                score = score_label_groups_from_logits(
                    logits_at_position=logits_at_position,
                    tokenizer=tokenizer,
                    label_token_ids=label_token_ids,
                    label_groups=label_groups,
                    token_text_cache=token_text_cache,
                )
                top_logits = (
                    top_token_rows_from_logits(
                        logits_at_position=logits_at_position,
                        tokenizer=tokenizer,
                        top_k=top_k_logits,
                        vocab_logsumexp=score["vocab_logsumexp"],
                        token_text_cache=token_text_cache,
                        include_token_ids=_flatten_label_token_ids(label_token_ids),
                    )
                    if save_top_logits and top_k_logits
                    else None
                )
                target_token = {
                    "token_id": target_info["target_token_id"],
                    "token_text": target_info["target_token_text"],
                    "text": target_info["target_token_text"],
                }

                top_logits_ref = None
                if sidecar_writer is not None and top_logits is not None:
                    top_logits_ref = sidecar_writer.write(
                        {
                            "idx": row_idx,
                            "output_idx": output_idx,
                            "source": "generation_logprobs",
                            "pred_label": target_info["label"],
                            "pred_prefix_before_target_token": target_info[
                                "prefix_before_target_token"
                            ],
                            "pred_target_token_id": target_info["target_token_id"],
                            "pred_target_token_text": target_info[
                                "target_token_text"
                            ],
                            "generation_target_token": target_token,
                            "vocab_logsumexp": score.get("vocab_logsumexp"),
                            "top_logits": top_logits,
                        },
                        top_k=top_k_logits,
                    )

                block = {
                    "mode": "generation_logprobs",
                    "source": "generation_logprobs",
                    "answer_prefix": answer_prefix,
                    "pred_label": target_info["label"],
                    "pred_prefix_before_label": target_info["prefix_before_label"],
                    "pred_prefix_before_target_token": target_info[
                        "prefix_before_target_token"
                    ],
                    "pred_label_start": target_info["label_start"],
                    "pred_label_end": target_info["label_end"],
                    "pred_target_token_id": target_info["target_token_id"],
                    "pred_target_token_text": target_info["target_token_text"],
                    "pred_target_token_index": target_info["target_token_index"],
                    "pred_target_token_start": target_info["target_token_start"],
                    "pred_target_token_end": target_info["target_token_end"],
                    "pred_target_token_prefix": target_info["target_token_prefix"],
                    "pred_target_token_label_text": target_info[
                        "target_token_label_text"
                    ],
                    "label_token_ids": label_token_ids,
                    "generation_consistency": {
                        "exact_token_match": True,
                        "token_text_match": True,
                        "stripped_label_match": True,
                        "top_logit_token_match": _top_token_matches_target(
                            top_logits,
                            target_info["target_token_id"],
                            target_info["target_token_text"],
                        ),
                    },
                    "sampling": {
                        "seed": int(getattr(args, "seed", 0))
                        if getattr(args, "seed", None) is not None
                        else None,
                        "temperature": 0.0
                        if getattr(args, "decoding_strategy", "greedy") == "greedy"
                        else getattr(args, "temperature", None),
                        "max_tokens": getattr(args, "max_token_output", None),
                        "logprobs": requested_logprobs,
                        "logprobs_mode": getattr(
                            args, "vllm_logprobs_mode", "raw_logits"
                        ),
                    },
                    "normalization": {
                        "vocab_probability_scope": (
                            "returned_generation_logprobs_top_k"
                            if requested_logprobs != -1
                            else "whole_vocab"
                        ),
                        "conditional_probability_scope": "configured_label_groups",
                    },
                    **score,
                }
                if top_logits_ref is not None:
                    block["top_logits_ref"] = top_logits_ref
                elif top_logits is not None:
                    block["top_logits"] = top_logits
                output_blocks.append(block)
            except Exception as exc:
                if fail_on_error and not allow_fallback:
                    raise
                output_blocks.append(
                    {
                        "mode": "generation_logprobs",
                        "source": "generation_logprobs",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

        if len(output_blocks) == 1:
            flattened.append(output_blocks[0])
        else:
            flattened.append(
                {"mode": "multi_output_label_probability", "outputs": output_blocks}
            )
    return flattened


def run_vllm_label_probability_probes(
    *,
    args,
    model,
    tokenizer,
    request_outputs,
    prompt_texts: Sequence[str],
    target_cfg: Mapping[str, Any],
    sidecar_writer: JsonlSidecarWriter | None = None,
    token_text_cache: dict[int, str] | None = None,
) -> list[dict[str, Any] | None]:
    """Probe exact generated label-token positions and return compact logit blocks."""
    cfg = getattr(args, "logits", {}) or {}
    lp_cfg = cfg.get("label_probability") or {}
    if not lp_cfg.get("enabled", False):
        return [None for _ in request_outputs]
    mode = str(lp_cfg.get("mode", "exact_output_token_probe"))
    if mode == "generation_logprobs":
        generation_records = _run_vllm_label_probability_from_generation_logprobs(
            args=args,
            tokenizer=tokenizer,
            request_outputs=request_outputs,
            target_cfg=target_cfg,
            sidecar_writer=sidecar_writer,
            token_text_cache=token_text_cache,
        )
        if not bool(lp_cfg.get("fallback_to_exact_probe", False)):
            return generation_records

        fallback_indices = [
            idx
            for idx, block in enumerate(generation_records)
            if _logit_block_has_error(block)
        ]
        if not fallback_indices:
            return generation_records

        fallback_args = copy.copy(args)
        fallback_cfg = copy.deepcopy(cfg)
        fallback_cfg.setdefault("label_probability", {})
        fallback_cfg["label_probability"]["mode"] = "exact_output_token_probe"
        fallback_args.logits = fallback_cfg
        fallback_records = run_vllm_label_probability_probes(
            args=fallback_args,
            model=model,
            tokenizer=tokenizer,
            request_outputs=request_outputs,
            prompt_texts=prompt_texts,
            target_cfg=target_cfg,
            sidecar_writer=sidecar_writer,
            token_text_cache=token_text_cache,
        )
        merged = list(generation_records)
        for idx in fallback_indices:
            fallback_block = fallback_records[idx]
            if fallback_block is not None and not _logit_block_has_error(
                fallback_block
            ):
                merged[idx] = _mark_exact_probe_fallback(
                    fallback_block,
                    generation_errors=_logit_block_errors(generation_records[idx]),
                )
            elif bool(lp_cfg.get("fail_on_error", False)):
                errors = _logit_block_errors(fallback_block) or _logit_block_errors(
                    generation_records[idx]
                )
                raise ValueError(
                    "Label probability fallback failed for row "
                    f"{idx}: {'; '.join(errors)}"
                )
        return merged
    if mode != "exact_output_token_probe":
        raise ValueError("Only label_probability.mode=exact_output_token_probe is supported.")

    from vllm import SamplingParams

    label_values = [str(label) for label in target_cfg["label_values"]]
    label_groups = target_cfg["label_groups"]
    answer_prefix = target_cfg.get("answer_prefix")
    top_k_logits = int(lp_cfg.get("top_k_logits", 0) or 0)
    save_top_logits = bool(lp_cfg.get("save_top_logits", True))
    fail_on_error = bool(lp_cfg.get("fail_on_error", False))
    probe_prompt_mode = str(lp_cfg.get("probe_prompt_mode", "token_ids"))

    records: list[dict[str, Any] | None] = [None for _ in request_outputs]
    probe_items: list[dict[str, Any]] = []
    errors: dict[tuple[int, int], str] = {}

    for row_idx, (request_output, prompt_text) in enumerate(zip(request_outputs, prompt_texts)):
        output_blocks = []
        for output_idx, output in enumerate(request_output.outputs):
            text = str(getattr(output, "text", "") or "")
            token_ids = [int(token_id) for token_id in (getattr(output, "token_ids", None) or [])]
            try:
                target_info = infer_label_target_from_generated_tokens(
                    text=text,
                    token_ids=token_ids,
                    tokenizer=tokenizer,
                    label_values=label_values,
                    answer_prefix=answer_prefix,
                    token_text_cache=token_text_cache,
                )
                probe_prompt = _build_probe_prompt(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    prefix_token_ids=target_info["prefix_token_ids"],
                    prefix_text=target_info["prefix_before_target_token"],
                    mode=probe_prompt_mode,
                )
                label_token_ids = _label_token_ids_for_prefix(
                    tokenizer=tokenizer,
                    target_cfg=target_cfg,
                    label_probability_cfg=lp_cfg,
                    target_prefix=target_info["target_token_prefix"],
                    model_name=getattr(args, "model_name", None),
                )
                output_blocks.append(None)
                probe_items.append(
                    {
                        "row_idx": row_idx,
                        "output_idx": output_idx,
                        "prompt": probe_prompt,
                        "target_info": target_info,
                        "label_token_ids": label_token_ids,
                    }
                )
            except Exception as exc:
                if fail_on_error:
                    raise
                errors[(row_idx, output_idx)] = f"{type(exc).__name__}: {exc}"
                output_blocks.append(
                    {
                        "mode": "exact_output_token_probe",
                        "error": errors[(row_idx, output_idx)],
                    }
                )
        records[row_idx] = {"outputs": output_blocks}

    sampling_params = SamplingParams(
        seed=int(getattr(args, "seed", 0)) if getattr(args, "seed", None) is not None else None,
        temperature=0.0,
        max_tokens=1,
        logprobs=-1,
    )
    probe_batch_size = max(1, int(lp_cfg.get("probe_batch_size", 64) or 64))

    for start in range(0, len(probe_items), probe_batch_size):
        chunk = probe_items[start : start + probe_batch_size]
        probe_outputs = model.generate(
            [item["prompt"] for item in chunk],
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        for item, output_one in zip(chunk, probe_outputs):
            score = score_vllm_probe_output(
                output_one=output_one,
                tokenizer=tokenizer,
                label_token_ids=item["label_token_ids"],
                label_groups=label_groups,
                top_k_logits=top_k_logits if save_top_logits else 0,
                token_text_cache=token_text_cache,
            )
            top_logits = score.pop("top_logits", None)
            target_info = item["target_info"]
            generated_token = score.get("probe_generated_token") or {}
            top_logits_ref = None
            if sidecar_writer is not None and top_logits is not None:
                top_logits_ref = sidecar_writer.write(
                    {
                        "idx": item["row_idx"],
                        "output_idx": item["output_idx"],
                        "pred_label": target_info["label"],
                        "pred_prefix_before_target_token": target_info[
                            "prefix_before_target_token"
                        ],
                        "pred_target_token_id": target_info["target_token_id"],
                        "pred_target_token_text": target_info["target_token_text"],
                        "probe_generated_token": generated_token,
                        "vocab_logsumexp": score.get("vocab_logsumexp"),
                        "top_logits": top_logits,
                    },
                    top_k=top_k_logits,
                )

            block = {
                "mode": "exact_output_token_probe",
                "answer_prefix": answer_prefix,
                "pred_label": target_info["label"],
                "pred_prefix_before_label": target_info["prefix_before_label"],
                "pred_prefix_before_target_token": target_info[
                    "prefix_before_target_token"
                ],
                "pred_label_start": target_info["label_start"],
                "pred_label_end": target_info["label_end"],
                "pred_target_token_id": target_info["target_token_id"],
                "pred_target_token_text": target_info["target_token_text"],
                "pred_target_token_index": target_info["target_token_index"],
                "pred_target_token_start": target_info["target_token_start"],
                "pred_target_token_end": target_info["target_token_end"],
                "pred_target_token_prefix": target_info["target_token_prefix"],
                "pred_target_token_label_text": target_info["target_token_label_text"],
                "label_token_ids": item["label_token_ids"],
                    "probe_consistency": {
                        "exact_token_match": generated_token.get("token_id")
                        == target_info["target_token_id"],
                        "token_text_match": generated_token.get("token_text")
                        == target_info["target_token_text"],
                        "stripped_label_match": _normalize_decoded_label_text(
                            generated_token.get("token_text")
                        )
                        == _normalize_decoded_label_text(target_info["label"]),
                    },
                "sampling": {
                    "seed": int(getattr(args, "seed", 0)) if getattr(args, "seed", None) is not None else None,
                    "temperature": 0.0,
                    "max_tokens": 1,
                    "logprobs": -1,
                },
                **score,
            }
            if top_logits_ref is not None:
                block["top_logits_ref"] = top_logits_ref
            elif top_logits is not None:
                block["top_logits"] = top_logits

            records[item["row_idx"]]["outputs"][item["output_idx"]] = block

    flattened: list[dict[str, Any] | None] = []
    for record in records:
        if record is None:
            flattened.append(None)
            continue
        outputs = record.get("outputs") or []
        if len(outputs) == 1:
            flattened.append(outputs[0])
        else:
            flattened.append({"mode": "multi_output_label_probability", "outputs": outputs})
    return flattened


def build_logit_rows(
    dataset,
    list_response: Sequence[Any],
    list_generation: Sequence[dict[str, Any] | None] | None = None,
    list_logit: Sequence[dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    """Build compact rows for a separate .result.logit.json file."""
    rows = []
    list_generation = list_generation or [None] * len(list_response)
    list_logit = list_logit or [None] * len(list_response)
    for idx, (source, pred, generation, logit) in enumerate(
        zip(dataset, list_response, list_generation, list_logit)
    ):
        row = {
            "idx": idx,
            "id": source.get("id"),
            "task": source.get("task"),
            "type": source.get("type"),
            "language": source.get("language"),
            "split": source.get("split"),
            "output": source.get("output"),
            "pred": pred,
        }
        if generation is not None:
            row["generation"] = generation
        if logit is not None:
            row["logit"] = logit
        rows.append(row)
    return rows
