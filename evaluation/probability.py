from __future__ import annotations

import copy
import gzip
import heapq
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from util.io import load_json, write_json

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    tqdm = None


DEFAULT_LABEL_GROUPS = {
    "negative": ["0", "1"],
    "positive": ["2", "3", "4"],
}


def logsumexp(values: Iterable[float]) -> float:
    """Numerically stable log(sum(exp(values)))."""
    values = [float(value) for value in values]
    if not values:
        return float("-inf")
    max_value = max(values)
    if math.isinf(max_value):
        return max_value
    return max_value + math.log(sum(math.exp(value - max_value) for value in values))


def encode_without_special_tokens(tokenizer, text: str) -> list[int]:
    """Tokenize text without adding BOS/EOS/chat special tokens."""
    try:
        return list(tokenizer.encode(text, add_special_tokens=False))
    except TypeError:
        return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def decode_one_token(tokenizer, token_id: int) -> str:
    """Decode one token id without assuming tokenizer-specific internals."""
    return tokenizer.decode([int(token_id)])


def decode_token_cached(tokenizer, token_id: int, cache: dict[int, str] | None = None) -> str:
    """Decode one token id and reuse the value across many records."""
    token_id = int(token_id)
    if cache is None:
        return decode_one_token(tokenizer, token_id)
    if token_id not in cache:
        cache[token_id] = decode_one_token(tokenizer, token_id)
    return cache[token_id]


def token_spans_for_text(tokenizer, text: str) -> list[dict[str, Any]]:
    """
    Tokenize generated text and return decoded-token character spans.

    This is used to reconstruct the token boundary that likely existed in the
    original vLLM generation. For example, Qwen tokenizes ``"Assessment: Yes"``
    with a label token ``" Yes"``, so the probe must stop after
    ``"Assessment:"`` rather than after ``"Assessment: "``.
    """
    token_ids = encode_without_special_tokens(tokenizer, text)
    spans: list[dict[str, Any]] = []
    cursor = 0
    decoded_parts = []
    for idx, token_id in enumerate(token_ids):
        token_text = decode_one_token(tokenizer, token_id)
        decoded_parts.append(token_text)
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

    decoded_text = "".join(decoded_parts)
    if decoded_text != text:
        # Fall back to boundary positions computed from decoded prefixes. This
        # is slower, but handles tokenizers whose one-token decode strings do
        # not concatenate exactly to the full decode.
        decoded_full = tokenizer.decode(token_ids)
        if decoded_full != text:
            raise ValueError(
                "Could not align generated text to tokenizer decode. "
                f"text={text!r}, decoded={decoded_full!r}"
            )
        spans = []
        previous = ""
        for idx, token_id in enumerate(token_ids):
            current = tokenizer.decode(token_ids[: idx + 1])
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


def extract_label_from_text(
    text: str,
    label_values: Sequence[str],
    answer_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Extract a discrete label from a generated answer string.

    For outputs like ``Assessment: 1``, set ``answer_prefix='Assessment:'``.
    The returned ``prefix_before_label`` is the generated text before the label
    token and can be appended to the original formatted prompt for one-token
    probability probing.
    """
    text = "" if text is None else str(text)
    labels = sorted([str(label) for label in label_values], key=len, reverse=True)

    search_start = 0
    if answer_prefix:
        prefix_pos = text.rfind(answer_prefix)
        if prefix_pos < 0:
            raise ValueError(
                f"Could not find answer_prefix={answer_prefix!r} in text: {text!r}"
            )
        search_start = prefix_pos + len(answer_prefix)

    tail = text[search_start:]
    leading_chars = len(tail) - len(tail.lstrip())
    label_start = search_start + leading_chars
    tail_after_space = text[label_start:]

    for label in labels:
        if tail_after_space.startswith(label):
            return {
                "label": label,
                "label_start": label_start,
                "label_end": label_start + len(label),
                "prefix_before_label": text[:label_start],
                "matched": True,
            }

    if answer_prefix:
        raise ValueError(
            "Could not extract a label immediately after "
            f"answer_prefix={answer_prefix!r}; labels={list(label_values)}, "
            f"tail={text[search_start:]!r}"
        )

    # Fallback for legacy callers that do not define a structured answer prefix.
    # Logit scoring tasks should pass answer_prefix and therefore use the strict
    # branch above.
    best = None
    for label in labels:
        pos = text.find(label, search_start)
        if pos >= 0 and (best is None or pos < best[0]):
            best = (pos, label)
    if best is not None:
        pos, label = best
        return {
            "label": label,
            "label_start": pos,
            "label_end": pos + len(label),
            "prefix_before_label": text[:pos],
            "matched": True,
        }

    raise ValueError(
        f"Could not extract one of labels={list(label_values)} from text: {text!r}"
    )


def infer_label_token_probe_target(
    text: str,
    tokenizer,
    label_values: Sequence[str],
    answer_prefix: str | None = None,
) -> dict[str, Any]:
    """
    Infer the original one-token probe position for a completed prediction.

    ``extract_label_from_text`` returns the character prefix immediately before
    the label text. That is not always the same as the token prefix used during
    generation. If the generated label token includes leading whitespace
    (``" Yes"``), the model's original label decision was made before that
    whitespace. This helper reconstructs that token-aligned boundary from the
    decoded prediction text.
    """
    text = "" if text is None else str(text)
    pred_info = extract_label_from_text(
        text=text,
        label_values=label_values,
        answer_prefix=answer_prefix,
    )
    spans = token_spans_for_text(tokenizer, text)
    label_start = int(pred_info["label_start"])
    label_end = int(pred_info["label_end"])

    target_span = None
    for span in spans:
        # A pure whitespace token before the label ends exactly at label_start;
        # the label decision is the next token in that case. A token such as
        # " Yes" spans across label_start, so it becomes the target token.
        if int(span["end"]) > label_start:
            target_span = span
            break
    if target_span is None:
        raise ValueError(f"Could not locate label token span in text: {text!r}")

    token_start = int(target_span["start"])
    token_end = int(target_span["end"])
    if token_start > label_start:
        raise ValueError(
            "Token boundary starts after label start; cannot build a one-token "
            f"label probe for text={text!r}, span={target_span!r}"
        )

    label_token_prefix = text[token_start:label_start]
    token_label_text = text[label_start:min(label_end, token_end)]
    return {
        **pred_info,
        "prefix_before_target_token": text[:token_start],
        "target_token_start": token_start,
        "target_token_end": token_end,
        "target_token_id": int(target_span["token_id"]),
        "target_token_text": target_span["token_text"],
        "target_token_index": int(target_span["index"]),
        "target_token_prefix": label_token_prefix,
        "target_token_label_text": token_label_text,
        "generated_token_spans": spans,
    }


def get_label_token_ids(
    tokenizer,
    label_values: Sequence[str],
    candidate_prefixes: Sequence[str] = ("",),
    scan_decoded_vocab: bool = True,
) -> dict[str, list[int]]:
    """
    Return token ids that can express each label as a single next token.

    ``candidate_prefixes`` controls variants such as ``"1"``, ``" 1"``, or
    ``"\\n1"``. For a probe context ending in ``"Assessment: "``, the exact
    next token is normally ``"1"``; add prefixes only when you intentionally
    want to count whitespace variants too.
    """
    label_to_texts = {
        str(label): {f"{prefix}{label}" for prefix in candidate_prefixes}
        for label in label_values
    }
    label_to_ids: dict[str, set[int]] = {str(label): set() for label in label_values}

    for label, texts in label_to_texts.items():
        for text in texts:
            token_ids = encode_without_special_tokens(tokenizer, text)
            if len(token_ids) == 1:
                label_to_ids[label].add(int(token_ids[0]))

    if scan_decoded_vocab:
        all_texts = {
            text: label for label, texts in label_to_texts.items() for text in texts
        }
        vocab_size = len(tokenizer)
        for token_id in range(vocab_size):
            try:
                decoded = decode_one_token(tokenizer, token_id)
            except Exception:
                continue
            label = all_texts.get(decoded)
            if label is not None:
                label_to_ids[label].add(int(token_id))

    missing = [label for label, token_ids in label_to_ids.items() if not token_ids]
    if missing:
        raise ValueError(f"No single-token ids found for labels: {missing}")

    return {label: sorted(token_ids) for label, token_ids in label_to_ids.items()}


def inspect_label_token_variants(
    tokenizer,
    label_values: Sequence[str],
    candidate_prefixes: Sequence[str],
) -> dict[str, Any]:
    """Return tokenizer details for label strings such as ``"Yes"`` and ``" Yes"``."""
    variants: dict[str, Any] = {}
    for label in [str(label) for label in label_values]:
        variants[label] = []
        for prefix in candidate_prefixes:
            text = f"{prefix}{label}"
            token_ids = encode_without_special_tokens(tokenizer, text)
            variants[label].append(
                {
                    "prefix": prefix,
                    "text": text,
                    "token_ids": [int(token_id) for token_id in token_ids],
                    "token_texts": [
                        decode_one_token(tokenizer, int(token_id))
                        for token_id in token_ids
                    ],
                    "token_count": len(token_ids),
                    "single_token": len(token_ids) == 1,
                }
            )
    return variants


def build_probability_probe_prompt(
    record: Mapping[str, Any],
    tokenizer,
    model_name: str,
    max_token_input: int,
    label_values: Sequence[str],
    answer_prefix: str = "Assessment:",
    pred_field: str = "pred",
    examples: Sequence[Mapping[str, Any]] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    probe_cut: str = "token_aligned",
) -> dict[str, Any]:
    """
    Build the prompt used to re-score a completed prediction.

    ``probe_cut='token_aligned'`` stops at the decoded-token boundary before
    the label token. This reproduces the original next-token decision more
    closely than blindly cutting before the label characters. For Qwen,
    ``"Assessment: Yes"`` is usually ``"Assessment:"`` + ``" Yes"``, while
    ``"Assessment: 1"`` is usually ``"Assessment: "`` + ``"1"``.

    ``probe_cut='text_before_label'`` keeps the previous behavior and cuts
    immediately before the label text.
    """
    if pred_field not in record:
        raise KeyError(f"Result record does not contain pred field: {pred_field}")

    from dataset.process import format_chat

    formatted_prompt = format_chat(
        model_name=model_name,
        tokenizer=tokenizer,
        data=dict(record),
        max_token_input=max_token_input,
        examples=list(examples or []),
        chat_kwargs=dict(chat_kwargs or {}),
    )

    pred_text = str(record[pred_field])
    if probe_cut == "token_aligned":
        pred_info = infer_label_token_probe_target(
            pred_text,
            tokenizer=tokenizer,
            label_values=label_values,
            answer_prefix=answer_prefix,
        )
        prompt_suffix = pred_info["prefix_before_target_token"]
    elif probe_cut == "text_before_label":
        pred_info = extract_label_from_text(
            pred_text,
            label_values=label_values,
            answer_prefix=answer_prefix,
        )
        prompt_suffix = pred_info["prefix_before_label"]
        pred_info = {
            **pred_info,
            "prefix_before_target_token": prompt_suffix,
            "target_token_start": pred_info["label_start"],
            "target_token_end": pred_info["label_end"],
            "target_token_id": None,
            "target_token_text": None,
            "target_token_index": None,
            "target_token_prefix": "",
            "target_token_label_text": pred_info["label"],
            "generated_token_spans": [],
        }
    else:
        raise ValueError(
            "probe_cut must be 'token_aligned' or 'text_before_label', "
            f"got {probe_cut!r}"
        )

    return {
        "prompt": formatted_prompt + prompt_suffix,
        "formatted_prompt": formatted_prompt,
        "probe_cut": probe_cut,
        "pred_label": pred_info["label"],
        "pred_prefix_before_label": pred_info["prefix_before_label"],
        "pred_prefix_before_target_token": pred_info["prefix_before_target_token"],
        "pred_label_start": pred_info["label_start"],
        "pred_label_end": pred_info["label_end"],
        "pred_target_token_id": pred_info["target_token_id"],
        "pred_target_token_text": pred_info["target_token_text"],
        "pred_target_token_index": pred_info["target_token_index"],
        "pred_target_token_start": pred_info["target_token_start"],
        "pred_target_token_end": pred_info["target_token_end"],
        "pred_target_token_prefix": pred_info["target_token_prefix"],
        "pred_target_token_label_text": pred_info["target_token_label_text"],
    }


def _logits_dict_to_float(logits_at_position: Mapping[int, Any]) -> dict[int, float]:
    return {
        int(token_id): float(logprob_obj.logprob)
        for token_id, logprob_obj in logits_at_position.items()
    }


def top_token_rows_from_logits(
    logits_at_position: Mapping[int, Any],
    tokenizer,
    top_k: int,
    vocab_logsumexp: float | None = None,
    token_text_cache: dict[int, str] | None = None,
    include_token_ids: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    """Return top-k token rows plus requested audit token ids when available."""
    if top_k <= 0:
        return []
    raw_logits = _logits_dict_to_float(logits_at_position)
    if vocab_logsumexp is None:
        vocab_logsumexp = logsumexp(raw_logits.values())
    top_items = heapq.nlargest(int(top_k), raw_logits.items(), key=lambda item: item[1])
    rows = []
    seen_token_ids = set()

    def _build_row(token_id: int, raw_logit: float, rank_idx: int, selection: str):
        logprob = raw_logit - vocab_logsumexp
        logprob_obj = logits_at_position[int(token_id)]
        return {
            "rank": getattr(logprob_obj, "rank", None) or rank_idx,
            "token_id": int(token_id),
            "token_text": decode_token_cached(
                tokenizer, int(token_id), cache=token_text_cache
            ),
            "raw_logit": float(raw_logit),
            "logprob": float(logprob),
            "probability": float(math.exp(logprob)),
            "selection": selection,
        }

    for rank_idx, (token_id, raw_logit) in enumerate(top_items, start=1):
        token_id = int(token_id)
        rows.append(_build_row(token_id, raw_logit, rank_idx, "top_k"))
        seen_token_ids.add(token_id)

    if include_token_ids:
        for token_id in include_token_ids:
            token_id = int(token_id)
            if token_id in seen_token_ids or token_id not in raw_logits:
                continue
            rows.append(
                _build_row(
                    token_id=token_id,
                    raw_logit=raw_logits[token_id],
                    rank_idx=len(rows) + 1,
                    selection="label_token",
                )
            )
            seen_token_ids.add(token_id)
    return rows


def score_label_groups_from_logits(
    logits_at_position: Mapping[int, Any],
    tokenizer,
    label_token_ids: Mapping[str, Sequence[int]],
    label_groups: Mapping[str, Sequence[str]] = DEFAULT_LABEL_GROUPS,
    token_text_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    """
    Aggregate whole-vocab raw logits into label and group probabilities.

    Raw logits are first normalized over the full vocabulary with log-softmax.
    Label probabilities sum the probabilities of all configured token variants.
    Group probabilities then sum label probabilities, using logsumexp in log
    space. This is the probability-valid version of "0+1" and "2+3+4".
    """
    raw_logits = _logits_dict_to_float(logits_at_position)
    vocab_logsumexp = logsumexp(raw_logits.values())

    labels: dict[str, Any] = {}
    for label, token_ids in label_token_ids.items():
        token_rows = []
        label_logits = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id not in raw_logits:
                raise KeyError(
                    f"Token id {token_id} for label {label!r} is missing from "
                    "logits_at_position. Use SamplingParams(logprobs=-1) and "
                    "LLM(max_logprobs=-1)."
                )
            raw_logit = raw_logits[token_id]
            token_logprob = raw_logit - vocab_logsumexp
            label_logits.append(raw_logit)
            token_rows.append(
                {
                    "token_id": token_id,
                    "token_text": decode_token_cached(
                        tokenizer, token_id, cache=token_text_cache
                    ),
                    "raw_logit": raw_logit,
                    "logprob": token_logprob,
                    "probability": math.exp(token_logprob),
                    "rank": getattr(logits_at_position[token_id], "rank", None),
                }
            )

        label_logit = logsumexp(label_logits)
        label_logprob = label_logit - vocab_logsumexp
        labels[str(label)] = {
            "token_ids": [int(token_id) for token_id in token_ids],
            "tokens": token_rows,
            "logit": label_logit,
            "logprob": label_logprob,
            "probability": math.exp(label_logprob),
        }

    groups: dict[str, Any] = {}
    for group_name, group_labels in label_groups.items():
        group_labels = [str(label) for label in group_labels]
        missing = [label for label in group_labels if label not in labels]
        if missing:
            raise KeyError(f"Group {group_name!r} references missing labels: {missing}")
        group_logit = logsumexp(labels[label]["logit"] for label in group_labels)
        group_logprob = group_logit - vocab_logsumexp
        groups[str(group_name)] = {
            "labels": group_labels,
            "logit": group_logit,
            "logprob": group_logprob,
            "probability": math.exp(group_logprob),
        }

    group_norm = logsumexp(group["logit"] for group in groups.values())
    for group in groups.values():
        group["conditional_logprob"] = group["logit"] - group_norm
        group["conditional_probability"] = math.exp(group["conditional_logprob"])

    return {
        "vocab_logsumexp": vocab_logsumexp,
        "labels": labels,
        "groups": groups,
    }


def score_vllm_probe_output(
    output_one,
    tokenizer,
    label_token_ids: Mapping[str, Sequence[int]],
    label_groups: Mapping[str, Sequence[str]] = DEFAULT_LABEL_GROUPS,
    top_k_logits: int = 0,
    token_text_cache: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Score the first generated token from one vLLM RequestOutput."""
    completion = output_one.outputs[0]
    if completion.logprobs is None:
        raise ValueError("completion.logprobs is None; set SamplingParams(logprobs=-1).")
    if not completion.logprobs:
        raise ValueError("completion.logprobs is empty.")

    score = score_label_groups_from_logits(
        completion.logprobs[0],
        tokenizer=tokenizer,
        label_token_ids=label_token_ids,
        label_groups=label_groups,
        token_text_cache=token_text_cache,
    )
    generated_token_id = int(completion.token_ids[0]) if completion.token_ids else None
    score["probe_generated_token"] = {
        "token_id": generated_token_id,
        "token_text": (
            decode_token_cached(tokenizer, generated_token_id, cache=token_text_cache)
            if generated_token_id is not None
            else None
        ),
        "text": completion.text,
    }
    if top_k_logits:
        include_token_ids = [
            int(token_id)
            for token_ids in label_token_ids.values()
            for token_id in token_ids
        ]
        score["top_logits"] = top_token_rows_from_logits(
            completion.logprobs[0],
            tokenizer=tokenizer,
            top_k=int(top_k_logits),
            vocab_logsumexp=score["vocab_logsumexp"],
            token_text_cache=token_text_cache,
            include_token_ids=include_token_ids,
        )
    return score


def add_vllm_label_probabilities(
    records: Sequence[Mapping[str, Any]],
    model,
    tokenizer,
    model_name: str,
    max_token_input: int,
    label_values: Sequence[str],
    label_groups: Mapping[str, Sequence[str]] = DEFAULT_LABEL_GROUPS,
    answer_prefix: str = "Assessment:",
    pred_field: str = "pred",
    output_field: str = "output",
    examples: Sequence[Mapping[str, Any]] | None = None,
    chat_kwargs: Mapping[str, Any] | None = None,
    batch_size: int = 32,
    seed: int = 42,
    label_token_prefixes: Sequence[str] | str | None = "auto",
    extra_label_token_prefixes: Sequence[str] = (),
    scan_decoded_vocab: bool = True,
    probe_cut: str = "token_aligned",
    top_k_logits: int = 0,
    top_logits_path: str | Path | None = None,
    in_place: bool = False,
    use_tqdm: bool = True,
) -> list[dict[str, Any]]:
    """
    Re-score completed result records with a one-token vLLM probability probe.

    The returned records contain a compact ``record["logit"]`` block. The name
    is kept user-facing for compatibility, but the block includes raw logits,
    logprobs, probabilities, and binary conditional probabilities.
    """
    from vllm import SamplingParams

    if batch_size < 1:
        raise ValueError("batch_size must be >= 1.")

    output_records = [record if in_place else copy.deepcopy(dict(record)) for record in records]
    label_values = [str(label) for label in label_values]

    observed_label_token_prefixes: list[str] = []
    if probe_cut == "token_aligned":
        for record in output_records:
            target_info = infer_label_token_probe_target(
                str(record[pred_field]),
                tokenizer=tokenizer,
                label_values=label_values,
                answer_prefix=answer_prefix,
            )
            prefix = str(target_info["target_token_prefix"])
            if prefix not in observed_label_token_prefixes:
                observed_label_token_prefixes.append(prefix)
    elif probe_cut != "text_before_label":
        raise ValueError(
            "probe_cut must be 'token_aligned' or 'text_before_label', "
            f"got {probe_cut!r}"
        )

    if label_token_prefixes in (None, "auto"):
        active_label_token_prefixes = observed_label_token_prefixes or [""]
    else:
        active_label_token_prefixes = [str(prefix) for prefix in label_token_prefixes]
    for prefix in extra_label_token_prefixes:
        prefix = str(prefix)
        if prefix not in active_label_token_prefixes:
            active_label_token_prefixes.append(prefix)

    label_token_ids = get_label_token_ids(
        tokenizer=tokenizer,
        label_values=label_values,
        candidate_prefixes=active_label_token_prefixes,
        scan_decoded_vocab=scan_decoded_vocab,
    )

    sampling_params = SamplingParams(
        seed=int(seed) if seed is not None else None,
        temperature=0.0,
        max_tokens=1,
        logprobs=-1,
    )

    top_logits_file = None
    top_logits_line = 0
    if top_logits_path is not None:
        top_logits_path = Path(top_logits_path)
        top_logits_path.parent.mkdir(parents=True, exist_ok=True)
        if str(top_logits_path).endswith(".gz"):
            top_logits_file = gzip.open(top_logits_path, "wt", encoding="utf-8")
        else:
            top_logits_file = top_logits_path.open("w", encoding="utf-8")

    iterator = range(0, len(output_records), batch_size)
    if use_tqdm and tqdm is not None:
        iterator = tqdm(iterator, desc="vLLM probability probes")

    token_text_cache: dict[int, str] = {}
    try:
        for start in iterator:
            batch_records = output_records[start : start + batch_size]
            prompt_infos = [
                build_probability_probe_prompt(
                    record=record,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    max_token_input=max_token_input,
                    label_values=label_values,
                    answer_prefix=answer_prefix,
                    pred_field=pred_field,
                    examples=examples,
                    chat_kwargs=chat_kwargs,
                    probe_cut=probe_cut,
                )
                for record in batch_records
            ]
            outputs = model.generate(
                [info["prompt"] for info in prompt_infos],
                sampling_params=sampling_params,
                use_tqdm=False,
            )

            for offset, (record, prompt_info, output_one) in enumerate(
                zip(batch_records, prompt_infos, outputs)
            ):
                score = score_vllm_probe_output(
                    output_one=output_one,
                    tokenizer=tokenizer,
                    label_token_ids=label_token_ids,
                    label_groups=label_groups,
                    top_k_logits=int(top_k_logits),
                    token_text_cache=token_text_cache,
                )
                top_logits = score.pop("top_logits", None)
                gold_label = None
                if output_field in record:
                    try:
                        gold_label = extract_label_from_text(
                            str(record[output_field]),
                            label_values=label_values,
                            answer_prefix=answer_prefix,
                        )["label"]
                    except ValueError:
                        gold_label = None

                generated_token = score.get("probe_generated_token") or {}
                generated_token_id = generated_token.get("token_id")
                generated_token_text = generated_token.get("token_text")
                pred_target_token_id = prompt_info.get("pred_target_token_id")
                pred_target_token_text = prompt_info.get("pred_target_token_text")
                exact_token_match = (
                    pred_target_token_id is not None
                    and generated_token_id == pred_target_token_id
                )
                text_match = (
                    pred_target_token_text is not None
                    and generated_token_text == pred_target_token_text
                )
                stripped_label_match = (
                    str(generated_token_text or "").strip()
                    == str(prompt_info["pred_label"]).strip()
                )

                top_logits_ref = None
                if top_logits_file is not None and top_logits is not None:
                    top_logits_line += 1
                    top_payload = {
                        "line": top_logits_line,
                        "idx": start + offset,
                        "id": record.get("id"),
                        "pred": record.get(pred_field),
                        "output": record.get(output_field),
                        "probe_cut": probe_cut,
                        "pred_label": prompt_info["pred_label"],
                        "pred_prefix_before_target_token": prompt_info[
                            "pred_prefix_before_target_token"
                        ],
                        "pred_target_token_id": pred_target_token_id,
                        "pred_target_token_text": pred_target_token_text,
                        "probe_generated_token": generated_token,
                        "vocab_logsumexp": score.get("vocab_logsumexp"),
                        "top_k": int(top_k_logits),
                        "tokens": top_logits,
                    }
                    top_logits_file.write(
                        json.dumps(top_payload, ensure_ascii=False) + "\n"
                    )
                    top_logits_ref = {
                        "path": str(top_logits_path),
                        "format": "jsonl.gz" if str(top_logits_path).endswith(".gz") else "jsonl",
                        "line": top_logits_line,
                        "top_k": int(top_k_logits),
                    }

                record["logit"] = {
                    "mode": "one_token_label_probability_probe",
                    "answer_prefix": answer_prefix,
                    "probe_cut": probe_cut,
                    "pred_label": prompt_info["pred_label"],
                    "gold_label": gold_label,
                    "pred_prefix_before_label": prompt_info["pred_prefix_before_label"],
                    "pred_prefix_before_target_token": prompt_info[
                        "pred_prefix_before_target_token"
                    ],
                    "pred_label_start": prompt_info["pred_label_start"],
                    "pred_label_end": prompt_info["pred_label_end"],
                    "pred_target_token_id": pred_target_token_id,
                    "pred_target_token_text": pred_target_token_text,
                    "pred_target_token_index": prompt_info["pred_target_token_index"],
                    "pred_target_token_start": prompt_info["pred_target_token_start"],
                    "pred_target_token_end": prompt_info["pred_target_token_end"],
                    "pred_target_token_prefix": prompt_info["pred_target_token_prefix"],
                    "pred_target_token_label_text": prompt_info[
                        "pred_target_token_label_text"
                    ],
                    "observed_label_token_prefixes": list(
                        observed_label_token_prefixes
                    ),
                    "label_token_prefixes": list(active_label_token_prefixes),
                    "label_token_ids": {
                        label: [int(token_id) for token_id in token_ids]
                        for label, token_ids in label_token_ids.items()
                    },
                    "probe_consistency": {
                        "exact_token_match": bool(exact_token_match),
                        "token_text_match": bool(text_match),
                        "stripped_label_match": bool(stripped_label_match),
                    },
                    "sampling": {
                        "seed": int(seed) if seed is not None else None,
                        "temperature": 0.0,
                        "max_tokens": 1,
                        "logprobs": -1,
                    },
                    **score,
                }
                if top_logits_ref is not None:
                    record["logit"]["top_logits_ref"] = top_logits_ref
                elif top_logits is not None:
                    record["logit"]["top_logits"] = top_logits
    finally:
        if top_logits_file is not None:
            top_logits_file.close()

    return output_records


def add_vllm_label_probabilities_to_file(
    path_result: str | Path,
    path_output: str | Path,
    model,
    tokenizer,
    model_name: str,
    max_token_input: int,
    label_values: Sequence[str],
    label_groups: Mapping[str, Sequence[str]] = DEFAULT_LABEL_GROUPS,
    **kwargs,
) -> list[dict[str, Any]]:
    """Load a result JSON, add ``logit`` blocks, write the augmented JSON."""
    records = load_json(path_result)
    if not isinstance(records, list):
        raise ValueError(f"Result file must contain a list of records: {path_result}")
    scored_records = add_vllm_label_probabilities(
        records=records,
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        max_token_input=max_token_input,
        label_values=label_values,
        label_groups=label_groups,
        **kwargs,
    )
    write_json(scored_records, path_output, indent=2)
    return scored_records


def probability_rows_from_records(
    records: Sequence[Mapping[str, Any]],
    positive_group: str = "positive",
    negative_group: str = "negative",
    positive_labels: Sequence[str] | None = None,
    score_name: str = "conditional_probability",
) -> list[dict[str, Any]]:
    """Flatten augmented result records into rows for metric calculation."""
    rows = []
    positive_label_set = set(str(label) for label in positive_labels or [])
    for idx, record in enumerate(records):
        logit = record.get("logit") or {}
        groups = logit.get("groups") or {}
        if positive_group not in groups:
            continue

        gold_label = logit.get("gold_label")
        if gold_label is None and "output" in record:
            # Leave gold unavailable if the caller used a custom output format
            # without storing extraction metadata in record["logit"].
            gold_label = None
        pred_label = logit.get("pred_label")
        if not positive_label_set:
            positive_label_set = set(groups[positive_group].get("labels", []))
        y_true = None if gold_label is None else int(str(gold_label) in positive_label_set)

        rows.append(
            {
                "idx": idx,
                "id": record.get("id"),
                "gold_label": gold_label,
                "pred_label": pred_label,
                "y_true": y_true,
                "score": float(groups[positive_group][score_name]),
                "positive_score": float(groups[positive_group][score_name]),
                "negative_score": (
                    float(groups[negative_group][score_name])
                    if negative_group in groups
                    else None
                ),
                "positive_group": positive_group,
                "negative_group": negative_group,
                "score_name": score_name,
                "pred": record.get("pred"),
                "output": record.get("output"),
            }
        )
    return rows


def compute_binary_predictions_from_scores(
    rows: Sequence[Mapping[str, Any]],
    threshold: float | None = None,
    positive_gt_negative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``y_true``, ``y_score``, and ``y_pred`` for a binary rule."""
    valid_rows = [row for row in rows if row.get("y_true") is not None]
    if not valid_rows:
        raise ValueError("No rows with y_true are available for metric calculation.")

    y_true = np.array([int(row["y_true"]) for row in valid_rows], dtype=int)
    y_score = np.array([float(row["score"]) for row in valid_rows], dtype=float)

    if positive_gt_negative:
        y_pred = np.array(
            [
                int(float(row["positive_score"]) > float(row["negative_score"]))
                for row in valid_rows
            ],
            dtype=int,
        )
    else:
        if threshold is None:
            raise ValueError("threshold must be set unless positive_gt_negative=True.")
        y_pred = (y_score >= float(threshold)).astype(int)
    return y_true, y_score, y_pred


def compute_binary_metrics_from_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Compute the standard binary metrics used in probability reports."""
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0
    specificity_neg = tn / (tn + fp) if (tn + fp) else 0.0
    f1_pos = (
        2 * ppv * recall_pos / (ppv + recall_pos)
        if (ppv + recall_pos)
        else 0.0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        labels=[0, 1],
        zero_division=0,
    )
    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            labels=[0, 1],
            zero_division=0,
        )
    )

    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1 (Pos)": float(f1_pos),
        "PPV": float(ppv),
        "NPV": float(npv),
        "Recall (Pos)": float(recall_pos),
        "Specificity (Neg)": float(specificity_neg),
        "F1 (Macro)": float(f1_macro),
        "Precision (Macro)": float(precision_macro),
        "Recall (Macro)": float(recall_macro),
        "F1 (Weighted)": float(f1_weighted),
        "Precision (Weighted)": float(precision_weighted),
        "Recall (Weighted)": float(recall_weighted),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def compute_binary_metrics_for_rule(
    rows: Sequence[Mapping[str, Any]],
    threshold: float | None = None,
    positive_gt_negative: bool = False,
) -> dict[str, Any]:
    """Compute requested binary metrics for one threshold/rule."""
    y_true, y_score, y_pred = compute_binary_predictions_from_scores(
        rows,
        threshold=threshold,
        positive_gt_negative=positive_gt_negative,
    )
    metrics = compute_binary_metrics_from_predictions(y_true, y_pred)
    metrics["n"] = int(len(y_true))
    metrics["threshold"] = None if threshold is None else float(threshold)
    metrics["rule"] = "positive>negative" if positive_gt_negative else "threshold"
    return metrics


def sweep_binary_probability_metrics(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
    include_positive_gt_negative: bool = True,
) -> list[dict[str, Any]]:
    """Compute report metrics for many thresholds plus optional pos>neg rule."""
    results = [
        compute_binary_metrics_for_rule(rows, threshold=threshold)
        for threshold in thresholds
    ]
    if include_positive_gt_negative:
        results.append(
            compute_binary_metrics_for_rule(rows, positive_gt_negative=True)
        )
    return results


def compute_binary_probability_metrics(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float] = (0.5,),
) -> dict[str, Any]:
    """
    Compute AUROC/AUPRC and threshold metrics from probability rows.

    PPV is precision. NPV is TN / (TN + FN). Recall is sensitivity.
    Specificity is TN / (TN + FP).
    """
    valid_rows = [row for row in rows if row.get("y_true") is not None]
    if not valid_rows:
        raise ValueError("No rows with y_true are available for metric calculation.")

    y_true = np.array([int(row["y_true"]) for row in valid_rows], dtype=int)
    y_score = np.array([float(row["score"]) for row in valid_rows], dtype=float)

    metrics: dict[str, Any] = {"n": int(len(valid_rows))}
    if len(set(y_true.tolist())) == 2:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    else:
        metrics["auroc"] = None
        metrics["auprc"] = None

    threshold_metrics = {}
    for threshold in thresholds:
        threshold = float(threshold)
        y_pred = (y_score >= threshold).astype(int)
        tp = int(((y_pred == 1) & (y_true == 1)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())

        ppv = tp / (tp + fp) if (tp + fp) else 0.0
        npv = tn / (tn + fn) if (tn + fn) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        specificity = tn / (tn + fp) if (tn + fp) else 0.0
        f1 = 2 * ppv * recall / (ppv + recall) if (ppv + recall) else 0.0

        threshold_metrics[str(threshold)] = {
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "ppv": ppv,
            "precision": ppv,
            "npv": npv,
            "recall": recall,
            "sensitivity": recall,
            "specificity": specificity,
            "f1": f1,
        }

    metrics["thresholds"] = threshold_metrics
    return metrics


def _valid_probability_rows(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    valid_rows = [row for row in rows if row.get("y_true") is not None]
    if not valid_rows:
        raise ValueError("No rows with y_true are available for metric calculation.")
    return valid_rows


def _probability_arrays(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    valid_rows = _valid_probability_rows(rows)
    y_true = np.array([int(row["y_true"]) for row in valid_rows], dtype=int)
    y_score = np.array([float(row["score"]) for row in valid_rows], dtype=float)
    positive_score = np.array(
        [float(row.get("positive_score", row["score"])) for row in valid_rows],
        dtype=float,
    )
    negative_score = np.array(
        [float(row.get("negative_score", 1.0 - row["score"])) for row in valid_rows],
        dtype=float,
    )
    return y_true, y_score, positive_score, negative_score


def _bootstrap_resample_indices(
    y_true: Sequence[int],
    rng: np.random.Generator,
    stratified: bool = True,
) -> np.ndarray:
    """Return one bootstrap sample of row indexes, preserving class counts by default."""
    y_true = np.array(y_true, dtype=int)
    n = len(y_true)
    if not stratified:
        return rng.integers(0, n, size=n)

    sampled = []
    for value in np.unique(y_true):
        class_idx = np.flatnonzero(y_true == value)
        if len(class_idx) == 0:
            continue
        sampled.append(rng.choice(class_idx, size=len(class_idx), replace=True))
    if not sampled:
        return rng.integers(0, n, size=n)
    out = np.concatenate(sampled)
    rng.shuffle(out)
    return out


def _safe_auroc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.array(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def _safe_auprc(y_true: Sequence[int], y_score: Sequence[float]) -> float:
    y_true = np.array(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def _ci_from_values(values: Sequence[float], alpha: float = 0.05) -> dict[str, float | int]:
    arr = np.array(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "bootstrap_n": 0,
            "bootstrap_mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }
    return {
        "bootstrap_n": int(arr.size),
        "bootstrap_mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
        "ci_low": float(np.quantile(arr, alpha / 2.0)),
        "ci_high": float(np.quantile(arr, 1.0 - alpha / 2.0)),
    }


def compute_probability_summary_with_ci(
    rows: Sequence[Mapping[str, Any]],
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stratified: bool = True,
) -> dict[str, Any]:
    """
    Compute AUROC/AUPRC/prevalence with nonparametric bootstrap CIs.

    The point estimate is always computed on the full sample. Bootstrap CIs are
    percentile intervals. Stratified resampling is the default so binary outcome
    prevalence and class availability remain stable across resamples.
    """
    y_true, y_score, _, _ = _probability_arrays(rows)
    point = {
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "negative_n": int(len(y_true) - y_true.sum()),
        "prevalence": float(y_true.mean()),
        "auroc": _safe_auroc(y_true, y_score),
        "auprc": _safe_auprc(y_true, y_score),
    }
    boot: dict[str, list[float]] = {key: [] for key in ["prevalence", "auroc", "auprc"]}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        idx = _bootstrap_resample_indices(y_true, rng, stratified=stratified)
        yt = y_true[idx]
        ys = y_score[idx]
        boot["prevalence"].append(float(yt.mean()))
        boot["auroc"].append(_safe_auroc(yt, ys))
        boot["auprc"].append(_safe_auprc(yt, ys))

    out = dict(point)
    for metric, values in boot.items():
        out[f"{metric}_point"] = point[metric]
        out[f"{metric}_ci_low"] = _ci_from_values(values, alpha=alpha)["ci_low"]
        out[f"{metric}_ci_high"] = _ci_from_values(values, alpha=alpha)["ci_high"]
        out[f"{metric}_bootstrap_mean"] = _ci_from_values(values, alpha=alpha)[
            "bootstrap_mean"
        ]
        out[f"{metric}_bootstrap_std"] = _ci_from_values(values, alpha=alpha)["std"]
    out["bootstrap_n"] = int(n_bootstrap)
    out["bootstrap_seed"] = int(seed)
    out["bootstrap_alpha"] = float(alpha)
    out["bootstrap_stratified"] = bool(stratified)
    return out


def compute_binary_rule_metrics_with_ci(
    rows: Sequence[Mapping[str, Any]],
    threshold: float | None = None,
    positive_gt_negative: bool = False,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stratified: bool = True,
) -> dict[str, Any]:
    """
    Compute fixed-rule binary metrics with bootstrap CIs.

    For threshold-based rows the threshold is fixed to the value supplied by the
    caller; this avoids reselecting a new threshold inside each bootstrap sample.
    """
    y_true, y_score, positive_score, negative_score = _probability_arrays(rows)
    point = compute_binary_metrics_for_rule(
        rows,
        threshold=threshold,
        positive_gt_negative=positive_gt_negative,
    )
    metric_values: dict[str, list[float]] = {
        metric: []
        for metric in [
            "Accuracy",
            "F1 (Pos)",
            "PPV",
            "NPV",
            "Recall (Pos)",
            "Specificity (Neg)",
            "F1 (Macro)",
            "Precision (Macro)",
            "Recall (Macro)",
            "F1 (Weighted)",
            "Precision (Weighted)",
            "Recall (Weighted)",
        ]
    }
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        idx = _bootstrap_resample_indices(y_true, rng, stratified=stratified)
        yt = y_true[idx]
        if positive_gt_negative:
            yp = (positive_score[idx] > negative_score[idx]).astype(int)
        else:
            if threshold is None:
                raise ValueError("threshold must be set unless positive_gt_negative=True.")
            yp = (y_score[idx] >= float(threshold)).astype(int)
        metrics = compute_binary_metrics_from_predictions(yt, yp)
        for metric in metric_values:
            metric_values[metric].append(float(metrics[metric]))

    out = dict(point)
    for metric, values in metric_values.items():
        ci = _ci_from_values(values, alpha=alpha)
        out[f"{metric} point"] = float(point[metric])
        out[f"{metric} ci_low"] = ci["ci_low"]
        out[f"{metric} ci_high"] = ci["ci_high"]
        out[f"{metric} bootstrap_mean"] = ci["bootstrap_mean"]
        out[f"{metric} bootstrap_std"] = ci["std"]
    out["bootstrap_n"] = int(n_bootstrap)
    out["bootstrap_seed"] = int(seed)
    out["bootstrap_alpha"] = float(alpha)
    out["bootstrap_stratified"] = bool(stratified)
    return out


def align_probability_rows_by_id(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    id_field: str = "id",
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Align two probability row sets by id, falling back to idx when ids are missing."""
    def key_for(row: Mapping[str, Any]) -> str:
        value = row.get(id_field)
        if value is None:
            value = row.get("idx")
        return str(value)

    map_a = {key_for(row): row for row in _valid_probability_rows(rows_a)}
    map_b = {key_for(row): row for row in _valid_probability_rows(rows_b)}
    keys = [key for key in map_a if key in map_b]
    if not keys:
        raise ValueError("No overlapping row ids are available for paired comparison.")

    aligned_a = []
    aligned_b = []
    for key in keys:
        row_a = map_a[key]
        row_b = map_b[key]
        if int(row_a["y_true"]) != int(row_b["y_true"]):
            raise ValueError(f"Aligned rows disagree on y_true for id={key}.")
        aligned_a.append(row_a)
        aligned_b.append(row_b)
    return aligned_a, aligned_b


def _score_metric_value(
    metric: str,
    y_true: Sequence[int],
    y_score: Sequence[float],
) -> float:
    metric = metric.lower()
    if metric == "auroc":
        return _safe_auroc(y_true, y_score)
    if metric == "auprc":
        return _safe_auprc(y_true, y_score)
    raise ValueError(f"Unsupported score metric for paired bootstrap: {metric}")


def paired_bootstrap_score_metric_difference(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    metric: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
    stratified: bool = True,
) -> dict[str, Any]:
    """
    Paired bootstrap difference for score metrics such as AUROC and AUPRC.

    The returned delta is ``metric(rows_b) - metric(rows_a)`` after aligning rows
    by sample id. The p value is a two-sided bootstrap sign probability and is
    intended as an interpretable screening statistic, not a DeLong test.
    """
    aligned_a, aligned_b = align_probability_rows_by_id(rows_a, rows_b)
    y_true = np.array([int(row["y_true"]) for row in aligned_a], dtype=int)
    score_a = np.array([float(row["score"]) for row in aligned_a], dtype=float)
    score_b = np.array([float(row["score"]) for row in aligned_b], dtype=float)
    point_a = _score_metric_value(metric, y_true, score_a)
    point_b = _score_metric_value(metric, y_true, score_b)
    point_delta = float(point_b - point_a)

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(int(n_bootstrap)):
        idx = _bootstrap_resample_indices(y_true, rng, stratified=stratified)
        delta = _score_metric_value(metric, y_true[idx], score_b[idx]) - _score_metric_value(
            metric, y_true[idx], score_a[idx]
        )
        deltas.append(float(delta))

    arr = np.array([value for value in deltas if np.isfinite(value)], dtype=float)
    if arr.size:
        p_low = (np.sum(arr <= 0.0) + 1.0) / (arr.size + 1.0)
        p_high = (np.sum(arr >= 0.0) + 1.0) / (arr.size + 1.0)
        p_two_sided = min(1.0, float(2.0 * min(p_low, p_high)))
    else:
        p_two_sided = float("nan")
    ci = _ci_from_values(arr, alpha=alpha)
    return {
        "metric": metric,
        "n": int(len(y_true)),
        "a": float(point_a),
        "b": float(point_b),
        "delta_b_minus_a": point_delta,
        "delta_ci_low": ci["ci_low"],
        "delta_ci_high": ci["ci_high"],
        "delta_bootstrap_mean": ci["bootstrap_mean"],
        "delta_bootstrap_std": ci["std"],
        "p_two_sided_bootstrap": p_two_sided,
        "bootstrap_n": int(n_bootstrap),
        "bootstrap_effective_n": ci["bootstrap_n"],
        "bootstrap_seed": int(seed),
        "bootstrap_alpha": float(alpha),
        "bootstrap_stratified": bool(stratified),
    }


def _compute_midrank(values: Sequence[float]) -> np.ndarray:
    """Return one-based midranks for DeLong AUROC covariance calculation."""
    arr = np.asarray(values, dtype=float)
    order = np.argsort(arr)
    sorted_arr = arr[order]
    ranks = np.empty(len(arr), dtype=float)
    i = 0
    while i < len(arr):
        j = i
        while j < len(arr) and sorted_arr[j] == sorted_arr[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    out = np.empty(len(arr), dtype=float)
    out[order] = ranks
    return out


def _fast_delong(
    y_true: Sequence[int],
    predictions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute correlated AUROC estimates and DeLong covariance.

    ``predictions`` is shaped ``(n_classifiers, n_samples)`` and y_true must be
    binary. Implementation follows the standard fast DeLong midrank algorithm.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    if predictions.ndim != 2:
        raise ValueError("predictions must be a 2D array.")
    if predictions.shape[1] != y_true_arr.size:
        raise ValueError("predictions and y_true must contain the same number of samples.")
    labels = np.unique(y_true_arr)
    if labels.size != 2:
        raise ValueError("DeLong AUROC comparison requires exactly two outcome classes.")

    order = np.argsort(-y_true_arr)
    label_1_count = int(np.sum(y_true_arr == 1))
    if label_1_count == 0 or label_1_count == y_true_arr.size:
        raise ValueError("DeLong AUROC comparison requires positive and negative samples.")
    m = label_1_count
    n = y_true_arr.size - m
    preds = np.asarray(predictions, dtype=float)[:, order]
    k = preds.shape[0]

    pos = preds[:, :m]
    neg = preds[:, m:]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds[r, :])

    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    covariance = sx / m + sy / n
    return aucs, covariance


def paired_delong_auroc_difference(
    rows_a: Sequence[Mapping[str, Any]],
    rows_b: Sequence[Mapping[str, Any]],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Compare two correlated AUROCs with DeLong's test after aligning by sample id.

    The returned delta is ``AUROC(rows_b) - AUROC(rows_a)``.
    """
    aligned_a, aligned_b = align_probability_rows_by_id(rows_a, rows_b)
    y_true = np.array([int(row["y_true"]) for row in aligned_a], dtype=int)
    score_a = np.array([float(row["score"]) for row in aligned_a], dtype=float)
    score_b = np.array([float(row["score"]) for row in aligned_b], dtype=float)
    aucs, covariance = _fast_delong(y_true, np.vstack([score_a, score_b]))
    point_a = float(aucs[0])
    point_b = float(aucs[1])
    point_delta = point_b - point_a

    contrast = np.array([-1.0, 1.0])
    variance = float(contrast @ covariance @ contrast.T)
    if variance > 0.0 and math.isfinite(variance):
        se = math.sqrt(variance)
        z_value = point_delta / se
        p_value = math.erfc(abs(z_value) / math.sqrt(2.0))
    else:
        se = 0.0
        z_value = float("inf") if point_delta != 0.0 else 0.0
        p_value = 0.0 if point_delta != 0.0 else 1.0

    z_crit = 1.959963984540054
    return {
        "metric": "auroc",
        "n": int(len(y_true)),
        "positive_n": int(y_true.sum()),
        "negative_n": int(len(y_true) - y_true.sum()),
        "a": point_a,
        "b": point_b,
        "delta_b_minus_a": point_delta,
        "delta_ci_low": point_delta - z_crit * se,
        "delta_ci_high": point_delta + z_crit * se,
        "delong_se": se,
        "delong_z": z_value,
        "p_two_sided_delong": p_value,
        "test": "DeLong",
        "alpha": float(alpha),
    }
