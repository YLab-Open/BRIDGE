# Logits Result Files

This document describes the sidecar files written by local vLLM inference when
`main.py --logits_config <logits.yaml>` is enabled.

Ordinary results are described in [README.result.md](README.result.md). Logits
configuration is described in [README.logits.md](README.logits.md).

## File Set

For an ordinary result path:

```text
<task>-<prompt>-<decoding>-<seed>.result.json
```

logits capture writes adjacent sidecars:

```text
<task>-<prompt>-<decoding>-<seed>.result.logit.json
<task>-<prompt>-<decoding>-<seed>.result.logit.top_logits.jsonl.gz
<task>-<prompt>-<decoding>-<seed>.result.metadata.json
<task>-<prompt>-<decoding>-<seed>.result.config.yaml
```

The normal `.result.json` remains lightweight. It does not inline `generation`
or `logit` blocks.

## `*.result.logit.json`

Compact per-sample logit/probability rows:

```json
{
  "idx": 0,
  "id": "sample-1",
  "task": "C0500_BIMS_SCRE_NUM.dynamic.history_all_note_latest.summary",
  "type": "classification",
  "language": "en",
  "split": "test",
  "output": "Assessment: Yes",
  "pred": "Assessment: No",
  "generation": {},
  "logit": {}
}
```

`generation` is present when `generation_trace.enabled: true`.
`logit` is present when `label_probability.enabled: true` and the task has a
matching `tasks.<task_name>` entry.

## `generation`

Generation trace records what vLLM actually produced:

```json
{
  "request_id": "0",
  "prompt_length": 12000,
  "prompt_hash": "...",
  "outputs": [
    {
      "index": 0,
      "text": "Assessment: Yes",
      "finish_reason": "stop",
      "stop_reason": null,
      "cumulative_logprob": null,
      "token_ids": [123, 456],
      "token_texts": ["Assessment:", " Yes"]
    }
  ]
}
```

Token ids and token texts are always saved when generation trace is enabled.
`outputs[].top_logprobs` appears only when
`generation_trace.top_k_logprobs_for_all_output_tokens > 0`.

## `logit`

For one greedy output, the probability block usually looks like:

```json
{
  "mode": "generation_logprobs",
  "source": "generation_logprobs",
  "answer_prefix": "Assessment:",
  "pred_label": "Yes",
  "pred_prefix_before_label": "Assessment: ",
  "pred_prefix_before_target_token": "Assessment:",
  "pred_target_token_id": 22032,
  "pred_target_token_text": " Yes",
  "pred_target_token_prefix": " ",
  "label_token_ids": {
    "No": [3501],
    "Yes": [13830]
  },
  "labels": {},
  "groups": {},
  "top_logits_ref": {}
}
```

Important fields:

| Field | Meaning |
| --- | --- |
| `mode` / `source` | `generation_logprobs`, `exact_output_token_probe`, or `exact_output_token_probe_fallback`. |
| `pred_label` | Parsed generated label. |
| `pred_target_token_text` | Actual generated token containing the label, e.g. `" Yes"` or `"1"`. |
| `label_token_ids` | Token ids scored for each configured label. |
| `labels` | Per-label token rows, raw logits, logprobs, and probabilities. |
| `groups` | Per-group scores and conditional probabilities. |
| `generation_consistency` / `probe_consistency` | Audits that the logits position matches the generated label token. |
| `top_logits_ref` | Pointer to the top-logits sidecar row. |
| `fallback_from_generation_logprobs_errors` | Original top-K error when fallback was used. |
| `error` | Unresolved error when `fail_on_error: false`. |

For sampling with multiple returned outputs, the block can be:

```json
{
  "mode": "multi_output_label_probability",
  "outputs": [
    {"mode": "generation_logprobs", "pred_label": "Yes"},
    {"mode": "generation_logprobs", "pred_label": "No"}
  ]
}
```

## Label Position

The code uses the generated token ids rather than text splitting alone:

1. Find `answer_prefix`, usually `Assessment:`.
2. Parse one configured label after the prefix and optional whitespace.
3. Reconstruct spans from actual generated token ids.
4. Select the token whose span contains the label text.

Therefore:

```text
Assessment: Yes
```

may target token `" Yes"`, while:

```text
Assessment: 1
```

may target token `"1"`.

## Probability Calculation

The code aggregates token variants with `logsumexp`.

For MDS:

```text
negative = label_logit("No")
positive = label_logit("Yes")
P(positive | labels) = exp(positive) / (exp(positive) + exp(negative))
```

For OASIS:

```text
negative = logsumexp(label_logit("0"), label_logit("1"))
positive = logsumexp(label_logit("2"), label_logit("3"), label_logit("4"))
```

Use `groups.<group>.conditional_probability` for binary threshold metrics.

## `*.result.logit.top_logits.jsonl.gz`

Compressed JSONL sidecar for label-position audit. Each line corresponds to one
sample/output and contains:

```json
{
  "line": 1,
  "idx": 0,
  "output_idx": 0,
  "source": "generation_logprobs",
  "pred_label": "Yes",
  "vocab_logsumexp": 123.4,
  "top_logits": [
    {
      "rank": 1,
      "token_id": 13830,
      "token_text": " Yes",
      "raw_logit": 12.3,
      "logprob": -0.2,
      "probability": 0.8,
      "selection": "top_k"
    }
  ]
}
```

If a scored label token is outside the saved top-K but present in returned
logits, it is appended with `selection: "label_token"`.

## `*.result.metadata.json`

Run-level reproducibility metadata shared by ordinary and logits inference. It includes:

- result/logit/top-logit paths;
- model name and model path;
- task, prompt mode, decoding strategy, seed, token limits;
- effective logits config and task label config;
- command-line argv and cwd;
- a separate `*.result.config.yaml` effective-config snapshot;
- result/config/model-registry/task-data/logit artifact paths and hashes;
- git commit/branch/dirty status;
- Python, torch, vLLM, and transformers versions when available.

API snapshots are redacted. This file replaces the older
`*.result.logit.metadata.json` name.

## Related Docs

- [README.logits.md](README.logits.md)
