# Logits YAML Configuration Guide

This document describes the `--logits_config` YAML used by local vLLM inference.
The canonical template is [logits.example.yaml](logits.example.yaml).

Logit outputs are always written as sidecar files. The ordinary
`*.result.json` remains compatible with existing evaluation code.

## Minimal Recommended Config

```yaml
enabled: true

vllm:
  max_logprobs: -1
  logprobs_mode: raw_logits

generation_trace:
  enabled: true
  top_k_logprobs_for_all_output_tokens: 0

label_probability:
  enabled: true
  mode: generation_logprobs
  generation_logprobs_top_k: 5000
  fallback_to_exact_probe: true
  probe_batch_size: 64
  top_k_logits: 1000
  save_top_logits: true
  label_token_prefixes: auto
  extra_label_token_prefixes: []
  scan_decoded_vocab: true
  fail_on_error: true
  probe_prompt_mode: token_ids
  model_label_token_id_overrides: {}

tasks:
  ExampleTaskName:
    answer_prefix: "Assessment:"
    label_values: ["No", "Yes"]
    label_groups:
      negative: ["No"]
      positive: ["Yes"]
    manifest_label_token_prefixes: ["", " "]
```

## Top-Level Keys

| Key | Default | Meaning |
| --- | --- | --- |
| `enabled` | `false` | Enables logits/probability capture. |
| `vllm` | see below | vLLM initialization settings required for logits. |
| `generation_trace` | see below | Saves exact generated token ids/texts and optional per-output-token top-K values. |
| `label_probability` | see below | Computes configured label/group probabilities. |
| `tasks` | `{}` | Task-specific answer prefix, label values, and label groups. |

There is no `output` block. Logit data are always saved separately:

```text
<task>-<prompt>-<decoding>-<seed>.result.json
<task>-<prompt>-<decoding>-<seed>.result.logit.json
<task>-<prompt>-<decoding>-<seed>.result.logit.top_logits.jsonl.gz
<task>-<prompt>-<decoding>-<seed>.result.metadata.json
<task>-<prompt>-<decoding>-<seed>.result.config.yaml
```

## `vllm`

```yaml
vllm:
  max_logprobs: -1
  logprobs_mode: raw_logits
```

`max_logprobs: -1` lets vLLM accept whole-vocabulary requests when exact
fallback probes need `SamplingParams(logprobs=-1)`.

`logprobs_mode: raw_logits` returns pre-softmax raw logits. The probability code
then applies `logsumexp` and computes label/group probabilities from those raw
values.

## `generation_trace`

```yaml
generation_trace:
  enabled: true
  top_k_logprobs_for_all_output_tokens: 0
```

When enabled, every row in `*.result.logit.json` stores an exact generation
trace:

- formatted prompt hash;
- generated token ids;
- generated token texts;
- finish/stop reason;
- optional top-K values for every generated output token.

`top_k_logprobs_for_all_output_tokens` is different from
`label_probability.top_k_logits`.

- `top_k_logprobs_for_all_output_tokens`: saves top-K values for every generated
  output token. Default `0` keeps files small.
- `label_probability.top_k_logits`: saves top-K values only at the task label
  position, for audit.

## `label_probability`

```yaml
label_probability:
  enabled: true
  mode: generation_logprobs
  generation_logprobs_top_k: 5000
  fallback_to_exact_probe: true
  probe_batch_size: 64
  top_k_logits: 1000
  save_top_logits: true
  label_token_prefixes: auto
  extra_label_token_prefixes: []
  scan_decoded_vocab: true
  fail_on_error: true
  probe_prompt_mode: token_ids
  model_label_token_id_overrides: {}
```

### `mode`

Supported values:

- `generation_logprobs`: score from logits returned during the original
  generation pass.
- `exact_output_token_probe`: after generation, rebuild the exact prompt prefix
  before the generated label token and request one-token whole-vocabulary logits.

Recommended default:

```yaml
mode: generation_logprobs
generation_logprobs_top_k: 5000
fallback_to_exact_probe: true
```

This is efficient for most samples and still robust. If a configured label token
is missing from the returned top-K, fallback re-probes that one label position
with whole-vocabulary logits.

### `generation_logprobs_top_k`

Controls the `SamplingParams(logprobs=K)` value used during the main generation
pass when `mode: generation_logprobs`.

`5000` is the recommended default. It usually captures label tokens while being
much lighter than returning the whole vocabulary for every generated token.

### `top_k_logits`

Controls how many high-logit tokens are written to
`*.result.logit.top_logits.jsonl.gz` for the label position.

If a scored label token is not inside the saved top-K but is present in the
returned logits, it is appended to the sidecar row with
`selection: "label_token"`. Therefore the sidecar may contain `top_k + x` rows.

### `scan_decoded_vocab`

When true, the tokenizer vocabulary is scanned for one-token decoded variants
that exactly match configured label strings such as `"Yes"`, `" Yes"`, `"0"`,
or `" 0"`. Keep this enabled for robust tokenizer handling.

### `fail_on_error`

When true, unresolved probability errors fail the run. With
`fallback_to_exact_probe: true`, generation top-K errors are first sent through
fallback; the run fails only if fallback also fails.

### `probe_prompt_mode`

Supported values:

- `token_ids`: rebuild the exact prompt prefix using generated token ids.
- `text`: rebuild with decoded text.

Use `token_ids` for strict token-boundary reproduction.

### `model_label_token_id_overrides`

Advanced escape hatch for tokenizer/model edge cases. Overrides add token ids to
automatically discovered label token ids; they do not replace auto discovery and
do not affect generation.

```yaml
model_label_token_id_overrides:
  Mistral-Small-3.2-24B-Instruct-2506:
    prefixes:
      " ":
        "No": [3501]
        "Yes": [13830]
```

## `tasks`

Each task entry must match `args.task_name`.

```yaml
tasks:
  M1700_COG_FUNCTION.dynamic.history_all_note_latest.summary:
    answer_prefix: "Assessment:"
    label_values: ["0", "1", "2", "3", "4"]
    label_groups:
      negative: ["0", "1"]
      positive: ["2", "3", "4"]
    manifest_label_token_prefixes: ["", " "]
```

Use `answer_prefix: "Assessment:"` rather than `"Assessment: "`. The code
finds the answer prefix, skips whitespace, parses the configured label, and then
uses the actual generated token ids to identify the label token boundary.

## Probability Computation

For each label, all configured single-token variants are aggregated with
`logsumexp`. Groups are then aggregated the same way.

For OASIS:

```text
negative = logsumexp(label_logit("0"), label_logit("1"))
positive = logsumexp(label_logit("2"), label_logit("3"), label_logit("4"))
P(positive | labels) = exp(positive) / (exp(positive) + exp(negative))
```

The saved `groups.*.conditional_probability` values are the recommended scores
for threshold-based metrics.

## Related Docs

- [README.result.logits.md](README.result.logits.md)
- [README.result.logits.md](README.result.logits.md)
