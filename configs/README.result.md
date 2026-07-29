# Result JSON Format

This document describes the ordinary inference result files written by
`clinical-llm-benchmark` for local models and commercial API providers.

Logits-specific files are described separately in `README.result.logits.md`.

## File Location

For both local inference and API inference, one experiment writes one primary
result plus adjacent reproducibility artifacts:

```text
<dir_result>/<task_name>/<model_name>/<task_name>-<prompt_mode>-<decoding_strategy>-<seed>.result.json
<dir_result>/<task_name>/<model_name>/<task_name>-<prompt_mode>-<decoding_strategy>-<seed>.result.config.yaml
<dir_result>/<task_name>/<model_name>/<task_name>-<prompt_mode>-<decoding_strategy>-<seed>.result.metadata.json
```

Example:

```text
result/MGB-Med-Class.highlight/Qwen3-30B/MGB-Med-Class.highlight-direct-greedy-42.result.json
```

The path components come from runtime config:

| Component | Source |
| --- | --- |
| `dir_result` | Local: `dir_result` / `dir_result_save` in `configs/BRIDGE.yaml`; API: `dir_result` in `configs/API.provider.<provider>.yaml`. |
| `task_name` | Current task under `tasks`. |
| `model_name` | CLI `--model_name` for local runs, or selected API model name. |
| `prompt_mode` | One experiment row, e.g. `direct`, `cot`, `direct-5-shot`. |
| `decoding_strategy` | One experiment row, usually `greedy` or `sampling`. |
| `seed` | One experiment row. |

## Local vLLM/HF Result Rows

Local inference uses `model.inference.save_result`.

When `save_input: true`, each row keeps the source metadata but replaces
`input` with only the current sample text retained after context truncation,
then adds `example` and `pred`. The system prompt remains in `instruction`.
Typical fields:

```json
{
  "id": "sample-1",
  "task": "ExampleTask",
  "type": "classification",
  "language": "en",
  "split": "test",
  "instruction": "...",
  "input": "truncated current-sample text...",
  "output": "Assessment: Yes",
  "example": ["example-1", "example-2", "example-3"],
  "pred": "Assessment: No"
}
```

The exact source fields depend on the dataset JSON. Inference adds:

| Field | Meaning |
| --- | --- |
| `input` | Current sample text after context truncation; it does not include the system prompt or few-shot examples. |
| `example` | Ordered IDs of the few-shot examples actually included after context budgeting. It is `[]` for zero-shot and may be shorter than the requested shot count. |
| `pred` | Model-generated text. For greedy decoding this is a string; for sampling with multiple samples it can be a list of strings. |

When `save_input: false`, local inference writes a compact row:

```json
{
  "id": "sample-1",
  "type": "classification",
  "language": "en",
  "output": "Assessment: Yes",
  "example": ["example-1", "example-2", "example-3"],
  "pred": "Assessment: No"
}
```

This reduces file size but removes prompt/source fields that may be useful for
debugging.

## Commercial API Result Rows

API inference uses `model.api_runner.result_rows_from_query_results`.

When `save_input: true`, `input` is replaced by the exact, possibly truncated
chat messages sent to the provider:

```json
{
  "id": "sample-1",
  "task": "ExampleTask",
  "type": "classification",
  "language": "en",
  "split": "test",
  "input": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "output": "Assessment: Yes",
  "example": ["example-1", "example-2", "example-3"],
  "pred": "Assessment: No",
  "error": "",
  "token_input": 1024,
  "token_output": 12
}
```

API-only fields:

| Field | Meaning |
| --- | --- |
| `input` | Exact request messages after context budgeting and input truncation. |
| `error` | Empty string for a successful sample; otherwise the provider/query error. |
| `summary` | Optional provider reasoning/summary text when available. |
| `token_input` | Provider-reported input token count when available. |
| `token_output` | Provider-reported output token count when available. |

When `save_input: false`, API result rows keep only compact metadata:

```json
{
  "task": "ExampleTask",
  "id": "sample-1",
  "type": "classification",
  "language": "en",
  "split": "test",
  "output": "Assessment: Yes",
  "example": ["example-1", "example-2", "example-3"],
  "pred": "Assessment: No",
  "error": ""
}
```

## Export Controls

### Local Inference

Local result export is controlled by `configs/BRIDGE.yaml` or another run YAML:

```yaml
save_input: true
save_every_n_batches: 20
dir_result: /path/to/result
```

| Field | Effect |
| --- | --- |
| `save_input` | `true` saves full source rows plus `example` and `pred`; `false` writes compact rows while retaining `example`. |
| `save_every_n_batches` | Appends newly completed rows to one `*.result.partial.jsonl` checkpoint during long vLLM runs. The final JSON files are written once. |
| `dir_result` | Result root. New inference writes to the first path if multiple are supplied. |

`main.py` internally sets:

```text
path_dir_result = <dir_result[0]>/<task_name>/<model_name>
name_exp = <task_name>-<prompt_mode>-<decoding_strategy>-<seed>
path_file_result = <path_dir_result>/<name_exp>.result.json
```

Every completed local or API experiment also writes:

```text
<name_exp>.result.config.yaml
<name_exp>.result.metadata.json
```

The YAML is the effective per-experiment configuration (API secrets are
redacted). The JSON records source/artifact hashes, task/model/decoding
identifiers, Git state, command, and core package versions. The internal
`*.result.partial.jsonl` is removed after the final result is safely written;
set `resume_partial_results: true` to resume an interrupted vLLM run.

### Commercial API Inference

API result export is controlled by `configs/API.provider.<provider>.yaml`:

```yaml
save_input: true
save_trace: true
save_raw_response: false
allow_partial: false
evaluate_on_partial: false
dir_result:
  - /path/to/result
```

| Field | Effect |
| --- | --- |
| `save_input` | `true` saves source metadata and places the exact request messages in `input`; `false` writes compact rows. Both modes retain actual example IDs. |
| `save_trace` | Writes provider request/response trace files under `api_trace/`. |
| `save_raw_response` | Includes raw provider responses in trace files. Useful for debugging, but can be large. |
| `allow_partial` | Allows writing partial result files when some sample-level API calls fail. |
| `evaluate_on_partial` | Allows automatic evaluation of partial results when `allow_partial` is true. |
| `dir_result` | Result root. API runs write to the first listed root. |

Trace files are written under:

```text
<dir_result>/<task_name>/<model_name>/api_trace/<task_name>-<prompt_mode>-<decoding_strategy>-<seed>/
```

Common trace files:

| File | Meaning |
| --- | --- |
| `run_config.json` | Redacted run metadata, provider settings, task/model identifiers, and export flags. |
| `requests.jsonl` | Exact API requests for single-query mode. |
| `responses.jsonl` | Normalized provider responses. |
| Azure batch artifacts | Batch input/status/output files when Azure batch mode is used and tracing is enabled. |
| `error_summary.json` | Written when one or more samples fail. |

## Evaluation Expectations

The evaluation code expects result rows to align with the task dataset by row
order and/or sample id. It reads:

| Field | Required For |
| --- | --- |
| `id` | Sample alignment and auditing. |
| `output` | Gold answer extraction. |
| `pred` | Prediction extraction. |
| `type`, `language`, `split` | Optional grouping/filtering and metadata. |

Partial result files are allowed for quick tests, but production evaluation
should use complete result files unless the report explicitly states that only a
subset was evaluated.

## Related Documents

- `README.result.logits.md`: logits/probability result sidecars.
- `README.logits.md`: logits YAML parameter reference.
- `README.result.logits.md`: logits/probability sidecar result format.
- `USAGE.md`: full runtime workflow.
