# Commercial API Inference And Evaluation

This document describes the unified commercial API workflow for Azure OpenAI,
Claude, and Gemini. These providers write the same `*.result.json` format as
local inference, so the shared evaluation runner can be reused.

## 1. Configuration Roles

Recommended separation:

- YAML defines experiments: `provider`, `models`, `tasks`, `experiments`, data
  paths, result paths, provider parameters, and automatic evaluation options.
- `run_API.sh` launches the run: choose a YAML, optionally override provider or
  model, run in foreground/background, and manage log paths.
- Python and notebooks should call `model.api_runner.run_api_from_config`
  directly instead of duplicating shell logic.

The provider model registry is:

```text
configs/API.model.yaml
```

It stores provider-side model names, deployments, reasoning support, and Azure
batch support. Credentials live in:

```text
configs/API.key.yaml
```

or provider-specific environment variables.

Implementation structure:

- `model.api_runner`: stable public Python entry point.
- `model/api/config.py`: config loading, registry lookup, and provider settings.
- `model/api/prompts.py`: task loading and prompt/message construction.
- `model/api/provider/`: Azure, Claude, and Gemini provider implementations.
- `model/api/results.py`: normalized result objects.

## 2. Shell Entry Point

Default run:

```bash
bash run_API.sh
```

Common debugging commands:

```bash
bash run_API.sh --dry-run
bash run_API.sh --provider claude --config configs/API.provider.claude.yaml --model claude-sonnet-4-5 --foreground
bash run_API.sh --provider gemini --config configs/API.provider.gemini.yaml --no-evaluate --foreground
```

Background runs write stdout/stderr to `log_API/`. `--dry-run` forces foreground
execution and validates the run config, provider model registry, experiment
matrix, and planned result paths without requiring task data or credentials,
calling an API, or writing result/trace files.

Provider defaults:

```text
azure  -> configs/API.provider.azure.yaml
claude -> configs/API.provider.claude.yaml
gemini -> configs/API.provider.gemini.yaml
```

## 3. Python / Notebook Entry Point

```python
from model.api_runner import run_api_from_config

run = run_api_from_config(
    config_path="configs/API.provider.azure.yaml",
    provider="azure",          # optional; defaults to YAML provider
    model_name="gpt-4o",       # optional; defaults to YAML models
    dry_run=True,
)

run.result_paths
```

You can also load and modify a config dictionary in a notebook:

```python
from util.runtime import load_yaml_config
from model.api_runner import run_api_from_config

cfg = load_yaml_config("configs/API.provider.azure.yaml")
cfg["tasks"] = ["BrainMRI-AIS"]
cfg["models"] = ["gpt-4o"]

run = run_api_from_config(
    config_path="configs/API.provider.azure.yaml",
    run_cfg=cfg,
)
```

The return value is `model.api.results.ApiRunResult`:

```python
run.provider       # azure / claude / gemini
run.model_names    # models actually run
run.result_paths   # written result.json paths
run.num_errors     # total sample-level errors across experiments
```

## 4. New Task Workflow

Assume a new task called `My-Clinical-Task` and a comparison between a local LLM
and Azure GPT-5.

1. Prepare data:

```text
dataset_raw/My-Clinical-Task.SFT.json
dataset_raw/example/My-Clinical-Task.example.json
dataset_raw/example_cot/My-Clinical-Task.example.json
```

2. Local LLM: set `tasks`, `dir_dataset`, and `dir_result` in
   `configs/BRIDGE.yaml`, then run:

```bash
python main.py --model_name gpt-oss-20b --engine vllm --config configs/BRIDGE.yaml
```

3. Azure GPT-5: register the deployment in `configs/API.model.yaml`:

```yaml
azure:
  gpt-5:
    deploy_name: gpt-5
    support_reasoning: true
    support_batch: true
    mode: response
```

Then set the run YAML, for example `configs/API.provider.azure.yaml`:

```yaml
provider: azure
models:
  - gpt-5
tasks:
  - My-Clinical-Task
experiments:
  - [direct, greedy, 42]
evaluate_after_run: true
```

Built-in BRIDGE tasks do not need `custom_tasks`. Project-specific tasks should
register their evaluation class:

```yaml
task_types:
  - ext
custom_tasks:
  ext:
    My-Clinical-Task: my_project.tasks:Task_ext_MyClinicalTask
```

Dry-run first:

```bash
bash run_API.sh --provider azure --config configs/API.provider.azure.yaml --model gpt-5 --dry-run
```

Then run:

```bash
bash run_API.sh --provider azure --config configs/API.provider.azure.yaml --model gpt-5
```

If `evaluate_after_run` is not enabled, evaluate manually:

```bash
python -m evaluation.bridge --config evaluation/evaluate.BRIDGE.yaml
python -m evaluation.task --config evaluation/evaluate.task.yaml
```

## 5. Common Run YAML Fields

| Field | Meaning |
| --- | --- |
| `provider` | `azure`, `claude`, or `gemini`. |
| `mode` | Dataset split, usually `test`. |
| `models` | Default model list. |
| `experiments` | Rows such as `[direct, greedy, 42]`; the third value is the inference seed and appears in result filenames. |
| `api` | Connection, retry, parallelism, and batch polling parameters. |
| `max_token_input` / `max_token_output` | Input and output token budgets. |
| `chat_params` | Azure Chat Completions parameters. |
| `responses_params` | Azure Responses API reasoning parameters. |
| `params` | Claude Messages parameters. |
| `generation_config` / `thinking_config` | Gemini generation and thinking parameters. |
| `save_input` | Whether to save source metadata with the exact formatted request messages in `input`. Actual few-shot IDs are always retained in `example`. |
| `save_trace` | Whether to save `api_trace/` request/response files. |
| `save_raw_response` | Whether `responses.jsonl` includes raw provider responses. |
| `allow_partial` | Whether to keep partial result files when sample-level API errors occur. |
| `evaluate_after_run` | Whether to run shared evaluation after inference. |
| `evaluation_seed` | Seed for evaluation bookkeeping such as bootstrap. |
| `evaluate_on_partial` | Whether partial outputs can still be automatically evaluated. |
| `task_types` | Evaluation task types, such as `clf`, `ext`, or `gen`. |
| `custom_tasks` | Project-specific task class registrations. |

`greedy` writes deterministic parameters when the provider supports them.
`sampling` should explicitly define `sampling_chat_params`, `sampling_params`, or
`sampling_generation_config`; otherwise the filename may say sampling while the
provider request still behaves like greedy.

## 6. Output And Error Handling

Standard result path:

```text
<dir_result>/<task>/<model>/<task>-<prompt_mode>-<decoding>-<seed>.result.json
<dir_result>/<task>/<model>/<task>-<prompt_mode>-<decoding>-<seed>.result.config.yaml
<dir_result>/<task>/<model>/<task>-<prompt_mode>-<decoding>-<seed>.result.metadata.json
```

The config snapshot contains the redacted effective per-experiment settings.
The metadata JSON adds file hashes, Git state, command information, and runtime
versions.

Sample-level API failures are written into the row-level `error` field. By
default `allow_partial: false`; any sample error raises after writing the result,
which prevents accidental evaluation of incomplete outputs.

Trace path:

```text
<dir_result>/<task>/<model>/api_trace/<task>-<prompt_mode>-<decoding>-<seed>/
```

For clinical data with PHI, a conservative run config is:

```yaml
save_input: false
save_raw_response: false
```

To avoid request/response trace files as well:

```yaml
save_trace: false
```

Azure batch may still create temporary upload/download artifacts. The runner
parses and removes PHI-containing batch artifacts when trace retention is
disabled.

## 7. Evaluation

Manual evaluation entry points:

```bash
python -m evaluation.bridge --config evaluation/evaluate.BRIDGE.yaml
python -m evaluation.task --config evaluation/evaluate.task.yaml
```

When `evaluate_after_run: true`, API inference calls the shared
`evaluation.runner` directly. Built-in BRIDGE tasks and project tasks registered
through `custom_tasks` both use the same evaluation path.

Example custom evaluation registration:

```yaml
task_types:
  - ext
tasks:
  - My-Clinical-Task
custom_tasks:
  ext:
    My-Clinical-Task: my_project.tasks:Task_ext_MyClinicalTask
```

In Python:

```python
from evaluation.config import config_from_yaml
from evaluation.runner import run_config

config = config_from_yaml("evaluation/evaluate.BRIDGE.yaml")
config.model_names = ["gpt-oss-20b", "gpt-5"]
config.prompt_modes = ["direct"]
run_config(config)
```

## 8. Related Documents

- `configs/API.provider.azure.yaml`
- `configs/API.provider.claude.yaml`
- `configs/API.provider.gemini.yaml`
- `configs/API.model.yaml`
- `configs/API.key.example.yaml`
- `README.result.md`
- `README.result.logits.md`
