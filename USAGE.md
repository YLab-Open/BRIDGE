# BRIDGE Usage

This repository provides a configurable pipeline for running local or batch
inference on clinical text tasks and evaluating model outputs with task-specific
parsers and metrics.

The core workflow is:

1. Store benchmark data as `<task_name>.SFT.json`.
2. Configure task names, data paths, result paths, prompt modes, decoding modes,
   token budgets, and runtime options in `configs/BRIDGE.yaml`.
3. Run local inference through `main.py`, `run.sh`, or `run_multiple_model.sh`,
   or commercial API inference through `run_API.sh`.
4. Save standardized result JSON files under `<dir_result[0]>/<task>/<model>/`.
5. Evaluate result files with `python -m evaluation.bridge` for BRIDGE tasks or
   `python -m evaluation.task` for project-specific tasks.

## Runtime

The default local runtime is vLLM. The runtime field is named `engine` in the
YAML and CLI:

```yaml
engine: vllm
```

Valid values:

- `vllm`: high-throughput local inference with vLLM.
- `hf`: Hugging Face `generate()` inference.

Recommended local environment:

```bash
mamba activate vllm
python -c "import vllm; print(vllm.__version__)"
```

Install the Python dependencies listed in `requirements.txt` when preparing a
new environment. Generation-task evaluation additionally requires `bert-score`;
classification and extraction evaluation do not.

Run help:

```bash
python main.py --help
```

Direct run example:

```bash
python main.py \
  --model_name gpt-oss-20b \
  --gpus 0,1 \
  --engine vllm \
  --config configs/BRIDGE.yaml \
  --model_file configs/dict_model_path.json
```

`--model_file` should point to a JSON mapping from model name to local model
path:

```json
{
  "gpt-oss-20b": "models/gpt-oss-20b"
}
```

Local machine-specific files are ignored by git. Start from the example files:

```bash
cp configs/BRIDGE.example.yaml configs/BRIDGE.yaml
cp configs/API.key.example.yaml configs/API.key.yaml
cp configs/dict_model_path.example.json configs/dict_model_path.json
```

Use `configs/API.key.yaml` for provider credentials such as `AZURE-OPENAI`. Use
`configs/dict_model_path.json` for local checkpoint paths. Shared loaders live in
`util/local_config.py`.

## Data Format

Each task file is a JSON list:

```text
<dir_dataset>/<task_name>.SFT.json
```

Each sample should contain at least:

```json
{
  "id": "sample-id",
  "split": "test",
  "language": "en",
  "type": "classification",
  "instruction": "Task instruction...",
  "input": "Clinical text...",
  "output": "Ground truth answer..."
}
```

Optional few-shot examples are loaded from:

```text
<dir_dataset>/example/<task_name>.example.json
<dir_dataset>/example_cot/<task_name>.example.json
```

## Configuration

Important `configs/BRIDGE.yaml` fields:

- `engine`: inference engine, `vllm` or `hf`.
- `mode`: dataset split to run, one of `train`, `dev`, `test`, or `all`.
- `experiments`: list of `[prompt_mode, decoding_strategy, seed]`. The third
  value is the local inference seed; it is normalized to an integer, assigned to
  `args.seed`, and passed to vLLM `SamplingParams.seed` when using vLLM.
- `batch_size`: number of formatted prompts submitted per dataloader batch.
- `num_workers`: optional prompt-formatting worker count. The default is twice
  the number of configured GPUs; use `0` for small/debug runs to avoid process
  startup overhead.
- `save_every_n_batches`: optional append-only JSONL checkpoint interval.
- `resume_partial_results`: resume a validated interrupted vLLM prefix.
- `save_input`: whether to keep instruction/input metadata in result JSON.
- `max_token_all`: full context window passed to vLLM as `max_model_len`.
- `max_token_input`: prompt budget after formatting and truncation.
- `max_token_output`: generation budget.
- `gpu_memory_util`: vLLM GPU memory utilization.
- `dir_dataset`: one or more comma-separated dataset roots.
- `dir_result`: result root(s). Use a comma-separated string or YAML list. In
  inference, new files are saved to the first root; evaluation searches all
  configured roots.
- `tasks`: list of task names matching `<task_name>.SFT.json`.

The config loader rejects duplicate YAML keys and validates required fields
before model loading starts.

## Sampling Configuration

For experiments whose `decoding_strategy` is `sampling`, decoding parameters are
resolved in this order:

1. Explicit values in YAML, such as `temperature`, `top_p`, `top_k`, or
   `repetition_penalty`.
2. The local model directory's `generation_config.json`.
3. Code defaults.

The default `sample_size` is `1`, so saved predictions remain directly
evaluable. If `sample_size > 1`, `pred` is saved as a list of generated strings.
Evaluation selects `sample_eval_index` from that list, with default index `0`.

Common optional fields:

- `sample_size`
- `temperature`
- `top_p`
- `top_k`
- `min_p`
- `repetition_penalty`
- `presence_penalty`
- `frequency_penalty`
- `sample_eval_index`

## Prompt Modes

Supported modes:

- `direct`
- `cot`
- `direct-N-shot`, for example `direct-5-shot`
- `cot-N-shot`, for example `cot-5-shot`

Prompt setup happens in `GeneralTask.setup()`. Each experiment starts from a
fresh copy of the base task data, so Direct, CoT, and few-shot transformations do
not mutate later experiments in the same process.

Prompt modes are validated strictly. Strings such as `directly`, `cot-shot`, or
`direct-0-shot` are rejected. CoT conversion raises an error if the task
instruction does not contain a known Direct template. Few-shot modes raise an
error if the matching example file is missing.

## Inference Flow

The main local path is:

1. `main.py` parses CLI arguments.
2. `util.runtime.load_yaml_config()` loads YAML and rejects duplicate keys.
3. `util.runtime.merge_config_into_args()` merges YAML into CLI args.
4. `util.runtime.validate_args()` validates runtime fields and token budgets.
5. `util.runtime.normalize_runtime_args()` normalizes GPU IDs and dataset roots.
6. `model.init.init_model()` resolves `model_path`, token budgets, chat template
   kwargs, tokenizer, and model.
7. `dataset.dataset.GeneralTask` loads the task file and split lists.
8. `GeneralTask.setup()` applies prompt transforms and optional examples.
9. `util.runtime.get_task_dataloader()` returns the selected split dataloader.
10. `model.inference.run_vllm()` or `run_hf()` generates model outputs.
11. `model.inference.save_result()` validates result length and writes JSON.

## Commercial API Inference

Commercial API inference uses the same task files and writes the same result JSON
format as local inference. The shell entry point is:

```bash
bash run_API.sh
bash run_API.sh --provider azure --config configs/API.provider.azure.yaml --model gpt-4o --foreground
bash run_API.sh --dry-run
```

The Python/notebook entry point is:

```python
from model.api_runner import run_api_from_config

run = run_api_from_config("configs/API.provider.azure.yaml", model_name="gpt-4o")
```

Use `configs/API.model.yaml` for provider model metadata and
`configs/API.key.yaml` or environment variables for credentials. See
See `configs/API_USAGE.md` for the detailed Azure/Claude/Gemini workflow, trace
files, batch polling, and automatic evaluation options.

## Result Format

Inference saves:

```text
<dir_result[0]>/<task_name>/<model_name>/<task_name>-<prompt_mode>-<decoding_strategy>-<seed>.result.json
```

For `save_input: true`, each output row keeps the source metadata and adds
`example` and `pred`. For local inference, `input` is replaced by only the
current sample text retained after context truncation; `instruction` keeps the
system prompt.
`example` is the ordered list of few-shot example IDs that actually fit in the
formatted context; it can contain fewer IDs than requested by the prompt mode.
For `save_input: false`, each row keeps only compact metadata:

```json
{
  "id": "sample-id",
  "type": "classification",
  "language": "en",
  "output": "gold answer",
  "example": ["example-1", "example-2", "example-3"],
  "pred": "model output"
}
```

For commercial API inference, `input` is the final formatted request-message
list after context budgeting. Local requests can be reconstructed from
`instruction`, truncated `input`, the examples referenced by `example`, and the
same model tokenizer/chat template.

`save_result()` raises an error if the number of predictions does not match the
number of samples being saved.

## Optional vLLM Logits Capture

Local vLLM runs can synchronously save generated token ids and task-specific
label probabilities without changing the standard result JSON. Use a separate
logits YAML:

```bash
python main.py \
  --model_name MyLocalModel \
  --config configs/BRIDGE.yaml \
  --logits_config configs/logits.example.yaml
```

Recommended layout:

```yaml
enabled: true
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
  fail_on_error: true
tasks:
  MyTask:
    answer_prefix: "Assessment:"
    label_values: ["No", "Yes"]
    label_groups:
      negative: ["No"]
      positive: ["Yes"]
```

The normal file remains:

```text
<result>/<task>/<model>/<task>-direct-greedy-42.result.json
```

Additional compact files are written next to it:

```text
<task>-direct-greedy-42.result.logit.json
<task>-direct-greedy-42.result.logit.top_logits.jsonl.gz
<task>-direct-greedy-42.result.metadata.json
<task>-direct-greedy-42.result.config.yaml
```

`generation` records the actual generated token ids/texts. `logit` records
label-token raw logits, positive/negative conditional probabilities, and a
reference to the compressed top-k sidecar. By default the code scores from
generation-time top-5000 logits and falls back to a one-token whole-vocabulary
probe only when a configured label token is missing from that top-K set.

## Example Notebooks

Maintained example notebooks:

- `notebooks/inference/Inference task.Local.ipynb`: structured local vLLM example. It builds a
  detailed interactive local vLLM workflow.
- `notebooks/inference/Inference task.Azure.ipynb`: structured Azure OpenAI single-query and batch
  example. Credentials are read from `configs/API.key.yaml` under the `AZURE-OPENAI`
  block.
- `notebooks/evaluation/evaluate_classification.ipynb`
- `notebooks/evaluation/evaluate_extraction.ipynb`
- `notebooks/evaluation/evaluate_generation.ipynb`

Older notebooks under `analysis result/` are exploratory analysis records. Common
helpers for new analysis notebooks live in `evaluation/`.

## Evaluation Flow

`python -m evaluation.bridge` reads `evaluation/evaluate.BRIDGE.yaml`, then
loads standardized result files and dispatches each task to the matching built-in
BRIDGE task class from `dataset.task`. The evaluation config is intentionally
independent from the inference YAML. `dir_result` may contain multiple result
roots; all of them are searched for matching task/model/prompt result files.

Common examples:

```bash
python -m evaluation.bridge

python -m evaluation.bridge --config evaluation/evaluate.BRIDGE.yaml

python -m evaluation.task --config evaluation/evaluate.task.yaml
```

Set models, prompt modes, task types, and task subsets in the YAML file. Use
`models: all` to evaluate every model folder found under `result/<task>/`, or
provide a YAML list for an explicit subset. Built-in BRIDGE tasks use
`evaluation/evaluate.BRIDGE.yaml`; project-specific tasks use
`evaluation/evaluate.task.yaml` and must explicitly set `tasks` and
`custom_tasks`.

Evaluation path:

1. `dataset.task` maps task names to classification, extraction, or generation
   classes.
2. `GeneralEvaluation.search_result_by_model()` finds result JSON files for a
   task/model/prompt mode.
3. Task-specific `get_label()` and `get_pred()` parse labels and model outputs.
4. Invalid predictions are normalized by task-specific `get_pred_none()`.
5. Metrics are computed through `metric.classification`, `metric.extraction`, or
   `metric.generation`.
6. Bootstrap statistics are aggregated and written to `performance/`.

Evaluation warns, but does not fail, if a result file has fewer or more rows than
the expected task split. This supports partial sampling runs while making
misaligned result files visible. Invalid classification predictions are still
randomly replaced with valid labels, but the evaluation RNG is seeded per result
file so repeated runs with the same seed are reproducible.

## Directory Guide

- `dataset/`: data loading, prompt setup, output parsing, and task registries.
- `configs/`: runtime YAML and local-only config templates.
- `scripts/`: delayed shell launchers for local inference jobs.
- `notebooks/`: interactive examples; inference notebooks live in
  `notebooks/inference/` and evaluation notebooks live in `notebooks/evaluation/`.
- `metric/`: metric implementations.
- `model/`: model loading, local inference loops, logits/metadata, and unified
  commercial API providers.
- `util/`: runtime config helpers, logging, JSON I/O, statistics, preprocessing.
- `evaluation/`: maintained BRIDGE evaluation CLI and reusable
  result/performance helpers.
- `data_contamination/`: n-gram based contamination checks.
- `tests/`: lightweight runtime regression tests.
- `log/`: runtime logs.
- `result/`: default inference result directory.
- `performance/`: default evaluation output directory.

## File Guide

Top-level:

- `main.py`: inference entry point. It is intentionally thin; runtime parsing and
  validation live in `util/runtime.py`.

`configs/`:

- `BRIDGE.yaml`: default inference YAML used by `main.py`.
- `API.key.example.yaml`: provider credential template.
- `dict_model_path.example.json`: local checkpoint mapping template.

Root launch scripts:

- `run.sh`: single-model launcher.
- `run_multiple_model.sh`: sequential multi-model launcher.
  Both support `--dry-run` to print commands without launching inference.

`scripts/`:

- `scripts/run_timing.sh`: delayed single-model launcher.
- `scripts/run_multiple_model_timing.sh`: delayed multi-model launcher.

`util/`:

- `runtime.py`: YAML loading, duplicate-key detection, CLI/YAML merge, config
  validation, GPU parsing, dataloader selection, task logging.
- `local_config.py`: validated loaders for local-only files such as
  `configs/API.key.yaml` and `configs/dict_model_path.json`.
- `io.py`: JSON/JSONL, iterable JSONL, text-file, and dictionary helpers.
- `logging.py`: logger setup and the legacy terminal progress bar.
- `stats.py`: confidence interval/statistical helpers.
- `text.py`: text/list cleanup helpers.
- `misc.py`: small argument, memory-size, and dictionary helpers.
- `seed.py`: single shared seed utility. Local inference uses
  `seed_inference()`; evaluation uses `seed_everything(..., verbose=False)`.
- `preprocess.py`: punctuation normalization and simple text cleanup.

`dataset/`:

- `dataset.py`: `GeneralDataset`, `GeneralTask`, and `GeneralEvaluation`.
- `process.py`: chat formatting, token counting, truncation, provider-specific
  message formats.
- `config.py`: prompt templates, Direct/CoT transforms, invalid-output helpers,
  model ordering.
- `classification.py`: classification task adapters.
- `extraction.py`: extraction task adapters.
- `generation.py`: generation task adapters.
- `task.py`: task registry.

`model/`:

- `init.py`: model path resolution, token budget setup, tokenizer/model loading,
  and decoding parameter builders.
- `inference.py`: HF/vLLM generation loops and result saving.
- `api_runner.py`: stable commercial API entry point used by `run_API.sh`,
  `main_api.py`, and notebooks.
- `api/`: commercial API internals split into `config.py`, `prompts.py`,
  `provider/`, and `results.py`.
- `metadata.py`: common config snapshots and reproducibility manifests.

`metric/`:

- `classification.py`: accuracy, F1, ordered and multi-label metrics.
- `extraction.py`: subject/event extraction and QA matching metrics.
- `generation.py`: BLEU, ROUGE, METEOR, and BERTScore.

`evaluation/`:

- `bridge.py`: BRIDGE CLI for built-in tasks, using `evaluate.BRIDGE.yaml`.
- `task.py`: project-task CLI, using `evaluate.task.yaml`.
- `config.py`: YAML parsing, `EvaluationConfig`, seed setup, task args adapter.
- `runner.py`: shared evaluation loop used by CLI and API automatic evaluation.
- `reporting.py`: performance JSON merge/load/print helpers.
- `task_registry.py`: task registry and metric-printer loading.
- `result_utils.py`: result-file discovery and inventory helpers.
- `README.md`: command examples and module guide.

`notebooks/evaluation/`:

- `evaluate_classification.ipynb`: classification evaluation example.
- `evaluate_extraction.ipynb`: extraction evaluation example.
- `evaluate_generation.ipynb`: generation evaluation example.

Every maintained code folder includes an English `README.md`. Large
data/result/log folders are intentionally documented from the top level because
their contents are generated or local-only.

## Performance Notes

- vLLM throughput is affected most by `batch_size`, `max_token_output`,
  `max_model_len`, and `max_num_seqs`.
- Very large `max_token_output` values reserve KV cache even when outputs are
  short; lower this first when memory pressure is high.
- `save_every_n_batches` appends only new rows to a JSONL checkpoint, avoiding
  repeated full-result rewrites while limiting work lost after a crash.
- `save_input: true` produces larger JSON files. Keep it enabled when debugging
  prompt quality; disable it for very large production runs if disk I/O becomes
  the bottleneck.
- Prompt formatting is streamed from dataloader workers without an unbounded
  per-worker cache, avoiding retention of every long prompt for the full run.
  Fixed few-shot example token counts are cached per worker/run.
  Each experiment still gets a fresh dataset copy to avoid prompt-state
  contamination.

## Lightweight Tests

Run:

```bash
pytest -q
```

`tests/test_runtime.py` guards strict prompt mode parsing, GPU parsing,
duplicate-key YAML rejection, CLI/YAML merge priority, runtime validation,
local-only config validation, model-path mapping validation, default evaluation
config path, and documentation naming. `tests/test_provider_helpers.py` covers
provider prompt/order helpers and Azure request/response helpers.
`tests/test_evaluation_notebooks.py` executes the maintained evaluation notebooks
with evaluation disabled. `pytest.ini` sets the repository import path and
prevents pytest from walking large data, result, provider-output, and analysis
folders.
