# Model Layer

This folder contains model initialization, local inference loops, and the
unified commercial API inference implementation.

## Files

- `init.py`: local model setup. It resolves model paths from
  `configs/dict_model_path.json`, sets token budgets, initializes tokenizers/models, and
  builds decoding configs for Hugging Face and vLLM.
- `inference.py`: local HF/vLLM inference loops and standardized result saving.
- `logits.py`: optional synchronous vLLM generation-token trace, exact
  label-position probability probes, and compact logits sidecar writers.
- `api_runner.py`: public commercial API entry point. It loads API YAML,
  prepares task prompts, dispatches provider calls, writes standard result JSON,
  and optionally triggers evaluation.
- `api/`: commercial API internals:
  - `config.py`: API YAML/model-registry parsing, credential merging, and
    per-experiment decoding parameter resolution.
  - `prompts.py`: task dataset loading and provider-neutral chat-message
    construction.
  - `provider/`: Azure, Claude, and Gemini request implementations, split by
    provider plus a shared factory and Azure request helpers.
  - `results.py`: normalized query/experiment/run result data structures.
- `metadata.py`: common local/API config snapshots, file hashes, Git state, and
  runtime-version manifests.

## Local Model Mapping

Real local model paths should be stored in `configs/dict_model_path.json`, which is
ignored by git. Keep `configs/dict_model_path.example.json` updated as the public
template:

```json
{
  "gpt-oss-20b": "/path/to/gpt-oss-20b"
}
```

`model.init.init_model()` loads this through `util.local_config.resolve_model_path()`.

## Local Inference

The primary local path is:

1. `main.py` loads YAML/CLI runtime config.
2. `model.init.init_model()` resolves the local checkpoint and loads tokenizer/model.
3. `dataset.GeneralTask` returns formatted prompts through `dataset.process`.
4. `model.inference.run_vllm()` or `run_hf()` generates predictions.
5. `model.inference.save_result()` writes standardized result JSON.

## Synchronous vLLM Logits

Local vLLM inference can optionally save logits through a separate YAML passed to
`main.py`:

```bash
python main.py \
  --model_name MyLocalModel \
  --config configs/BRIDGE.yaml \
  --logits_config configs/logits.example.yaml
```

The normal `<task>-<prompt>-<decoding>-<seed>.result.json` remains unchanged and
evaluable. Logits are stored separately as:

```text
<task>-<prompt>-<decoding>-<seed>.result.logit.json
<task>-<prompt>-<decoding>-<seed>.result.logit.top_logits.jsonl.gz
<task>-<prompt>-<decoding>-<seed>.result.metadata.json
<task>-<prompt>-<decoding>-<seed>.result.config.yaml
```

The generic `generation` block records generated token ids/texts from the actual
vLLM `RequestOutput`. For configured classification tasks, `logit` defaults to
`generation_logprobs`: it scores the generated label position from generation-time
top-K logits. If a configured label token is missing, the code falls back to
`exact_output_token_probe`, re-probes one next-token position with `logprobs=-1`,
and saves compact label/group probabilities plus top-k raw logits.

Commercial APIs are not treated as raw-logit providers. If a provider response
contains logprobs in its raw trace, provider-specific parsing can standardize
them later, but local vLLM is the supported path for raw logits.

## Commercial API Inference

The public API entry point is `model.api_runner.run_api_from_config`, which is
used by `main_api.py`, `run_API.sh`, and notebooks:

```python
from model.api_runner import run_api_from_config

run = run_api_from_config("configs/API.provider.azure.yaml", model_name="gpt-4o")
```

`model/api/` keeps the commercial API workflow split by responsibility, while
`model.api_runner` remains the stable import path for existing code.

Key internal functions:

```python
from model.api.config import require_model_section, resolve_api_settings
from model.api.prompts import load_task_dataset, build_messages_for_dataset
from model.api.provider import make_provider
from model.api.results import QueryResult, ApiRunResult
```

Provider code is split by API surface:

- `model/api/provider/azure.py`: Azure Chat Completions, Responses, and Batch API.
- `model/api/provider/claude.py`: Anthropic Messages API.
- `model/api/provider/gemini.py`: Gemini `generate_content`.
- `model/api/provider/factory.py`: maps the YAML `provider` value to the right
  provider class.

## Commercial API Prompt Order

Commercial API providers keep the same logical order as local inference:

1. Instruction as system message when the provider supports it.
2. Optional few-shot user/assistant examples.
3. Current sample input as the final user message.
4. Model output corresponds to the task `output` field.

CoT modes are converted through the same Direct-to-CoT prompt helpers used by
local inference, and invalid prompt modes are rejected by the shared runtime
parser.
