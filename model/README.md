# Model Layer

This folder contains model initialization, local inference loops, and provider
batch helpers.

## Files

- `init.py`: local model setup. It resolves model paths from
  `configs/dict_model_path.json`, sets token budgets, initializes tokenizers/models, and
  builds decoding configs for Hugging Face and vLLM.
- `inference.py`: local HF/vLLM inference loops and standardized result saving.
- `azure.py`: Azure OpenAI single-query, batch-request creation, batch-result
  parsing, and task-result integration.
- `azure_utils.py`: Azure request-body builders, response text extractors,
  model-family detection, tokenizer-name resolution, custom-id parsing, and cost
  helpers. These names are re-exported by `azure.py` for compatibility.
- `provider_common.py`: provider-agnostic CoT instruction conversion and
  few-shot example path helpers shared by Azure, GCP, and DeepSeek code.
- `gcp.py`: Gemini/Vertex batch-request creation, result parsing, integration,
  and cost helpers.
- `deepseek.py`: DeepSeek batch-request creation, result parsing, and
  integration helpers.
- `README-AZURE.md`: Azure-specific detailed notes.

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

## Provider Batch Prompt Order

Provider batch helpers keep the same logical order as local inference:

1. Instruction as system message when the provider supports it.
2. Optional few-shot user/assistant examples.
3. Current sample input as the final user message.
4. Model output corresponds to the task `output` field.

CoT modes are converted through the same Direct-to-CoT prompt helpers used by
local inference, and invalid prompt modes are rejected by the shared runtime
parser.
