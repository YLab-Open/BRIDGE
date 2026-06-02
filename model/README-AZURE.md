# Azure OpenAI Wrapper — User Guide

Module `model/azure.py` wraps the Azure OpenAI **Chat Completions** and **Responses** APIs, providing single-query execution, batch payload construction, result parsing, and cost estimation.

---

## 1. Install Dependencies

```bash
pip install openai tiktoken regex
```

## 2. Create a Client

```python
from model.azure import create_azure_client

client = create_azure_client(
    azure_endpoint="https://<your-resource>.openai.azure.com/",
    api_key="<your-api-key>",
    api_version="2025-04-01-preview",
)
```

---

## 3. Single Query — `azure_query_single`

Automatically detects the model family (reasoning / conventional) and selects the appropriate API mode.

```python
from model.azure import azure_query_single

# Plain text input
text, response = azure_query_single(
    client=client,
    input="What is the capital of France?",
    model_name="gpt-4o",
)
print(text)

# Dict input with instruction + input
text, response = azure_query_single(
    client=client,
    input={
        "instruction": "You are a helpful clinical assistant.",
        "input": "Summarize the patient's chief complaint.",
    },
    model_name="gpt-4o",
)

# Reasoning model (auto-selects Responses API)
text, response = azure_query_single(
    client=client,
    input="Explain the mechanism of action of metformin.",
    model_name="o3",
)

# Explicit mode + few-shot examples
text, response = azure_query_single(
    client=client,
    input={"instruction": "Classify the diagnosis.", "input": "Patient has fever..."},
    examples=[
        {"input": "Headache and nausea", "output": "Migraine"},
        {"input": "Chest pain and SOB", "output": "ACS"},
    ],
    model_name="gpt-4o",
    mode="chat",            # "chat", "response", or None (auto)
    max_token_output=4096,
)

# Custom decoding config
text, response = azure_query_single(
    client=client,
    input="Explain transformer architecture.",
    model_name="gpt-4o",
    dict_config={"temperature": 0.7, "top_p": 0.9},
)
```

---

## 4. Batch Query

### 4.1 From Task — `azure_query_batch_from_task`

Automatically loads task data and few-shot examples from the standard directory layout:

```
dataset_raw/
  ├── TaskName.SFT.json
  └── example/
      └── TaskName.example.json
```

```python
from model.azure import azure_query_batch_from_task

records = azure_query_batch_from_task(
    task_name="TaskName",
    model_name="gpt-4o-batch",
    prompt_mode="3-shot-cot",       # direct, cot, 3-shot, 3-shot-cot, ...
    split="test",
    mode=None,                      # None=auto, "chat", "response"
    max_token_input=100 * 1024,
    max_token_output=3072,
)
# Output: azure/input/gpt-4o-batch/3-shot-cot/TaskName.batch.jsonl
```

### 4.2 From Files — `azure_query_batch_from_file`

Specify data, example, and output file paths directly:

```python
from model.azure import azure_query_batch_from_file

records = azure_query_batch_from_file(
    task_name="TaskName",
    model_name="o3-batch",
    prompt_mode="direct",
    split="test",
    mode="response",                # recommended for reasoning models
    path_file_data="path/to/TaskName.SFT.json",
    path_file_example="path/to/TaskName.example.json",  # optional
    path_file_result="output/TaskName.batch.jsonl",      # optional
    max_token_output=4096,
)
```

### 4.3 From Data — `azure_query_batch_from_data`

Pass pre-loaded data and examples directly:

```python
from model.azure import azure_query_batch_from_data

list_dict_data = [
    {"id": "001", "instruction": "Classify.", "input": "...", "language": "en", "split": "test"},
    {"id": "002", "instruction": "Classify.", "input": "...", "language": "en", "split": "test"},
]
examples = [{"input": "example input", "output": "example output"}]

records = azure_query_batch_from_data(
    task_name="TaskName",
    model_name="gpt-4o-batch",
    prompt_mode="direct",
    split="test",
    list_dict_data=list_dict_data,
    examples=examples,
    mode="chat",                    # "chat", "response", or None (auto)
    max_token_output=3072,
)
```

---

## 5. Batch File Operations

### Merge Batch Files

```python
from model.azure import merge_azure_query_batch

merge_azure_query_batch(
    model_name="gpt-4o-batch",
    prompt_mode="3-shot-cot",
    path_dir_raw="azure/input",
    path_dir_merged="azure/input/merged",
    max_lines=50000,
    max_size_mb=180,
)
```

### Save Batch File

```python
from model.azure import save_azure_batch_file

save_azure_batch_file(records, "output/my_batch.jsonl")
```

---

## 6. Result Parsing

### Parse Batch Results

```python
from model.azure import parse_azure_batch_result

results = parse_azure_batch_result("azure/output/batch_result.jsonl")
# [{"id": "custom_id", "output": "response text", "error": ...}, ...]
```

### Convert to Task-Level Results

```python
from model.azure import process_azure_result_to_task_result

process_azure_result_to_task_result(
    path_dir_azure_result="azure/output",
    path_dir_task_save="result",
    path_dir_raw="dataset_raw",
    path_dir_batch="azure/input",
)
```

---

## 7. Cost Estimation

```python
from model.azure import cost_calculation_token, cost_calculation_word, get_azure_price

# Query model pricing (USD / 1M tokens)
input_price, output_price = get_azure_price("gpt-4o", flag_batch=True)

# Token-based calculation
cost = cost_calculation_token(
    num_token_input_all=500_000,
    num_token_output_all=100_000,
    model_name="gpt-4o",
    flag_batch=True,
)
print(f"Estimated cost: ${cost:.4f}")

# Text-based calculation
cost = cost_calculation_word(
    list_input=["input text 1", "input text 2"],
    list_output=["output text 1", "output text 2"],
    model_name="gpt-4o",
    flag_batch=True,
)
```

---

## 8. API Mode Reference

| Mode | Endpoint | Applicable Models | Description |
|------|----------|-------------------|-------------|
| `chat` | `/chat/completions` | gpt-4o, gpt-4o-mini, etc. | Standard Chat Completions API |
| `response` | `/responses` | o1, o3, o4-mini, gpt-5, etc. | Responses API with reasoning effort support |

- When `mode=None`, the mode is auto-detected based on the model name.
- Reasoning models default to `response`; conventional models default to `chat`.
- You can always override by explicitly passing the `mode` parameter.

---

## 9. Supported Model Pricing

| Model | Input ($/1M tokens) | Output ($/1M tokens) | Batch (50% off) |
|-------|------|--------|------|
| gpt-5.2 | 1.75 | 14.0 | ✅ |
| gpt-5.1 | 1.25 | 10.0 | ✅ |
| gpt-5 | 1.25 | 10.0 | ✅ |
| gpt-4o | 2.5 | 10.0 | ✅ |
| gpt-4o-mini | 0.15 | 0.6 | ✅ |
| o1 | 15.0 | 60.0 | ✅ |
| o3 | 2.0 | 8.0 | ✅ |
| o3-mini | 1.1 | 4.4 | ✅ |
| o4-mini | 1.1 | 4.4 | ✅ |
