# Dataset Layer

This folder defines how BRIDGE task files are loaded, formatted as prompts, and
parsed during evaluation.

## Files

- `dataset.py`: shared task loader. `GeneralTask` loads `<task>.SFT.json`,
  creates train/dev/test splits, applies prompt modes, and returns dataloaders.
  `GeneralEvaluation` extends it with result-file discovery, alignment warnings,
  deterministic evaluation seeding, bootstrap evaluation, and metric aggregation.
- `process.py`: prompt-formatting helpers for local models and provider batch
  formats. It handles chat templates, rough token estimates, truncation, and
  provider-specific message structures.
- `config.py`: prompt templates, Direct/CoT conversion helpers, invalid-output
  fallback helpers, metric-name lists, and preferred model display order.
- `classification.py`: task adapters for binary, multi-class, multi-label, and
  question-style classification tasks.
- `extraction.py`: task adapters for entity, relation, normalization, temporal,
  medication, and QA-style extraction tasks.
- `generation.py`: task adapters for generation and generation-as-extraction
  tasks.
- `task.py`: registry functions that map task names to task adapter classes.

## Data Contract

Each task file should be stored as:

```text
<dir_dataset>/<task_name>.SFT.json
```

The file is a JSON list. Each record should contain at least:

```json
{
  "id": "sample-id",
  "split": "test",
  "language": "en",
  "type": "classification",
  "instruction": "Task instruction...",
  "input": "Clinical text...",
  "output": "Gold answer..."
}
```

Few-shot examples are loaded from:

```text
<dir_dataset>/example/<task_name>.example.json
<dir_dataset>/example_cot/<task_name>.example.json
```

## Prompt Modes

Prompt modes are parsed by `util.runtime.parse_prompt_mode()` and then applied
in `GeneralTask.setup()`.

Supported modes:

- `direct`
- `cot`
- `direct-N-shot`, for example `direct-5-shot`
- `cot-N-shot`, for example `cot-5-shot`

CoT conversion raises an error if the task instruction does not match a known
Direct template. Few-shot modes raise an error if the required example file is
missing.

## Evaluation Notes

`GeneralEvaluation` warns when a result file has a different row count or ID
order from the expected task data. This supports partial sampling runs while
making data-alignment issues visible.

Classification adapters may replace invalid predictions with randomly selected
valid labels. The evaluation seed is set per result file, so repeated runs with
the same seed are reproducible.
