# BRIDGE Evaluation

This folder contains the maintained evaluation entry point and reusable helpers
for BRIDGE result analysis.

Evaluation is configured through `evaluate_BRIDGE.yaml` in this same folder. The
root-level `evaluate_BRIDGE.py` has been removed to keep the entry point and its
config together.

## Command Examples

Run with the default config:

```bash
python -m evaluation.bridge
```

Run with another config:

```bash
python -m evaluation.bridge --config evaluation/evaluate_BRIDGE.yaml
```

Configure multiple models and prompt modes in YAML:

```yaml
models:
  - model_a
  - model_b

prompt_modes:
  - direct
  - cot
  - direct-5-shot

dir_result:
  - result
  - result_gpt_oss
```

`dir_result` can be one path, a comma-separated string, or a YAML list. The
evaluation search traverses every configured result root.

## Modules

- `bridge.py`: CLI entry point and high-level task evaluation orchestration.
- `evaluate_BRIDGE.yaml`: default evaluation config.
- `common.py`: CLI parsing helpers, JSON I/O, seed setup, task args adapter.
- `task_registry.py`: classification, extraction, and generation task registry
  loading.
- `performance.py`: performance JSON merge/load/print helpers.
- `result_utils.py`: result-file discovery, filename parsing, and lightweight
  result inventory helpers for notebooks.

Evaluation notebooks live in `notebooks/evaluation/` so all interactive examples
are grouped under the top-level `notebooks/` folder.

The folder is named `evaluation` instead of `evaluate` to avoid shadowing the
Hugging Face `evaluate` package used by generation metrics.
