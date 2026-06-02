# Tests

This folder contains lightweight regression tests for runtime configuration and
local-only config validation.

## Files

- `test_runtime.py`: tests strict prompt-mode parsing, GPU parsing,
  duplicate-key YAML rejection, CLI/YAML merge priority, runtime validation,
  local API-key config validation, model-path mapping validation, default
  evaluation config path, and documentation naming.
- `test_provider_helpers.py`: tests shared provider prompt helpers and Azure
  request-body, response-parsing, model-detection, and cost helpers.
- `test_evaluation_notebooks.py`: executes maintained evaluation notebooks with
  `RUN_EVALUATION = False` to keep examples importable and runnable.

## Running

```bash
pytest -q
```

`pytest.ini` sets the repository import path and prevents pytest from walking
large data, result, performance, notebook, and provider-output folders.

These tests intentionally avoid loading real LLM checkpoints or calling external
APIs, so they should be fast enough to run before committing framework changes.
