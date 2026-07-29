# Evaluation

This folder contains the shared evaluation engine for BRIDGE benchmark tasks and
project-specific clinical tasks.

There are now two CLI entry points so the default BRIDGE registry and ad hoc
project task registrations are not mixed in one workflow.

## Command Examples

Evaluate built-in BRIDGE tasks:

```bash
python -m evaluation.bridge
python -m evaluation.bridge --config evaluation/evaluate.BRIDGE.yaml
```

Evaluate project-specific tasks:

```bash
python -m evaluation.task --config evaluation/evaluate.task.yaml
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

## Config Types

`evaluate.BRIDGE.yaml` is only for built-in BRIDGE registry tasks. It should not
contain `custom_tasks`; `evaluation.bridge` rejects that and tells you to use
`evaluation.task`.

`evaluate.task.yaml` is for project-specific tasks and non-default datasets. It
must explicitly set both `tasks` and `custom_tasks` so `tasks: null` cannot
accidentally evaluate every built-in BRIDGE task. Use
`evaluate.task.example.yaml` as the template.

## Custom Tasks

The default registry covers BRIDGE tasks. For project-specific tasks, do not copy
the evaluation entry point. Register the task class in `evaluate.task.yaml`:

```yaml
task_types:
  - ext
tasks:
  - My-Clinical-Task
custom_tasks:
  ext:
    My-Clinical-Task: my_project.tasks:Task_ext_MyClinicalTask
```

Task class paths can use `module:ClassName` or `module.ClassName`. Custom classes
should follow the existing classification, extraction, or generation task class
interfaces and implement task-specific label/pred parsing and metrics.

API inference with `evaluate_after_run: true` calls the shared
`evaluation.runner`, not either CLI. That means API YAML can evaluate either
BRIDGE tasks or custom tasks registered through `custom_tasks`.

The same runner can be used directly from Python:

```python
from evaluation.config import config_from_yaml
from evaluation.runner import run_config

config = config_from_yaml("evaluation/evaluate.BRIDGE.yaml")
config.model_names = ["gpt-oss-20b", "gpt-5"]
config.prompt_modes = ["direct"]
run_config(config)
```

For a project-specific task, use `evaluation/evaluate.task.yaml` and register
the task class in `custom_tasks`; then call `run_config(config)` the same way.

## Modules

- `bridge.py`: CLI entry point for built-in BRIDGE tasks, using
  `evaluate.BRIDGE.yaml`.
- `task.py`: CLI entry point for project-specific tasks, using
  `evaluate.task.yaml`.
- `config.py`: YAML parsing, `EvaluationConfig`, seed setup, task args adapter.
- `runner.py`: shared evaluation loop used by CLI and API automatic evaluation.
- `reporting.py`: performance JSON merge/load/print helpers.
- `task_registry.py`: classification, extraction, and generation task registry
  loading.
- `result_utils.py`: result-file discovery, filename parsing, and lightweight
  result inventory helpers for notebooks.
- `evaluate.BRIDGE.yaml`: default BRIDGE evaluation config.
- `evaluate.task.example.yaml`: custom task registration template.
- `evaluate.task.yaml`: current project task evaluation config.

General evaluation notebooks live in `notebooks/evaluation/`. Medication-specific
analysis is a task-specific workflow; keep it in the root quick-analysis
notebooks and the ignored `medication/` folder, not in the generic
`evaluation/` module.

The folder is named `evaluation` instead of `evaluate` to avoid shadowing the
Hugging Face `evaluate` package used by generation metrics.
