from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from util.runtime import load_yaml_config, parse_prompt_mode
from util.seed import seed_everything

from .task_registry import TASK_TYPE_ORDER, get_default_models_to_print


@dataclass
class EvaluationConfig:
    """Runtime configuration for result evaluation."""

    dir_dataset: list[str] = field(default_factory=lambda: ["dataset_raw"])
    dir_result: list[str] = field(default_factory=lambda: ["result"])
    dir_performance: str = "performance"
    prompt_modes: list[str] = field(default_factory=lambda: ["direct"])
    model_names: list[str] | None = None
    models_to_print: list[str] = field(default_factory=get_default_models_to_print)
    task_types: list[str] = field(default_factory=lambda: list(TASK_TYPE_ORDER))
    tasks: list[str] | None = None
    custom_tasks: dict = field(default_factory=dict)
    bootstrap: int = 1000
    seed: int = 42
    sample_eval_index: int = 0
    print_summary: bool = True


def split_csv(value: str | None) -> list[str]:
    """Parse a comma-separated value into a clean list."""
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def as_list(value, field_name: str, allow_none: bool = True) -> list[str] | None:
    """Normalize a YAML scalar/list field into a list of strings."""
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field_name} must be set.")
    if isinstance(value, str):
        return split_csv(value)
    if isinstance(value, list):
        values = [str(item).strip() for item in value if str(item).strip()]
        if not values and not allow_none:
            raise ValueError(f"{field_name} must not be empty.")
        return values
    raise ValueError(f"{field_name} must be a string, list, or null.")


def parse_models(value) -> list[str] | None:
    """Parse configured models; `all` or null means every model folder found."""
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() == "all":
        return None
    models = as_list(value, "models", allow_none=False)
    if not models:
        raise ValueError("models must be a non-empty list or 'all'.")
    return models


def parse_prompt_modes(value) -> list[str]:
    """Parse and validate configured prompt modes."""
    prompt_modes = as_list(value, "prompt_modes", allow_none=False)
    if not prompt_modes:
        raise ValueError("prompt_modes must not be empty.")
    for prompt_mode in prompt_modes:
        parse_prompt_mode(prompt_mode)
    return prompt_modes


def parse_task_types(
    value,
    valid_task_types: tuple[str, ...] = TASK_TYPE_ORDER,
) -> list[str]:
    """Parse configured task types and validate against known groups."""
    task_types = as_list(value, "task_types", allow_none=True)
    if task_types is None:
        task_types = list(valid_task_types)
    invalid = [task_type for task_type in task_types if task_type not in valid_task_types]
    if invalid:
        raise ValueError(
            f"Invalid task type(s): {invalid}. Use {', '.join(valid_task_types)}."
        )
    return task_types or list(valid_task_types)


def load_evaluation_yaml(path_config: str | Path) -> dict:
    """Load an evaluation YAML config with duplicate-key checking."""
    return load_yaml_config(str(path_config))


def config_from_yaml(path_config: str | Path) -> EvaluationConfig:
    """Build a validated EvaluationConfig from a YAML file."""
    cfg = load_evaluation_yaml(path_config)
    if not isinstance(cfg, dict):
        raise ValueError(f"Evaluation config must be a YAML mapping: {path_config}")

    bootstrap = int(cfg.get("bootstrap", 1000))
    sample_eval_index = int(cfg.get("sample_eval_index", 0))
    if bootstrap < 0:
        raise ValueError("bootstrap must be >= 0.")
    if sample_eval_index < 0:
        raise ValueError("sample_eval_index must be >= 0.")

    return EvaluationConfig(
        dir_dataset=as_list(
            cfg.get("dir_dataset", ["dataset_raw"]),
            "dir_dataset",
            allow_none=False,
        ),
        dir_result=as_list(
            cfg.get("dir_result", ["result"]),
            "dir_result",
            allow_none=False,
        ),
        dir_performance=str(cfg.get("dir_performance", "performance")),
        prompt_modes=parse_prompt_modes(cfg.get("prompt_modes", ["direct"])),
        model_names=parse_models(cfg.get("models", "all")),
        task_types=parse_task_types(cfg.get("task_types")),
        tasks=as_list(cfg.get("tasks"), "tasks", allow_none=True),
        custom_tasks=cfg.get("custom_tasks", {}) or {},
        bootstrap=bootstrap,
        seed=int(cfg.get("seed", 42)),
        sample_eval_index=sample_eval_index,
        print_summary=bool(cfg.get("print_summary", True)),
    )


def set_evaluation_seed(seed: int):
    """Seed Python and NumPy for deterministic evaluation bookkeeping."""
    return seed_everything(seed, verbose=False)


def make_task_args(config: Any):
    """Build the minimal args object expected by task evaluation classes."""
    return SimpleNamespace(
        dir_dataset=config.dir_dataset,
        dir_result=config.dir_result,
        seed=config.seed,
        sample_eval_index=config.sample_eval_index,
    )
