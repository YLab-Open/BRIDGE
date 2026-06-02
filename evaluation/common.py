import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from util.runtime import load_yaml_config, parse_prompt_mode


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


def parse_task_types(value, valid_task_types: tuple[str, ...]) -> list[str]:
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


def set_evaluation_seed(seed: int):
    """Seed Python and NumPy for deterministic evaluation bookkeeping."""
    random.seed(seed)
    np.random.seed(seed)


def make_task_args(config: Any):
    """Build the minimal args object expected by task evaluation classes."""
    return SimpleNamespace(
        dir_dataset=config.dir_dataset,
        dir_result=config.dir_result,
        seed=config.seed,
        sample_eval_index=config.sample_eval_index,
    )


def read_json(path_file: str | Path):
    """Read a JSON file using UTF-8 encoding."""
    with Path(path_file).open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(data, path_file: str | Path):
    """Write JSON data with parent-directory creation."""
    path_file = Path(path_file)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
