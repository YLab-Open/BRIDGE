from pathlib import Path
from typing import Any

from .common import read_json, write_json
from .task_registry import TASK_TYPE_ORDER, get_task_groups, select_task_groups


def get_models_to_print(dict_model_performance: dict, config: Any) -> list[str] | None:
    """Choose display model order without hiding models discovered from results."""
    if config.model_names:
        return config.model_names
    preferred = [model for model in config.models_to_print if model in dict_model_performance]
    extra = sorted(set(dict_model_performance) - set(preferred))
    return preferred + extra


def merge_and_write_performance(path_file: str | Path, dict_model_performance: dict):
    """Merge one task/prompt performance result into a JSON file."""
    path_file = Path(path_file)
    if path_file.exists():
        dict_performance_old = read_json(path_file)
    else:
        dict_performance_old = {}

    dict_performance_old.update(dict_model_performance)
    write_json(dict_performance_old, path_file)
    return dict_performance_old


def print_task_performance(
    dict_prompt_model_performance: dict,
    task_type: str,
    config: Any,
):
    """Print the metric summary for one task across configured prompt modes."""
    printer = get_task_groups()[task_type]["printer"]
    dict_mode_performance = {}
    for prompt_mode in config.prompt_modes:
        list_model = get_models_to_print(dict_prompt_model_performance[prompt_mode], config)
        print(f"Prompt Mode: {prompt_mode}")
        str_metrics = printer(
            dict_prompt_model_performance[prompt_mode],
            list_model=list_model,
            flag_print_missing=False,
        )
        print(str_metrics)
        print("===============================")
        dict_mode_performance[prompt_mode] = str_metrics
    return dict_mode_performance


def print_all_performance(config: Any):
    """Print saved performance files for the selected BRIDGE task groups."""
    selected_groups = select_task_groups(config.task_types, config.tasks)
    for task_type in TASK_TYPE_ORDER:
        if task_type not in selected_groups:
            continue
        group = selected_groups[task_type]
        print(group["name"])
        printer = group["printer"]
        for prompt_mode in config.prompt_modes:
            print(f"Prompt Mode: {prompt_mode}")
            for task_name in group["registry"]:
                path_file_performance = (
                    Path(config.dir_performance)
                    / f"{task_name}.{prompt_mode}.performance.json"
                )
                if not path_file_performance.exists():
                    print(f"Missing performance file: {path_file_performance}")
                    continue
                dict_model_performance = read_json(path_file_performance)
                list_model = get_models_to_print(dict_model_performance, config)
                print(
                    printer(
                        dict_model_performance,
                        list_model,
                        flag_print_missing=False,
                    )
                )
            print("===============================")
        print()


def load_performance_file(
    task_name: str,
    prompt_mode: str,
    dir_performance: str | Path = "performance",
) -> dict:
    """Load one `{task}.{prompt_mode}.performance.json` file."""
    return read_json(Path(dir_performance) / f"{task_name}.{prompt_mode}.performance.json")
