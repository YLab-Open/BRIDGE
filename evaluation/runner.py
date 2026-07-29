from __future__ import annotations

from pathlib import Path

from .config import EvaluationConfig, make_task_args, set_evaluation_seed
from .reporting import (
    merge_and_write_performance,
    print_all_performance,
    print_task_performance,
)
from .task_registry import TASK_TYPE_ORDER, select_task_groups


def evaluate_task(task_name: str, task_cls, task_type: str, config: EvaluationConfig):
    """Evaluate one registered task across configured prompt modes."""
    task_args = make_task_args(config)
    task = task_cls(args=task_args, task=task_name)
    dict_prompt_model_performance = {}

    for prompt_mode in config.prompt_modes:
        dict_model_performance = task.evaluate_by_model(
            prompt_mode=prompt_mode,
            model_name=config.model_names,
            bootstrap=config.bootstrap,
        )
        path_file_performance = (
            Path(config.dir_performance) / f"{task.name}.{prompt_mode}.performance.json"
        )
        dict_prompt_model_performance[prompt_mode] = merge_and_write_performance(
            path_file_performance,
            dict_model_performance,
        )

    print_task_performance(dict_prompt_model_performance, task_type, config)
    print(flush=True)
    return dict_prompt_model_performance


def evaluate_all(config: EvaluationConfig):
    """Evaluate all selected task groups and task names in registry order."""
    selected_groups = select_task_groups(
        config.task_types,
        config.tasks,
        custom_tasks=config.custom_tasks,
    )
    for task_type in TASK_TYPE_ORDER:
        if task_type not in selected_groups:
            continue
        group = selected_groups[task_type]
        for task_name, task_cls in group["registry"].items():
            print(task_name, flush=True)
            evaluate_task(task_name, task_cls, task_type, config)


def run_config(config: EvaluationConfig):
    """Seed, evaluate, and optionally print the saved performance summary."""
    set_evaluation_seed(config.seed)
    evaluate_all(config)
    if config.print_summary:
        print_all_performance(config)
