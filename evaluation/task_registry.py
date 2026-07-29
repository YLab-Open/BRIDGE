from __future__ import annotations

import importlib
from typing import Any

TASK_TYPE_ORDER = ("clf", "ext", "gen")


def get_default_models_to_print():
    """Return the display-order model list used by performance printers."""
    from dataset.config import get_models_evaluate

    return get_models_evaluate()


def import_object(path: str):
    """Import `module:object` or `module.object` and return the object."""
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Import path must be a non-empty string.")
    path = path.strip()
    if ":" in path:
        module_name, object_name = path.split(":", 1)
    else:
        module_name, object_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def normalize_custom_tasks(custom_tasks: dict[str, Any] | None) -> dict[str, dict[str, type]]:
    """
    Parse custom task registrations from evaluation YAML.

    Expected shape:

    custom_tasks:
      ext:
        My-Task: my_package.my_module:MyTaskClass
    """
    if not custom_tasks:
        return {}
    if not isinstance(custom_tasks, dict):
        raise ValueError("custom_tasks must be a mapping from task type to task registrations.")

    normalized: dict[str, dict[str, type]] = {}
    for task_type, registry in custom_tasks.items():
        task_type = str(task_type)
        if task_type not in TASK_TYPE_ORDER:
            raise ValueError(
                f"Invalid custom task type `{task_type}`. Use one of: {', '.join(TASK_TYPE_ORDER)}."
            )
        if not isinstance(registry, dict):
            raise ValueError(f"custom_tasks.{task_type} must be a mapping.")
        normalized[task_type] = {}
        for task_name, class_ref in registry.items():
            task_cls = import_object(str(class_ref))
            normalized[task_type][str(task_name)] = task_cls
    return normalized


def get_task_groups(custom_tasks: dict[str, Any] | None = None):
    """Return task registries and printer functions for each BRIDGE task group."""
    from dataset.task import get_cls_for_clf, get_cls_for_ext, get_cls_for_gen
    from metric.classification import print_metrics_clf
    from metric.extraction import print_metrics_ext
    from metric.generation import print_metrics_gen

    groups = {
        "clf": {
            "name": "Classification",
            "registry": get_cls_for_clf(),
            "printer": print_metrics_clf,
        },
        "ext": {
            "name": "Extraction",
            "registry": get_cls_for_ext(),
            "printer": print_metrics_ext,
        },
        "gen": {
            "name": "Generation",
            "registry": get_cls_for_gen(),
            "printer": print_metrics_gen,
        },
    }
    for task_type, registry in normalize_custom_tasks(custom_tasks).items():
        groups[task_type]["registry"].update(registry)
    return groups


def select_task_groups(
    task_types: list[str],
    tasks: list[str] | None = None,
    custom_tasks: dict[str, Any] | None = None,
):
    """Select task registries by type and optionally filter by task name."""
    groups = get_task_groups(custom_tasks=custom_tasks)
    selected = {task_type: groups[task_type] for task_type in task_types}

    if tasks is None:
        return selected

    all_task_names = {
        task_name for group in groups.values() for task_name in group["registry"]
    }
    missing = [task_name for task_name in tasks if task_name not in all_task_names]
    if missing:
        raise ValueError(f"Task(s) not registered in evaluation: {missing}")

    task_filter = set(tasks)
    for group in selected.values():
        group["registry"] = {
            task_name: task_cls
            for task_name, task_cls in group["registry"].items()
            if task_name in task_filter
        }
    return selected
