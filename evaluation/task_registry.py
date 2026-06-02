TASK_TYPE_ORDER = ("clf", "ext", "gen")


def get_default_models_to_print():
    """Return the display-order model list used by performance printers."""
    from dataset.config import get_models_evaluate

    return get_models_evaluate()


def get_task_groups():
    """Return task registries and printer functions for each BRIDGE task group."""
    from dataset.task import get_cls_for_clf, get_cls_for_ext, get_cls_for_gen
    from metric.classification import print_metrics_clf
    from metric.extraction import print_metrics_ext
    from metric.generation import print_metrics_gen

    return {
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


def select_task_groups(task_types: list[str], tasks: list[str] | None = None):
    """Select task registries by type and optionally filter by task name."""
    groups = get_task_groups()
    selected = {task_type: groups[task_type] for task_type in task_types}

    if tasks is None:
        return selected

    all_task_names = {
        task_name for group in groups.values() for task_name in group["registry"]
    }
    missing = [task_name for task_name in tasks if task_name not in all_task_names]
    if missing:
        raise ValueError(f"Task(s) not registered in BRIDGE evaluation: {missing}")

    task_filter = set(tasks)
    for group in selected.values():
        group["registry"] = {
            task_name: task_cls
            for task_name, task_cls in group["registry"].items()
            if task_name in task_filter
        }
    return selected
