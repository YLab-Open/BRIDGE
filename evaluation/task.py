from __future__ import annotations

import argparse
from pathlib import Path

from .config import EvaluationConfig, config_from_yaml
from .runner import run_config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("evaluate.task.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the project-task evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate project-specific clinical task result files from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an evaluation YAML config. Defaults to evaluation/evaluate.task.yaml.",
    )
    return parser


def config_from_args(argv=None) -> EvaluationConfig:
    """Parse CLI arguments and load the referenced evaluation config."""
    args = build_parser().parse_args(argv)
    return config_from_yaml(args.config)


def _custom_task_names(custom_tasks: dict) -> set[str]:
    task_names = set()
    for registry in custom_tasks.values():
        if isinstance(registry, dict):
            task_names.update(str(task_name) for task_name in registry)
    return task_names


def validate_task_config(config: EvaluationConfig) -> None:
    """Keep the generic task entry point scoped to explicitly registered tasks."""
    if not config.custom_tasks:
        raise ValueError(
            "evaluation.task requires `custom_tasks` in the YAML. "
            "Use `python -m evaluation.bridge` for built-in BRIDGE tasks."
        )
    if not config.tasks:
        raise ValueError(
            "evaluation.task requires an explicit `tasks` list so it does not "
            "accidentally evaluate all built-in BRIDGE tasks."
        )

    custom_task_names = _custom_task_names(config.custom_tasks)
    non_custom_tasks = [
        task_name for task_name in config.tasks if task_name not in custom_task_names
    ]
    if non_custom_tasks:
        raise ValueError(
            "evaluation.task is for custom tasks only. These tasks are not in "
            f"`custom_tasks`: {non_custom_tasks}. Use evaluation.bridge for "
            "built-in BRIDGE tasks or register the task class here."
        )


def main(argv=None):
    """CLI entry point for project-specific task evaluation."""
    config = config_from_args(argv)
    try:
        validate_task_config(config)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
    run_config(config)


if __name__ == "__main__":
    main()
