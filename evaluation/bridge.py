import argparse
from dataclasses import dataclass, field
from pathlib import Path

from .common import (
    as_list,
    load_evaluation_yaml,
    make_task_args,
    parse_models,
    parse_prompt_modes,
    parse_task_types,
    set_evaluation_seed,
)
from .performance import (
    merge_and_write_performance,
    print_all_performance,
    print_task_performance,
)
from .task_registry import (
    TASK_TYPE_ORDER,
    get_default_models_to_print,
    select_task_groups,
)


DEFAULT_CONFIG_PATH = Path(__file__).with_name("evaluate_BRIDGE.yaml")


@dataclass
class EvaluationConfig:
    """Runtime configuration for BRIDGE result evaluation."""

    dir_dataset: list[str] = field(default_factory=lambda: ["dataset_raw"])
    dir_result: list[str] = field(default_factory=lambda: ["result"])
    dir_performance: str = "performance"
    prompt_modes: list[str] = field(default_factory=lambda: ["direct"])
    model_names: list[str] | None = None
    models_to_print: list[str] = field(default_factory=get_default_models_to_print)
    task_types: list[str] = field(default_factory=lambda: list(TASK_TYPE_ORDER))
    tasks: list[str] | None = None
    bootstrap: int = 1000
    seed: int = 42
    sample_eval_index: int = 0
    print_summary: bool = True


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate BRIDGE result files from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an evaluation YAML config. Defaults to evaluation/evaluate_BRIDGE.yaml.",
    )
    return parser


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
        task_types=parse_task_types(cfg.get("task_types"), TASK_TYPE_ORDER),
        tasks=as_list(cfg.get("tasks"), "tasks", allow_none=True),
        bootstrap=bootstrap,
        seed=int(cfg.get("seed", 42)),
        sample_eval_index=sample_eval_index,
        print_summary=bool(cfg.get("print_summary", True)),
    )


def config_from_args(argv=None) -> EvaluationConfig:
    """Parse CLI arguments and load the referenced evaluation config."""
    args = build_parser().parse_args(argv)
    return config_from_yaml(args.config)


def evaluate_task(task_name: str, task_cls, task_type: str, config: EvaluationConfig):
    """Evaluate one registered BRIDGE task across configured prompt modes."""
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
    print()
    return dict_prompt_model_performance


def evaluate_all(config: EvaluationConfig):
    """Evaluate all selected task groups and task names in registry order."""
    selected_groups = select_task_groups(config.task_types, config.tasks)
    for task_type in TASK_TYPE_ORDER:
        if task_type not in selected_groups:
            continue
        group = selected_groups[task_type]
        for task_name, task_cls in group["registry"].items():
            print(task_name)
            evaluate_task(task_name, task_cls, task_type, config)


def main(argv=None):
    """CLI entry point for BRIDGE evaluation."""
    config = config_from_args(argv)
    set_evaluation_seed(config.seed)
    evaluate_all(config)
    if config.print_summary:
        print_all_performance(config)


if __name__ == "__main__":
    main()
