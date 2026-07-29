import argparse
from pathlib import Path

from .config import EvaluationConfig, config_from_yaml
from .runner import run_config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("evaluate.BRIDGE.yaml")


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""
    parser = argparse.ArgumentParser(
        description="Evaluate BRIDGE result files from a YAML config."
    )
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an evaluation YAML config. Defaults to evaluation/evaluate.BRIDGE.yaml.",
    )
    return parser


def config_from_args(argv=None) -> EvaluationConfig:
    """Parse CLI arguments and load the referenced evaluation config."""
    args = build_parser().parse_args(argv)
    return config_from_yaml(args.config)


def validate_bridge_config(config: EvaluationConfig) -> None:
    """Keep the BRIDGE entry point scoped to built-in BRIDGE tasks."""
    if config.custom_tasks:
        raise ValueError(
            "evaluation.bridge is reserved for built-in BRIDGE tasks. "
            "Use `python -m evaluation.task --config ...` for configs with custom_tasks."
        )


def main(argv=None):
    """CLI entry point for BRIDGE evaluation."""
    config = config_from_args(argv)
    try:
        validate_bridge_config(config)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from None
    run_config(config)


if __name__ == "__main__":
    main()
