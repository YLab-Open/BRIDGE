#!/usr/bin/env python3
"""Run BRIDGE task inference through commercial model APIs."""

from __future__ import annotations

import argparse
import datetime

from model.api_runner import run_api_from_config
from util.runtime import load_yaml_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run clinical-llm-benchmark tasks through Azure, Claude, or Gemini APIs."
    )
    parser.add_argument(
        "--provider",
        choices=["azure", "claude", "gemini"],
        default=None,
        help="Commercial API provider. Defaults to the `provider` field in the run config.",
    )
    parser.add_argument(
        "--model_name",
        "--model",
        default=None,
        help="Optional single model override. If omitted, all models in `models` from the run config are run.",
    )
    parser.add_argument(
        "--config",
        default="configs/API.provider.azure.yaml",
        help="API run YAML, e.g. configs/API.provider.azure.yaml.",
    )
    parser.add_argument(
        "--api_model_config",
        "--api-model-config",
        default="configs/API.model.yaml",
        help="Provider/model capability YAML.",
    )
    parser.add_argument(
        "--api_key_file",
        "--api-key-file",
        default="configs/API.key.yaml",
        help="Optional YAML containing provider credentials. Environment variables are also supported.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and print planned result paths without calling APIs or writing result/trace files.",
    )
    eval_group = parser.add_mutually_exclusive_group()
    eval_group.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate generated result files after inference.",
    )
    eval_group.add_argument(
        "--no-evaluate",
        action="store_true",
        help="Skip evaluation even if the run YAML has evaluate_after_run: true.",
    )
    partial_group = parser.add_mutually_exclusive_group()
    partial_group.add_argument(
        "--allow-partial",
        action="store_true",
        help="Keep partial result files when one or more API calls fail.",
    )
    partial_group.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Fail the run when one or more sample-level API calls fail.",
    )
    return parser


def run_evaluation(run_cfg: dict, model_names: list[str], prompt_modes: list[str]) -> None:
    """Evaluate result files with the shared evaluation runner."""
    from evaluation.config import EvaluationConfig, as_list, parse_task_types
    from evaluation.runner import run_config

    eval_cfg = EvaluationConfig(
        dir_dataset=as_list(
            run_cfg.get("dir_dataset", ["dataset_raw"]),
            "dir_dataset",
            allow_none=False,
        ),
        dir_result=as_list(
            run_cfg.get("dir_result", ["result"]),
            "dir_result",
            allow_none=False,
        ),
        dir_performance=str(run_cfg.get("dir_performance", "performance")),
        prompt_modes=prompt_modes,
        model_names=model_names,
        task_types=parse_task_types(run_cfg.get("task_types")),
        tasks=as_list(run_cfg.get("tasks"), "tasks", allow_none=True),
        custom_tasks=run_cfg.get("custom_tasks", {}) or {},
        bootstrap=int(run_cfg.get("bootstrap", 1000)),
        seed=int(run_cfg.get("evaluation_seed", 42)),
        sample_eval_index=int(run_cfg.get("sample_eval_index", 0)),
        print_summary=bool(run_cfg.get("print_summary", True)),
    )
    run_config(eval_cfg)


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    run_cfg = load_yaml_config(args.config)
    if args.allow_partial:
        run_cfg["allow_partial"] = True
    if args.fail_on_error:
        run_cfg["allow_partial"] = False
    if args.no_evaluate:
        run_cfg["evaluate_after_run"] = False

    provider = args.provider or str(run_cfg.get("provider", "azure")).lower()
    model_names_preview = [args.model_name] if args.model_name else run_cfg.get("models", [])
    if isinstance(model_names_preview, str):
        model_names_preview = [model_names_preview]

    print("=" * 60)
    print(f"API provider: {provider}")
    print(f"Models: {', '.join(str(item) for item in model_names_preview)}")
    print(f"Config: {args.config}")
    print(f"Start: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    run_result = run_api_from_config(
        config_path=args.config,
        provider=provider,
        model_name=args.model_name,
        api_model_config_path=args.api_model_config,
        api_key_file=args.api_key_file,
        dry_run=args.dry_run,
        run_cfg=run_cfg,
    )
    for experiment in run_result.experiments:
        print(f"Result path: {experiment.result_path}")
        if experiment.num_errors:
            print(f"Errors: {experiment.num_errors}/{experiment.num_samples}")
        print("-" * 60)

    should_evaluate = (
        args.evaluate
        or (bool(run_cfg.get("evaluate_after_run", False)) and not args.no_evaluate)
    )
    if should_evaluate and not args.dry_run:
        has_partial_errors = run_result.num_errors > 0
        evaluate_on_partial = bool(run_cfg.get("evaluate_on_partial", False))
        if has_partial_errors and not evaluate_on_partial:
            print("Skip evaluation because partial API errors were detected.")
        else:
            print("Evaluating generated result files...")
            prompt_modes_run = sorted({item.prompt_mode for item in run_result.experiments})
            run_evaluation(run_cfg, run_result.model_names, prompt_modes_run)

    print(f"End: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
