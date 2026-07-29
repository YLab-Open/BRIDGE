"""Unified commercial API runner for BRIDGE task inference.

This module keeps the public API path close to the local inference path: it
loads BRIDGE task JSON, builds chat prompts, calls one commercial provider, and
writes `<task>-<prompt>-<decoding>-<seed>.result.json` files that existing
evaluation code can consume.
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
from pathlib import Path
from typing import Any

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only for minimal test environments.
    def tqdm(iterable, **_kwargs):
        return iterable

from model.api.config import (
    as_bool,
    effective_run_config,
    load_yaml,
    models_from_config,
    parse_numeric,
    redact,
    require_model_section,
    resolve_api_settings,
    validate_api_run_config,
)
from model.api.prompts import (
    build_messages_for_dataset,
    custom_id,
    load_task_dataset,
)
from model.api.provider import (
    AzureProvider,
    make_provider,
)
from model.api.results import (
    ApiRunResult,
    ExperimentResult,
    QueryResult,
    summarize_query_results,
)
from model.metadata import write_result_metadata
from util.io import write_json, write_jsonl, write_jsonl_iter
from util.runtime import load_yaml_config, parse_path_list, parse_prompt_mode


def run_single_queries(
    provider_obj: Any,
    dataset: list[dict[str, Any]],
    messages_by_id: dict[str, list[dict[str, str]]],
    task_name: str,
    model_name: str,
    prompt_mode: str,
    split: str,
    max_token_output: int,
    trace_dir: Path,
    workers: int,
    save_trace: bool = True,
    save_raw_response: bool = True,
) -> list[QueryResult]:
    """Run provider single-query calls concurrently and save raw traces."""
    if save_trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for row in dataset:
        sid = str(row["id"])
        requests.append(
            {
                "custom_id": custom_id(task_name, model_name, prompt_mode, split, sid),
                "messages": messages_by_id[sid],
            }
        )
    if save_trace:
        write_jsonl(requests, str(trace_dir / "requests.jsonl"))

    results_by_id: dict[str, QueryResult] = {}

    def _call(request: dict[str, Any]) -> QueryResult:
        return provider_obj.query_single(
            messages=request["messages"],
            cid=request["custom_id"],
            max_token_output=max_token_output,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        futures = {executor.submit(_call, request): request for request in requests}
        response_file = None
        if save_trace:
            response_file = (trace_dir / "responses.jsonl").open("w", encoding="utf-8")
        try:
            for future in tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc=f"{model_name} {task_name} API",
            ):
                result = future.result()
                results_by_id[result.custom_id.rsplit("|", 1)[-1]] = result
                if response_file is not None:
                    response_file.write(
                        json.dumps(
                            result.to_trace_dict(include_raw=save_raw_response),
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    response_file.flush()
        finally:
            if response_file is not None:
                response_file.close()

    ordered = []
    for row in dataset:
        sid = str(row["id"])
        ordered.append(
            results_by_id.get(
                sid,
                QueryResult(
                    custom_id=custom_id(task_name, model_name, prompt_mode, split, sid),
                    error="Missing result after API query",
                ),
            )
        )
    return ordered


def result_rows_from_query_results(
    dataset: list[dict[str, Any]],
    messages_by_id: dict[str, list[dict[str, str]]],
    results: list[QueryResult],
    save_input: bool = True,
    examples_by_id: dict[str, list[Any]] | None = None,
) -> list[dict[str, Any]]:
    """Join source rows, exact API inputs, actual example IDs, and predictions."""
    examples_by_id = examples_by_id or {}
    result_by_sample_id = {
        result.custom_id.rsplit("|", 1)[-1]: result
        for result in results
    }
    rows = []
    for row in dataset:
        sid = str(row["id"])
        result = result_by_sample_id.get(sid, QueryResult(custom_id=sid, error="Missing result"))
        if save_input:
            out = row.copy()
            # Store the exact formatted/truncated messages submitted to the
            # provider so the request can be reproduced from the result row.
            out["input"] = messages_by_id.get(sid, [])
        else:
            out = {
                key: row.get(key)
                for key in ("task", "id", "type", "language", "split", "output")
                if key in row
            }
        out["example"] = list(examples_by_id.get(sid) or [])
        out["pred"] = result.text if not result.error else ""
        out["error"] = result.error
        if result.summary:
            out["summary"] = result.summary
        if result.tokens_in or result.tokens_out:
            out["token_input"] = result.tokens_in
            out["token_output"] = result.tokens_out
        rows.append(out)
    return rows


def run_api_experiment(
    provider_name: str,
    model_name: str,
    run_cfg: dict[str, Any],
    api_model_cfg: dict[str, Any],
    api_key_cfg: dict[str, Any],
    task_name: str,
    prompt_mode: str,
    decoding_strategy: str,
    seed: int,
    dry_run: bool = False,
) -> ExperimentResult:
    """Run one provider/model/task/prompt experiment and save result JSON."""
    run_cfg = effective_run_config(run_cfg, provider_name, decoding_strategy, seed)
    split = str(run_cfg.get("mode", "test"))
    dir_dataset = parse_path_list(run_cfg.get("dir_dataset"), "dir_dataset")
    dir_result = parse_path_list(run_cfg.get("dir_result"), "dir_result")
    max_token_input = int(parse_numeric(run_cfg.get("max_token_input"), 1024 * 64))
    max_token_output = int(parse_numeric(run_cfg.get("max_token_output"), 1024 * 16))
    require_registered = as_bool(run_cfg.get("require_model_registered"), True)
    model_cfg = require_model_section(
        api_model_cfg=api_model_cfg,
        provider=provider_name,
        model_name=model_name,
        require_registered=require_registered,
    )
    result_dir = Path(dir_result[0]) / task_name / model_name
    exp_name = f"{task_name}-{prompt_mode}-{decoding_strategy}-{seed}"
    trace_dir = result_dir / "api_trace" / exp_name
    result_path = result_dir / f"{exp_name}.result.json"

    if dry_run:
        print(f"Dry run: would write {result_path}")
        return ExperimentResult(
            result_path=result_path,
            trace_dir=trace_dir,
            provider=provider_name,
            model_name=model_name,
            task_name=task_name,
            prompt_mode=prompt_mode,
            decoding_strategy=decoding_strategy,
            seed=seed,
            num_samples=0,
            dry_run=True,
        )

    api_settings = resolve_api_settings(provider_name, run_cfg, api_key_cfg)
    provider_obj = make_provider(provider_name, model_name, model_cfg, run_cfg, api_settings)
    save_input = as_bool(run_cfg.get("save_input"), True)
    save_trace = as_bool(run_cfg.get("save_trace"), True)
    save_raw_response = as_bool(run_cfg.get("save_raw_response"), True)
    allow_partial = as_bool(run_cfg.get("allow_partial"), False)

    dataset, examples = load_task_dataset(
        task_name=task_name,
        prompt_mode=prompt_mode,
        split=split,
        dir_dataset=dir_dataset,
        max_data_size=run_cfg.get("max_data_size"),
    )
    prompt_build = build_messages_for_dataset(
        dataset=dataset,
        examples=examples,
        model_name=model_cfg.get("deploy_name") or model_name,
        max_token_input=max_token_input,
        return_example_ids=True,
    )
    # Accept the historical messages-only return shape from integrations that
    # monkeypatch or wrap this helper.
    if isinstance(prompt_build, tuple):
        messages_by_id, examples_by_id = prompt_build
    else:
        messages_by_id, examples_by_id = prompt_build, {}

    error_summary_path = (
        trace_dir / "error_summary.json"
        if save_trace
        else result_dir / f"{exp_name}.error_summary.json"
    )

    result_dir.mkdir(parents=True, exist_ok=True)
    run_meta = {
        "provider": provider_name,
        "model_name": model_name,
        "model_config": model_cfg,
        "api_settings": redact(api_settings),
        "task_name": task_name,
        "prompt_mode": prompt_mode,
        "decoding_strategy": decoding_strategy,
        "seed": seed,
        "split": split,
        "num_samples": len(dataset),
        "max_token_input": max_token_input,
        "max_token_output": max_token_output,
        "save_input": save_input,
        "save_trace": save_trace,
        "save_raw_response": save_raw_response,
        "allow_partial": allow_partial,
        "run_config": redact(run_cfg),
    }
    if save_trace:
        trace_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_meta, str(trace_dir / "run_config.json"))

    if provider_name == "azure" and isinstance(provider_obj, AzureProvider) and provider_obj.should_use_batch():
        records, endpoint = provider_obj.build_batch_records(
            task_name=task_name,
            prompt_mode=prompt_mode,
            split=split,
            dataset=dataset,
            messages_by_id=messages_by_id,
            max_token_output=max_token_output,
        )
        results = provider_obj.run_batch(records, endpoint, trace_dir, keep_artifacts=save_trace)
        if save_trace:
            write_jsonl_iter(
                (result.to_trace_dict(include_raw=save_raw_response) for result in results),
                trace_dir / "responses.jsonl",
            )
    else:
        workers = int(api_settings.get("single_parallel_workers", api_settings.get("parallel_workers", 4)) or 4)
        results = run_single_queries(
            provider_obj=provider_obj,
            dataset=dataset,
            messages_by_id=messages_by_id,
            task_name=task_name,
            model_name=model_name,
            prompt_mode=prompt_mode,
            split=split,
            max_token_output=max_token_output,
            trace_dir=trace_dir,
            workers=workers,
            save_trace=save_trace,
            save_raw_response=save_raw_response,
        )

    rows = result_rows_from_query_results(
        dataset=dataset,
        messages_by_id=messages_by_id,
        results=results,
        save_input=save_input,
        examples_by_id=examples_by_id,
    )
    write_json(rows, str(result_path))
    num_errors, error_examples = summarize_query_results(results)
    summary = {
        "provider": provider_name,
        "model_name": model_name,
        "task_name": task_name,
        "prompt_mode": prompt_mode,
        "decoding_strategy": decoding_strategy,
        "seed": seed,
        "num_samples": len(dataset),
        "num_errors": num_errors,
        "error_examples": error_examples,
        "result_path": str(result_path),
    }
    if num_errors:
        write_json(summary, str(error_summary_path))

    prompt_base, num_example = parse_prompt_mode(prompt_mode)
    task_data_path = next(
        (
            Path(root) / f"{task_name}.SFT.json"
            for root in dir_dataset
            if (Path(root) / f"{task_name}.SFT.json").is_file()
        ),
        None,
    )
    example_path = None
    if num_example is not None:
        example_dir = "example" if prompt_base == "direct" else "example_cot"
        example_path = next(
            (
                Path(root) / example_dir / f"{task_name}.example.json"
                for root in dir_dataset
                if (
                    Path(root) / example_dir / f"{task_name}.example.json"
                ).is_file()
            ),
            None,
        )
    metadata_sources = run_cfg.get("_metadata_sources") or {}
    metadata_paths = write_result_metadata(
        result_path=result_path,
        run_type=f"api_{provider_name}",
        n_records=len(rows),
        effective_config={
            "run": redact(run_cfg),
            "model": redact(model_cfg),
            "api": redact(api_settings),
        },
        source_files={
            "main_config": metadata_sources.get("main_config"),
            "model_registry": metadata_sources.get("model_registry"),
            "task_data": task_data_path,
            "few_shot_examples": example_path,
        },
        related_files={
            "requests": trace_dir / "requests.jsonl" if save_trace else None,
            "responses": trace_dir / "responses.jsonl" if save_trace else None,
            "trace_config": trace_dir / "run_config.json" if save_trace else None,
            "azure_batch_input": (
                trace_dir / "azure_batch_input.jsonl"
                if save_trace and provider_name == "azure"
                else None
            ),
            "azure_batch_output": (
                trace_dir / "azure_batch_output.jsonl"
                if save_trace and provider_name == "azure"
                else None
            ),
            "azure_batch_status": (
                trace_dir / "azure_batch_status.jsonl"
                if save_trace and provider_name == "azure"
                else None
            ),
            "azure_batch_metadata": (
                trace_dir / "azure_batch_metadata.json"
                if save_trace and provider_name == "azure"
                else None
            ),
            "error_summary": error_summary_path if num_errors else None,
        },
        experiment={
            "provider": provider_name,
            "model_name": model_name,
            "task_name": task_name,
            "prompt_mode": prompt_mode,
            "decoding_strategy": decoding_strategy,
            "seed": seed,
            "split": split,
        },
        extra={
            "num_errors": num_errors,
            "error_examples": error_examples,
        },
    )
    print(f"Metadata: {metadata_paths['metadata']}", flush=True)

    experiment_result = ExperimentResult(
        result_path=result_path,
        trace_dir=trace_dir,
        provider=provider_name,
        model_name=model_name,
        task_name=task_name,
        prompt_mode=prompt_mode,
        decoding_strategy=decoding_strategy,
        seed=seed,
        num_samples=len(dataset),
        num_errors=num_errors,
        error_examples=error_examples,
        dry_run=False,
    )
    if num_errors and not allow_partial:
        raise RuntimeError(
            f"API experiment wrote {result_path} but {num_errors}/{len(dataset)} samples "
            f"failed. See {error_summary_path}. Set allow_partial: true only when you "
            "intentionally want to keep and optionally evaluate partial outputs."
        )
    return experiment_result


def run_api_for_experiment(
    provider_name: str,
    model_name: str,
    run_cfg: dict[str, Any],
    api_model_cfg: dict[str, Any],
    api_key_cfg: dict[str, Any],
    task_name: str,
    prompt_mode: str,
    decoding_strategy: str,
    seed: int,
    dry_run: bool = False,
) -> Path:
    """Backward-compatible wrapper returning only the result path."""
    return run_api_experiment(
        provider_name=provider_name,
        model_name=model_name,
        run_cfg=run_cfg,
        api_model_cfg=api_model_cfg,
        api_key_cfg=api_key_cfg,
        task_name=task_name,
        prompt_mode=prompt_mode,
        decoding_strategy=decoding_strategy,
        seed=seed,
        dry_run=dry_run,
    ).result_path


def run_api_from_config(
    config_path: str | Path = "configs/API.provider.azure.yaml",
    provider: str | None = None,
    model_name: str | None = None,
    model_names: list[str] | None = None,
    api_model_config_path: str | Path = "configs/API.model.yaml",
    api_key_file: str | Path | None = "configs/API.key.yaml",
    dry_run: bool = False,
    run_cfg: dict[str, Any] | None = None,
) -> ApiRunResult:
    """
    Run API inference from YAML.

    This is the import-friendly entry point for Python scripts and notebooks:

    ```
    from model.api_runner import run_api_from_config
    run = run_api_from_config("configs/API.provider.azure.yaml", model_name="gpt-4o")
    ```
    """
    run_cfg_supplied = run_cfg is not None
    run_cfg = (
        copy.deepcopy(run_cfg)
        if run_cfg_supplied
        else load_yaml_config(str(config_path))
    )
    run_cfg["_metadata_sources"] = {
        "main_config": (
            None
            if run_cfg_supplied or not config_path
            else str(config_path)
        ),
        "model_registry": (
            str(api_model_config_path) if api_model_config_path else None
        ),
    }
    validate_api_run_config(run_cfg)
    provider_name = (provider or str(run_cfg.get("provider", "azure"))).lower()
    if provider_name not in {"azure", "claude", "gemini"}:
        raise ValueError("provider must be one of: azure, claude, gemini")

    if model_name and model_names:
        raise ValueError("Pass either model_name or model_names, not both.")
    selected_models = [model_name] if model_name else (model_names or models_from_config(run_cfg))
    api_model_cfg = load_yaml(api_model_config_path, required=True)
    api_key_cfg = load_yaml(api_key_file, required=False)

    experiment_results: list[ExperimentResult] = []
    for selected_model in selected_models:
        for task_name in run_cfg["tasks"]:
            for prompt_mode, decoding_strategy, seed in run_cfg["experiments"]:
                experiment_results.append(
                    run_api_experiment(
                        provider_name=provider_name,
                        model_name=str(selected_model),
                        run_cfg=run_cfg,
                        api_model_cfg=api_model_cfg,
                        api_key_cfg=api_key_cfg,
                        task_name=str(task_name),
                        prompt_mode=str(prompt_mode),
                        decoding_strategy=str(decoding_strategy).lower(),
                        seed=int(seed),
                        dry_run=dry_run,
                    )
                )

    return ApiRunResult(
        provider=provider_name,
        model_names=[str(model) for model in selected_models],
        experiments=experiment_results,
    )
