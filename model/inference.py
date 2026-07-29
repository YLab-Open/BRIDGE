import json
import os
from pathlib import Path

import torch

from model.logits import (
    JsonlSidecarWriter,
    build_logit_rows,
    collect_vllm_text_outputs,
    get_task_target_config,
    logit_output_paths,
    logits_enabled,
    run_vllm_label_probability_probes,
    serialize_vllm_generation,
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - only for minimal test environments.
    def tqdm(iterable, **_kwargs):
        return iterable


def _log_text_preview(logger, title, text, max_chars=2000):
    """Log a prompt or response preview without flooding the task log."""
    text = str(text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated log preview, total chars={len(text)}]"
    logger.info(title)
    logger.info(text)


def _collect_vllm_outputs(args, response_generators):
    """Convert vLLM RequestOutput objects to saved prediction strings/lists."""
    return collect_vllm_text_outputs(args, response_generators)


def _unpack_prompt_batch(dataloader, batch_data):
    """Extract prompt text and copy worker-produced metadata to source rows."""
    batch_records = list(batch_data)
    if not batch_records:
        return []
    if not (
        isinstance(batch_records[0], dict)
        and "prompt" in batch_records[0]
        and "index" in batch_records[0]
    ):
        # Backward compatibility for caller-provided dataloaders of plain strings.
        return batch_records

    source_rows = getattr(getattr(dataloader, "dataset", None), "data", None)
    prompts = []
    for record in batch_records:
        prompts.append(record["prompt"])
        if source_rows is not None:
            source_row = source_rows[int(record["index"])]
            # Keep only the current sample's post-budget text in ``input``.
            # ``instruction`` and actual few-shot IDs are saved separately.
            source_row["input"] = record.get("input", record["prompt"])
            source_row["example"] = list(record.get("example") or [])
    return prompts


def save_result(
    logger,
    dataset,
    list_response,
    path_file_save,
    save_input=True,
    list_generation=None,
    list_logit=None,
    path_file_logit_save=None,
):
    """Save predictions into json file."""
    if len(dataset) != len(list_response):
        raise ValueError(
            f"Cannot save results because dataset and response lengths differ: "
            f"{len(dataset)} samples vs {len(list_response)} responses. "
            f"Target file: {path_file_save}"
        )
    if list_generation is not None and len(list_generation) != len(list_response):
        raise ValueError("list_generation must have the same length as list_response.")
    if list_logit is not None and len(list_logit) != len(list_response):
        raise ValueError("list_logit must have the same length as list_response.")

    list_to_save = _build_result_rows(
        dataset=dataset,
        list_response=list_response,
        save_input=save_input,
    )

    # Save the results to a json file
    path_file_save = Path(path_file_save)
    path_file_save.parent.mkdir(parents=True, exist_ok=True)
    with path_file_save.open("w", encoding="utf-8") as f:
        json.dump(list_to_save, f, indent=4, ensure_ascii=False)

    logger.info(f"Save: {path_file_save} ({len(list_to_save)} rows)")
    if (
        list_generation is not None or list_logit is not None
    ) and path_file_logit_save is not None:
        rows = build_logit_rows(
            dataset=dataset,
            list_response=list_response,
            list_generation=list_generation,
            list_logit=list_logit,
        )
        path_file_logit_save = Path(path_file_logit_save)
        path_file_logit_save.parent.mkdir(parents=True, exist_ok=True)
        with path_file_logit_save.open("w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)
        logger.info(f"Save logits: {path_file_logit_save} ({len(rows)} rows)")
    logger.info("\n" + "-" * 30 + "\n")
    return list_to_save


def _build_result_rows(dataset, list_response, save_input=True):
    """Build standard result rows without performing file I/O."""
    list_to_save = []
    for dict_data, pred in zip(dataset, list_response, strict=True):
        if save_input:
            # Save the original data and the response
            dict_result = dict_data.copy()
            dict_result["pred"] = pred
            list_to_save.append(dict_result)
        else:
            # Only save the response, not the original instruction and input
            dict_result = {
                "id": dict_data.get("id"),
                "type": dict_data.get("type"),
                "language": dict_data.get("language"),
                "output": dict_data.get("output"),
                "example": list(dict_data.get("example") or []),
                "pred": pred,
            }
            list_to_save.append(dict_result)
    return list_to_save


def partial_result_path(result_path):
    """Return the append-only checkpoint path for a standard result file."""
    return Path(result_path).with_suffix(".partial.jsonl")


def remove_partial_result(path_file_result):
    """Remove a completed run's internal checkpoint, if present."""
    path = partial_result_path(path_file_result)
    if path.exists():
        path.unlink()


def _append_partial_vllm_state(
    *,
    args,
    dataset,
    list_response,
    list_generation,
    list_logit,
    start,
):
    """Append only newly completed vLLM rows to a unified JSONL checkpoint."""
    end = len(list_response)
    if end <= start:
        return start
    path = partial_result_path(args.path_file_result)
    path.parent.mkdir(parents=True, exist_ok=True)
    # A legacy JSON checkpoint may have been resumed. Seed the new JSONL file
    # with that prefix before appending newly generated rows.
    if start and not path.exists():
        start = 0
    result_rows = _build_result_rows(
        dataset=dataset[start:end],
        list_response=list_response[start:end],
        save_input=args.save_input,
    )
    with path.open("a", encoding="utf-8") as handle:
        for offset, result_row in enumerate(result_rows, start=start):
            checkpoint_row = {"result": result_row}
            if list_generation is not None:
                checkpoint_row["generation"] = list_generation[offset]
            if list_logit is not None:
                checkpoint_row["logit"] = list_logit[offset]
            handle.write(json.dumps(checkpoint_row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    return end


def run_hf(args, logger, dataloader, model, tokenizer):
    """Run one Hugging Face generation pass over an already formatted dataloader."""
    from model.init import model_config_hf

    logger.info("Run via Hugging face style:")

    # set the decoding strategy and parameters
    tokenizer, model = model_config_hf(args, logger, model, tokenizer)

    list_response = []
    # model eval
    model.eval()
    with torch.no_grad():
        for idx_batch, batch_data in enumerate(dataloader):
            batch_data = _unpack_prompt_batch(dataloader, batch_data)
            batch_input = tokenizer(
                batch_data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_token_input,
                # after applying the chat template, remove one <bos>
                add_special_tokens=False,
            ).to(model.device)
            generation_kwargs = {
                "inputs": batch_input.input_ids,
                "attention_mask": batch_input.attention_mask,
                "max_new_tokens": args.max_token_output,
                "do_sample": model.generation_config.do_sample,
                "repetition_penalty": model.generation_config.repetition_penalty,
            }
            if model.generation_config.do_sample:
                generation_kwargs.update(
                    {
                        "temperature": model.generation_config.temperature,
                        "top_k": model.generation_config.top_k,
                        "top_p": model.generation_config.top_p,
                    }
                )
                sample_size = int(getattr(args, "sample_size", 1))
                if sample_size > 1:
                    generation_kwargs["num_return_sequences"] = sample_size

            batch_output = model.generate(**generation_kwargs)
            batch_output = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(
                    batch_input.input_ids.repeat_interleave(
                        generation_kwargs.get("num_return_sequences", 1), dim=0
                    ),
                    batch_output,
                )
            ]
            batch_response = tokenizer.batch_decode(
                batch_output, skip_special_tokens=True
            )
            if generation_kwargs.get("num_return_sequences", 1) > 1:
                sample_size = generation_kwargs["num_return_sequences"]
                batch_response = [
                    batch_response[i : i + sample_size]
                    for i in range(0, len(batch_response), sample_size)
                ]
            list_response += batch_response
            if idx_batch == 0:
                logger.info("\n" + "-" * 30 + "\n")
                _log_text_preview(logger, "Input: ", batch_data[0])
                _log_text_preview(logger, "Output: ", batch_response[0])
                logger.info("\n" + "-" * 30 + "\n")
            torch.cuda.empty_cache()

    return list_response


def _load_partial_vllm_state(
    *,
    args,
    logger,
    dataset,
    logit_paths,
    trace_enabled,
    label_probability_enabled,
):
    """Load a strictly validated saved prefix for explicit vLLM resume."""
    result_path = Path(args.path_file_result)
    logit_path = Path(logit_paths["logit_json"])
    checkpoint_path = partial_result_path(result_path)
    if checkpoint_path.exists():
        checkpoint_rows = []
        with checkpoint_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    checkpoint_rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid partial JSONL at line {line_number}: "
                        f"{checkpoint_path}"
                    ) from exc
        result_rows = [row.get("result") for row in checkpoint_rows]
        if any(not isinstance(row, dict) for row in result_rows):
            raise ValueError(
                f"Partial checkpoint contains an invalid result row: {checkpoint_path}"
            )
        logit_rows = [
            {
                "id": result_row.get("id"),
                "pred": result_row.get("pred"),
                "generation": checkpoint.get("generation"),
                "logit": checkpoint.get("logit"),
            }
            for result_row, checkpoint in zip(
                result_rows, checkpoint_rows, strict=True
            )
        ]
    elif not result_path.exists() and not logit_path.exists():
        return [], ([] if trace_enabled else None), (
            [] if label_probability_enabled else None
        ), 0
    else:
        if (trace_enabled or label_probability_enabled) and (
            result_path.exists() != logit_path.exists()
        ):
            raise ValueError(
                "Cannot resume logits inference unless both legacy result and "
                f"logit JSON exist: result={result_path.exists()}, "
                f"logit={logit_path.exists()}."
            )
        if not result_path.exists():
            return [], ([] if trace_enabled else None), (
                [] if label_probability_enabled else None
            ), 0
        with result_path.open("r", encoding="utf-8") as handle:
            result_rows = json.load(handle)
        if trace_enabled or label_probability_enabled:
            with logit_path.open("r", encoding="utf-8") as handle:
                logit_rows = json.load(handle)
        else:
            logit_rows = [
                {"id": row.get("id"), "pred": row.get("pred")}
                for row in result_rows
            ]

    if not isinstance(result_rows, list) or not isinstance(logit_rows, list):
        raise ValueError(
            "Partial result/logit artifacts must contain row lists."
        )
    if len(result_rows) != len(logit_rows):
        raise ValueError(
            "Partial result/logit row counts disagree: "
            f"{len(result_rows)} vs {len(logit_rows)}."
        )
    resume_count = len(result_rows)
    if resume_count > len(dataset):
        raise ValueError(
            f"Partial result has {resume_count} rows but dataset has only {len(dataset)}."
        )

    for index, (source, result_row, logit_row) in enumerate(
        zip(dataset[:resume_count], result_rows, logit_rows)
    ):
        source_id = source.get("id")
        result_id = result_row.get("id")
        logit_id = logit_row.get("id")
        if not (source_id == result_id == logit_id):
            raise ValueError(
                "Partial result is not an exact dataset-ID prefix at "
                f"row {index}: dataset={source_id!r}, result={result_id!r}, "
                f"logit={logit_id!r}."
            )
        if result_row.get("pred") != logit_row.get("pred"):
            raise ValueError(
                f"Partial result/logit predictions disagree at row {index}."
            )

    list_response = [row.get("pred") for row in result_rows]
    list_generation = None
    if trace_enabled:
        list_generation = [row.get("generation") for row in logit_rows]
        if any(not isinstance(value, dict) for value in list_generation):
            raise ValueError(
                "Cannot resume generation tracing because a saved generation "
                "payload is missing or invalid."
            )
    list_logit = None
    if label_probability_enabled:
        list_logit = [row.get("logit") for row in logit_rows]
        if any(not isinstance(value, dict) for value in list_logit):
            raise ValueError(
                "Cannot resume label probabilities because a saved logit "
                "payload is missing or invalid."
            )

    top_logits_path = Path(logit_paths["top_logits"])
    top_refs = [
        ((row.get("logit") or {}).get("top_logits_ref") or {}).get("line")
        for row in logit_rows
    ]
    if any(line is not None for line in top_refs) and not top_logits_path.exists():
        raise ValueError(
            "Saved logits reference a missing top-logits sidecar: "
            f"{top_logits_path}."
        )

    logger.info(
        "Resume partial vLLM inference: validated "
        f"{resume_count}/{len(dataset)} saved prefix rows."
    )
    return list_response, list_generation, list_logit, resume_count


def run_vllm(args, logger, dataset, dataloader, model, tokenizer=None):
    """Run one vLLM generation pass and optionally save partial results."""
    from model.init import model_config_vllm

    logger.info("Run via vLLM:")
    if logits_enabled(args) and tokenizer is None:
        raise ValueError("tokenizer is required when logits capture is enabled.")

    # set the decoding strategy and parameters
    sampling_params = model_config_vllm(args, logger)

    # whether to save partial results every n batches
    save_every_n_batches = getattr(args, "save_every_n_batches", None)

    logits_cfg = getattr(args, "logits", {}) or {}
    trace_enabled = bool((logits_cfg.get("generation_trace") or {}).get("enabled", True))
    task_target = get_task_target_config(args, getattr(args, "task_name", None))
    label_probability_enabled = bool(
        logits_cfg.get("label_probability", {}).get("enabled", False) and task_target
    )
    logit_paths = logit_output_paths(args.path_file_result)
    token_text_cache = {}

    if logits_enabled(args):
        trace_cfg = logits_cfg.get("generation_trace") or {}
        lp_cfg = logits_cfg.get("label_probability") or {}
        logger.info("Logits capture: enabled")
        logger.info(f" - result sidecar: {logit_paths['logit_json']}")
        logger.info(f" - top logits sidecar: {logit_paths['top_logits']}")
        logger.info(f" - run metadata: {logit_paths['metadata']}")
        logger.info(
            " - generation_trace: "
            f"enabled={trace_enabled}, "
            "top_k_logprobs_for_all_output_tokens="
            f"{trace_cfg.get('top_k_logprobs_for_all_output_tokens', trace_cfg.get('top_k_logprobs', 0))}"
        )
        logger.info(
            " - label_probability: "
            f"enabled={label_probability_enabled}, mode={lp_cfg.get('mode')}, "
            f"generation_logprobs_top_k={lp_cfg.get('generation_logprobs_top_k')}, "
            f"fallback_to_exact_probe={lp_cfg.get('fallback_to_exact_probe')}, "
            f"top_k_logits={lp_cfg.get('top_k_logits')}, "
            f"fail_on_error={lp_cfg.get('fail_on_error')}"
        )
        if task_target:
            logger.info(
                " - label task target: "
                f"answer_prefix={task_target.get('answer_prefix')!r}, "
                f"label_values={task_target.get('label_values')}, "
                f"label_groups={task_target.get('label_groups')}"
            )
        elif lp_cfg.get("enabled", False):
            logger.warning(
                " - label_probability is enabled but no task-specific logits target "
                f"was found for task={getattr(args, 'task_name', None)!r}."
            )

    list_response = []
    list_generation = [] if logits_enabled(args) and trace_enabled else None
    list_logit = [] if logits_enabled(args) and label_probability_enabled else None

    top_logits_path = (
        logit_paths["top_logits"]
        if logits_enabled(args)
        and label_probability_enabled
        and bool((logits_cfg.get("label_probability") or {}).get("save_top_logits", True))
        else None
    )

    resume_count = 0
    resume_partial = bool(getattr(args, "resume_partial_results", False))
    if resume_partial:
        (
            list_response,
            list_generation,
            list_logit,
            resume_count,
        ) = _load_partial_vllm_state(
            args=args,
            logger=logger,
            dataset=dataset,
            logit_paths=logit_paths,
            trace_enabled=bool(list_generation is not None),
            label_probability_enabled=bool(list_logit is not None),
        )
    else:
        remove_partial_result(args.path_file_result)
    partial_saved_count = resume_count

    samples_seen = 0
    logged_sample = False
    with JsonlSidecarWriter(
        top_logits_path,
        append=resume_count > 0,
    ) as sidecar_writer:
        for idx_batch, batch_data in enumerate(tqdm(dataloader, desc="vLLM batches")):
            batch_data = _unpack_prompt_batch(dataloader, batch_data)
            batch_start = samples_seen
            batch_end = batch_start + len(batch_data)
            samples_seen = batch_end
            if batch_end <= resume_count:
                continue
            if batch_start < resume_count:
                batch_data = batch_data[resume_count - batch_start :]
            logger.info(
                f"vLLM batch {idx_batch + 1}/{len(dataloader)}: "
                f"{len(batch_data)} prompts"
            )
            batch_prompts = list(batch_data)
            list_response_generator = model.generate(
                batch_prompts,
                sampling_params,
                use_tqdm=True,
            )
            list_response_batch = _collect_vllm_outputs(args, list_response_generator)
            list_response.extend(list_response_batch)

            if list_generation is not None:
                list_generation.extend(
                    [
                        serialize_vllm_generation(
                            request_output=response_generator,
                            tokenizer=tokenizer,
                            prompt_text=prompt,
                            cfg=logits_cfg,
                            token_text_cache=token_text_cache,
                        )
                        for response_generator, prompt in zip(
                            list_response_generator, batch_prompts
                        )
                    ]
                )

            if list_logit is not None:
                list_logit.extend(
                    run_vllm_label_probability_probes(
                        args=args,
                        model=model,
                        tokenizer=tokenizer,
                        request_outputs=list_response_generator,
                        prompt_texts=batch_prompts,
                        target_cfg=task_target,
                        sidecar_writer=sidecar_writer,
                        token_text_cache=token_text_cache,
                    )
                )

            if not logged_sample:
                logged_sample = True
                logger.info("\n" + "-" * 30 + "\n")
                if dataset:
                    source = dataset[len(list_response) - len(list_response_batch)]
                    logger.info(
                        "Sample source: "
                        f"id={source.get('id')}, output={source.get('output')!r}"
                    )
                _log_text_preview(logger, "Input: ", batch_data[0])
                _log_text_preview(logger, "Output: ", list_response_batch[0])
                if list_generation:
                    first_generation = list_generation[0]
                    logger.info(
                        "Generation trace sample: "
                        f"{json.dumps(first_generation.get('outputs', [])[:1], ensure_ascii=False)[:2000]}"
                    )
                if list_logit:
                    first_logit = list_logit[0]
                    logger.info(
                        "Logit sample: "
                        f"{json.dumps(first_logit, ensure_ascii=False)[:2000]}"
                    )
                logger.info("\n" + "-" * 30 + "\n")

            should_save_partial = (
                save_every_n_batches is not None
                and (idx_batch + 1) % int(save_every_n_batches) == 0
            )
            if should_save_partial:
                partial_saved_count = _append_partial_vllm_state(
                    args=args,
                    dataset=dataset,
                    list_response=list_response,
                    list_generation=list_generation,
                    list_logit=list_logit,
                    start=partial_saved_count,
                )
                logger.info(
                    "Checkpoint append: "
                    f"{partial_result_path(args.path_file_result)} "
                    f"({partial_saved_count} rows total)"
                )

    if logits_enabled(args):
        return list_response, list_generation, list_logit

    return list_response
