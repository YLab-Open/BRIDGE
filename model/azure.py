import os
from typing import Any, Dict, List, Optional, Sequence

import tiktoken
from openai import AzureOpenAI

from dataset.process import get_formatted_chat
from util.tool import load_json, load_jsonl, write_json, write_jsonl
from util.runtime import parse_prompt_mode
from model.azure_utils import (
    MODEL_PRICE_PER_1M,
    apply_output_token_key as _apply_output_token_key,
    build_chat_completion_batch_body,
    build_messages as _build_messages,
    build_response_batch_body,
    cost_calculation_token,
    cost_calculation_word,
    cost_estimation_word,
    deep_merge_dict as _deep_merge_dict,
    default_chat_config,
    default_query_mode_for_model,
    default_response_config,
    extract_summary_from_responses_api_output,
    extract_text_from_chat_completion_body,
    extract_text_from_responses_api_output,
    get_azure_price,
    is_reasoning_model,
    parse_custom_id as _parse_custom_id,
    resolve_tokenizer_model as _resolve_tokenizer_model,
    safe_model_dump as _safe_model_dump,
    supports_chat_reasoning_effort,
)
from model.provider_common import (
    apply_cot_instruction_transform as _apply_cot_instruction_transform,
    get_example_path as _get_example_path,
)

__all__ = [
    "MODEL_PRICE_PER_1M",
    "create_azure_client",
    "azure_query_single",
    "azure_query_batch_from_task",
    "azure_query_batch_from_file",
    "azure_query_batch_from_data",
    "merge_azure_query_batch",
    "parse_azure_result_batch",
    "transform_azure_result_to_task_result",
    "integrate_task_data_batch_result",
    "is_reasoning_model",
    "supports_chat_reasoning_effort",
    "default_query_mode_for_model",
    "default_chat_config",
    "default_response_config",
    "build_chat_completion_batch_body",
    "build_response_batch_body",
    "extract_text_from_responses_api_output",
    "extract_text_from_chat_completion_body",
    "extract_summary_from_responses_api_output",
    "get_azure_price",
    "cost_calculation_token",
    "cost_calculation_word",
    "cost_estimation_word",
]


# =========================== Client Creation ===============================


def create_azure_client(
    azure_endpoint: str,
    api_version: str,
    api_key: str,
) -> AzureOpenAI:
    """Create and return an `AzureOpenAI` client."""
    return AzureOpenAI(
        api_version=api_version, azure_endpoint=azure_endpoint, api_key=api_key
    )


# ========================== Single Query ===================================


def azure_query_single(
    client: AzureOpenAI,
    input: Any,
    examples: Optional[Sequence[Dict[str, str]]] = None,
    model_name: str = "gpt-4",
    mode: Optional[str] = None,
    dict_config: Optional[Dict[str, Any]] = None,
    max_token_output: int = 16384,
):
    """
    Execute a single Azure OpenAI request.

    Args:
        client: AzureOpenAI client.
        input: Either a dict with `instruction` and `input`, a list of messages,
            or a plain text prompt.
        examples: Optional few-shot examples in `{"input": ..., "output": ...}`.
        model_name: Deployment/model name.
        mode: `chat`, `response`, or `None` (auto by model type).
        dict_config: Optional decoding config override.
        max_token_output: Unified output token cap used by internal mapping.
            - response + reasoning model -> `max_output_tokens`
            - otherwise -> `max_completion_tokens`

    Returns:
        Dict with keys:
            - input: request input payload for reproducibility.
            - output: extracted text / summary / raw model_dump.
            - params: final mode/model/config used for request.
    """
    query_mode = mode or default_query_mode_for_model(model_name)
    messages = _build_messages(input_data=input, examples=examples)

    if query_mode == "response":
        # Model configuration for Responses API
        config = default_response_config(model_name, max_output_tokens=max_token_output)
        _deep_merge_dict(config, dict_config)
        config = _apply_output_token_key(
            config=config,
            model_name=model_name,
            mode="response",
            max_token_output=max_token_output,
        )
        model = config.pop("model", model_name)
        # Send request to Azure OpenAI API
        query = {"input": messages, "model": model, **config}
        response = client.responses.create(**query)
        response_dump = _safe_model_dump(response)
        response_text = extract_text_from_responses_api_output(response_dump)
        response_summary = extract_summary_from_responses_api_output(response_dump)
        return {
            "input": {
                "raw_input": input,
                "messages": messages,
                "examples": list(examples or []),
            },
            "output": {
                "text": response_text,
                "summary": response_summary,
            },
            "query": query,
            "response": response_dump,
            "params": {
                "model": model,
                "mode": query_mode,
                "config_decoding": config,
            },
        }

    if query_mode != "chat":
        raise ValueError(f"Unknown mode={query_mode!r}. Use 'chat' or 'response'.")

    # Model configuration for Chat Completions
    config = default_chat_config(model_name, max_completion_tokens=max_token_output)
    _deep_merge_dict(config, dict_config)
    config = _apply_output_token_key(
        config=config,
        model_name=model_name,
        mode="chat",
        max_token_output=max_token_output,
    )
    model = config.pop("model", model_name)
    # Send request to Azure OpenAI API
    query = {"model": model, "messages": messages, **config}
    response = client.chat.completions.create(**query)
    response_dump = _safe_model_dump(response)
    # Extract the assistant text from the response
    response_text = extract_text_from_chat_completion_body(response_dump)
    return {
        "input": {
            "raw_input": input,
            "messages": messages,
            "examples": list(examples or []),
        },
        "output": {
            "text": response_text,
        },
        "query": query,
        "response": response_dump,
        "params": {
            "model": model,
            "mode": query_mode,
            "config_decoding": config,
        },
    }


# ========================== Batch Query ====================================


def azure_query_batch_from_task(
    task_name: str,
    model_name: str,
    prompt_mode: str,
    split: str = "test",
    mode: Optional[str] = None,
    path_dir_raw: str = "dataset_raw",
    path_dir_batch: str = "azure/input",
    max_token_input: int = 100 * 1024,
    max_token_output: int = 16384,
    dict_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create and save Azure batch input records for one task.

    Automatically loads task data and example data from the standard directory
    layout. Runtime decoding parameters should be passed via `dict_config`.

    Args:
        task_name: Name of the task (maps to `{path_dir_raw}/{task_name}.SFT.json`).
        model_name: Deployment/model name.
        prompt_mode: Prompt strategy, e.g. `direct`, `cot`, `direct-3-shot`, `cot-3-shot`.
        split: Data split to use (default `test`).
        mode: `chat`, `response`, or `None` (auto by model type).
        path_dir_raw: Root directory containing raw task JSON files.
        path_dir_batch: Root directory for batch output files.
        max_token_input: Maximum number of input tokens.
        max_token_output: Maximum number of output tokens.
        dict_config: Optional decoding config override.

    Returns:
        List of Azure batch request dicts.
    """
    path_file_task = f"{path_dir_raw}/{task_name}.SFT.json"
    dataset = [d for d in load_json(path_file_task) if d["split"] == split]

    print(f"Task: {task_name}")
    print(f" - Split: {split} ({len(dataset)} samples)")
    print(f" - Model: {model_name}")
    print(f" - Prompt mode: {prompt_mode}")

    prompt_base, num_example = parse_prompt_mode(prompt_mode)

    # Apply CoT instruction transform if requested
    if prompt_base == "cot":
        _apply_cot_instruction_transform(dataset, task_name)
        if dataset:
            print(" - Applied CoT instruction transform")
            print(f" - Sample instruction:\n{dataset[0]['instruction']}")

    # Load few-shot examples if requested
    examples: List[Dict[str, str]] = []
    if num_example is not None:
        path_file_example = _get_example_path(path_dir_raw, task_name, prompt_base)
        examples = load_json(path_file_example)[:num_example]
        print(f" - Few-shot examples: {num_example}")
    else:
        print(" - Few-shot examples: 0")

    query_batch = azure_query_batch_from_data(
        task_name=task_name,
        model_name=model_name,
        prompt_mode=prompt_mode,
        split=split,
        dataset=dataset,
        examples=examples,
        mode=mode,
        max_token_input=max_token_input,
        max_token_output=max_token_output,
        dict_config=dict_config,
    )

    path_dir_azure_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}"
    os.makedirs(path_dir_azure_batch, exist_ok=True)
    path_file_azure_batch = f"{path_dir_azure_batch}/{task_name}.batch.jsonl"
    write_jsonl(data=query_batch, path_file=path_file_azure_batch)
    print(f" - Saved {len(query_batch)} records to {path_file_azure_batch}")
    return query_batch


def azure_query_batch_from_file(
    task_name: str,
    model_name: str,
    prompt_mode: str,
    split: str = "test",
    mode: Optional[str] = None,
    path_file_data: Optional[str] = None,
    path_file_example: Optional[str] = None,
    path_file_result: Optional[str] = None,
    max_token_input: int = 100 * 1024,
    max_token_output: int = 8192,
    dict_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Create Azure batch input records from explicit file paths.

    A convenience wrapper that loads data and examples from user-specified
    files, builds batch payloads, and optionally saves the result.

    Args:
        task_name: Name of the task (used for custom_id prefix).
        model_name: Deployment/model name.
        prompt_mode: Prompt strategy, e.g. `direct`, `cot`, `direct-3-shot`, `cot-3-shot`.
        split: Data split to use (default `test`).
        mode: `chat`, `response`, or `None` (auto by model type).
        path_file_data: Path to the task data JSON file. Each element must
            contain at least `id`, `instruction`, `input`, `language`, `split`.
        path_file_example: Path to the example JSON file (few-shot). `None`
            means no examples are loaded.
        path_file_result: Path to save the batch JSONL output. `None` means
            the result is not saved to disk.
        max_token_input: Maximum number of input tokens.
        max_token_output: Maximum number of output tokens.
        dict_config: Optional decoding config override.

    Returns:
        List of Azure batch request dicts.
    """
    if not path_file_data:
        raise ValueError("`path_file_data` is required.")

    # Load task data, filter by split
    dataset = load_json(path_file_data)
    if dataset and "split" in dataset[0]:
        dataset = [d for d in dataset if d["split"] == split]

    print(f"Task: {task_name}")
    print(f" - Data file: {path_file_data} ({len(dataset)} samples)")
    print(f" - Model: {model_name}")
    print(f" - Prompt mode: {prompt_mode}")

    prompt_base, num_example = parse_prompt_mode(prompt_mode)

    # Apply CoT instruction transform if requested
    if prompt_base == "cot":
        _apply_cot_instruction_transform(dataset, task_name)
        if dataset:
            print(" - Applied CoT instruction transform")

    # Load few-shot examples if a path is provided
    examples: List[Dict[str, str]] = []
    if path_file_example:
        all_examples = load_json(path_file_example)
        limit = num_example if num_example is not None else len(all_examples)
        examples = all_examples[:limit]
        print(f" - Example file: {path_file_example} ({len(examples)} examples)")
    else:
        print(" - Few-shot examples: 0")

    query_batch = azure_query_batch_from_data(
        task_name=task_name,
        model_name=model_name,
        prompt_mode=prompt_mode,
        split=split,
        dataset=dataset,
        examples=examples,
        mode=mode,
        max_token_input=max_token_input,
        max_token_output=max_token_output,
        dict_config=dict_config,
    )

    # Save batch JSONL if an output path is given
    if path_file_result:
        os.makedirs(os.path.dirname(path_file_result) or ".", exist_ok=True)
        write_jsonl(data=query_batch, path_file=path_file_result)
        print(f" - Saved {len(query_batch)} records to {path_file_result}")

    return query_batch


def azure_query_batch_from_data(
    task_name: str,
    model_name: str,
    prompt_mode: str,
    dataset: Sequence[Dict[str, Any]],
    split: str = "test",
    examples: Optional[Sequence[Dict[str, str]]] = None,
    mode: Optional[str] = None,
    max_token_input: int = 100 * 1024,
    max_token_output: int = 8192,
    num_token_reserve: int = 5,
    dict_config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Prepare Azure batch request payloads.

    Supports both Chat Completions (`/chat/completions`) and Responses
    (`/responses`) API modes, determined by the `mode` parameter or
    auto-detected from `model_name`.

    Args:
        task_name: Name of the task (used for custom_id prefix).
        model_name: Deployment/model name.
        prompt_mode: Prompt strategy.
        dataset: Sequence of data dicts, each with at least `id`,
            `instruction`, `input`, `language`.
        split: Data split name.
        examples: Optional few-shot examples.
        mode: `chat`, `response`, or `None` (auto by model type).
        max_token_input: Maximum number of input tokens.
        max_token_output: Maximum number of output tokens.
        num_token_reserve: Tokens to reserve for response generation.
        dict_config: Optional decoding config override.

    Returns:
        List of Azure batch request dicts.
    """
    if not dataset:
        return []

    parse_prompt_mode(prompt_mode)
    query_mode = mode or default_query_mode_for_model(model_name)
    batch_url = "/v1/responses" if query_mode == "response" else "/v1/chat/completions"
    body_builder = (
        build_response_batch_body
        if query_mode == "response"
        else build_chat_completion_batch_body
    )

    tokenizer_name = _resolve_tokenizer_model(model_name)
    tokenizer = tiktoken.encoding_for_model(tokenizer_name)
    print(f" - Loading tokenizer: {tokenizer_name}")
    print(f" - API mode: {query_mode} ({batch_url})")

    if "gpt-35" in model_name.lower():
        max_token_input_valid = 16 * 1024 - max_token_output - num_token_reserve
        max_token_input = min(max_token_input, max_token_input_valid)

    print(f" - Max input tokens: {max_token_input}")
    print(f" - Max output tokens: {max_token_output}")

    language = dataset[0]["language"]
    exp_name = f"{task_name}|{model_name}|{prompt_mode}|{split}"

    query_batch: List[Dict[str, Any]] = []
    for dict_data in dataset:
        if dict_data["split"] != split:
            continue
        _, list_message = get_formatted_chat(
            tokenizer=tokenizer,
            language=language,
            input_system=dict_data["instruction"],
            input_user=dict_data["input"],
            max_token_input=max_token_input,
            times_word_token=4,
            num_token_reserve=num_token_reserve,
            examples=list(examples or []),
            flag_openai=True,
        )
        query_batch.append(
            {
                "custom_id": f"{exp_name}|{dict_data['id']}",
                "method": "POST",
                "url": batch_url,
                "body": body_builder(
                    model_name=model_name,
                    messages=list_message,
                    max_token_output=max_token_output,
                    dict_config=dict_config,
                ),
            }
        )

    return query_batch


# ========================== Batch File Operations ==========================


def merge_azure_query_batch(
    model_name: str,
    prompt_mode: str,
    path_dir_raw: str = "azure/input",
    path_dir_merged: str = "azure/input/merged",
    max_lines: int = 50000,
    max_size_mb: int = 180,
) -> None:
    """Merge multiple task-level batch files into larger chunk files."""
    path_dir_raw = f"{path_dir_raw}/{model_name}/{prompt_mode}"
    list_path_file = sorted(
        os.path.join(path_dir_raw, file)
        for file in os.listdir(path_dir_raw)
        if file.endswith(".jsonl")
    )
    os.makedirs(path_dir_merged, exist_ok=True)

    max_size_bytes = max_size_mb * 1024 * 1024
    chunk_index = 0
    current_lines: List[str] = []
    current_size = 0

    for task_file in list_path_file:
        with open(task_file, "r", encoding="utf-8") as infile:
            file_lines = [line.rstrip("\n") for line in infile]
        file_size = sum(len(line.encode("utf-8")) for line in file_lines)

        overflow = (
            len(current_lines) + len(file_lines) > max_lines
            or current_size + file_size > max_size_bytes
        )
        if overflow and current_lines:
            output_file = os.path.join(
                path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
            )
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(current_lines))
            print(
                f"Created: {output_file} with {len(current_lines)} lines "
                f"({current_size / (1024 * 1024):.2f} MB)"
            )
            chunk_index += 1
            current_lines = []
            current_size = 0

        current_lines.extend(file_lines)
        current_size += file_size

    if current_lines:
        output_file = os.path.join(
            path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
        )
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(current_lines))
        print(
            f"Created: {output_file} with {len(current_lines)} lines "
            f"({current_size / (1024 * 1024):.2f} MB)"
        )


# ========================== Result Parsing =================================


def parse_azure_result_batch(path_file_result_batch: str) -> List[Dict[str, Any]]:
    """Parse an Azure batch output JSONL file into normalized records.

    Automatically detects whether each record came from the Chat Completions
    API or the Responses API and extracts text accordingly.
    """
    list_dict_result = load_jsonl(path_file_result_batch)

    list_dict_id_result: List[Dict[str, Any]] = []
    for idx_result, dict_result in enumerate(list_dict_result):
        custom_id = dict_result.get("custom_id", "")
        error = dict_result.get("error")
        body = dict_result.get("response", {}).get("body", {})
        query_mode = body.get("object", "")
        if query_mode == "chat.completion":
            query_mode = "chat"

        text = ""
        summary = ""
        if isinstance(body, dict):
            if query_mode == "chat":
                text = extract_text_from_chat_completion_body(body)
            elif query_mode == "response":
                text = extract_text_from_responses_api_output(body)
                summary = extract_summary_from_responses_api_output(body)
            else:
                raise ValueError(f"Unknown query mode: {query_mode}")

        if not custom_id:
            print(f"Warning: missing custom_id at result index {idx_result}")

        dict_result = {"id": custom_id, "output": text, "error": error}
        if query_mode == "response" and len(summary) > 0:
            dict_result["summary"] = summary
        list_dict_id_result.append(dict_result)

    return list_dict_id_result


def transform_azure_result_to_task_result(
    path_dir_azure_result: str = "azure/output",
    path_dir_task_save: str = "result",
    path_dir_raw: str = "dataset_raw",
    path_dir_batch: str = "azure/input",
    decoding_strategy: str = "greedy",
    seed: int = 42,
) -> None:
    """Convert Azure batch outputs into task-level result JSON files."""
    list_path_file = sorted(
        os.path.join(path_dir_azure_result, path_file)
        for path_file in os.listdir(path_dir_azure_result)
        if path_file.endswith(".jsonl")
    )

    for path_file_result in list_path_file:
        list_dict_id_result = parse_azure_result_batch(path_file_result)
        dict_task_data_all: Dict[str, Dict[str, Any]] = {}

        for dict_result in list_dict_id_result:
            parsed = _parse_custom_id(dict_result["id"])
            if not parsed:
                print(f"Warning: invalid custom_id format: {dict_result['id']}")
                continue
            task, model_name, prompt_mode, split, _sample_id = parsed

            if task not in dict_task_data_all:
                dict_task_data_all[task] = {
                    "model_name": model_name,
                    "prompt_mode": prompt_mode,
                    "split": split,
                    "list_dict_result": [],
                }
            dict_task_data_all[task]["list_dict_result"].append(dict_result)

        for task, dict_task_data in dict_task_data_all.items():
            model_name = dict_task_data["model_name"]
            prompt_mode = dict_task_data["prompt_mode"]
            split = dict_task_data["split"]
            print(f"Task: {task}")
            print(f" - Model name: {model_name}")
            print(f" - Prompt mode: {prompt_mode}")
            print(f" - Split: {split}")
            print("----------------------------------------")

            dataset = integrate_task_data_batch_result(
                task=task,
                model_name=model_name,
                prompt_mode=prompt_mode,
                split=split,
                dict_task_data=dict_task_data,
                path_dir_raw=path_dir_raw,
                path_dir_batch=path_dir_batch,
            )

            model_name_output = model_name.replace("-batch", "")
            path_file_save = os.path.join(
                path_dir_task_save,
                f"{task}/{model_name_output}/{task}-{prompt_mode}-{decoding_strategy}-{seed}.result.json",
            )
            os.makedirs(os.path.dirname(path_file_save), exist_ok=True)
            write_json(data=dataset, path_file=path_file_save)
            print(f" - Saved: {path_file_save}")
            print("========================================")


def integrate_task_data_batch_result(
    task: str,
    model_name: str,
    prompt_mode: str,
    split: str,
    dict_task_data: Dict[str, Any],
    path_dir_raw: str = "dataset_raw",
    path_dir_batch: str = "azure/input",
) -> List[Dict[str, Any]]:
    """Join raw task data, batch requests, and parsed batch outputs."""
    path_file_raw = f"{path_dir_raw}/{task}.SFT.json"
    dataset = [d for d in load_json(path_file_raw) if d["split"] == split]
    print(f" - Num of data: {len(dataset)}")

    path_file_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}/{task}.batch.jsonl"
    list_dict_batch = load_jsonl(path_file_batch)
    print(f" - Num of batch: {len(list_dict_batch)}")

    if len(dataset) != len(list_dict_batch):
        raise ValueError(
            f"Data and batch size mismatch for task={task}: "
            f"{len(dataset)} vs {len(list_dict_batch)}"
        )

    for dict_data, dict_batch in zip(dataset, list_dict_batch):
        body = dict_batch.get("body", {})
        dict_data["input"] = body.get("messages", body.get("input", []))

    dict_id_result: Dict[str, Dict[str, Any]] = {}
    for dict_result in dict_task_data["list_dict_result"]:
        parsed = _parse_custom_id(dict_result["id"])
        if not parsed:
            continue
        sample_id = parsed[-1]
        dict_id_result[sample_id] = dict_result

    print(f" - Num of result: {len(dict_id_result)}")
    count_match = 0
    for dict_data in dataset:
        sample_id = str(dict_data["id"])
        if sample_id in dict_id_result:
            result = dict_id_result[sample_id]
            dict_data["pred"] = result["output"]
            if "summary" in result:
                dict_data["summary"] = result["summary"]
            dict_data["error"] = result["error"]
            count_match += 1
        else:
            print(f" - Missing: {sample_id}")
            dict_data["pred"] = ""
            dict_data["error"] = ""

    print(f" - Matched: {count_match}")
    if count_match == len(dataset):
        print(" - All matched.")
    else:
        print(f" - Lost {len(dataset) - count_match} samples.")
    return dataset
