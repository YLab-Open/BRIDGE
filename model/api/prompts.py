from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from dataset.process import get_formatted_chat
from model.api.azure_common import resolve_tokenizer_model


def custom_id(
    task_name: str,
    model_name: str,
    prompt_mode: str,
    split: str,
    sample_id: Any,
) -> str:
    """Build a stable provider custom id."""
    return f"{task_name}|{model_name}|{prompt_mode}|{split}|{sample_id}"


def load_task_dataset(
    task_name: str,
    prompt_mode: str,
    split: str,
    dir_dataset: list[str],
    max_data_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load and prepare one task split with the same prompt transforms as local inference."""
    from dataset.dataset import GeneralTask

    args = SimpleNamespace(dir_dataset=dir_dataset)
    if max_data_size:
        args.max_data_size = int(max_data_size)
    task = GeneralTask(args=args, task_name=task_name)
    task.setup(tokenizer=None, prompt_mode=prompt_mode)
    if split == "test":
        dataset = task.dataset_test
    elif split == "dev":
        dataset = task.dataset_dev
    elif split == "train":
        dataset = task.dataset_train
    elif split == "all":
        dataset = task.dataset_train + task.dataset_dev + task.dataset_test
    else:
        raise ValueError(f"Unsupported split/mode: {split}")
    return [row.copy() for row in dataset], [row.copy() for row in task.examples]


def tokenizer_for_model(model_name: str):
    """Return a tiktoken tokenizer, falling back to gpt-4o."""
    try:
        import tiktoken
    except ImportError as exc:
        raise ImportError(
            "tiktoken is required for commercial API prompt token budgeting."
        ) from exc

    try:
        return tiktoken.encoding_for_model(resolve_tokenizer_model(model_name))
    except Exception:
        return tiktoken.encoding_for_model("gpt-4o")


def build_messages_for_dataset(
    dataset: list[dict[str, Any]],
    examples: list[dict[str, Any]],
    model_name: str,
    max_token_input: int,
    return_example_ids: bool = False,
) -> (
    dict[str, list[dict[str, str]]]
    | tuple[
        dict[str, list[dict[str, str]]],
        dict[str, list[Any]],
    ]
):
    """Build OpenAI-style messages for every row in a dataset."""
    if not dataset:
        return ({}, {}) if return_example_ids else {}
    tokenizer = tokenizer_for_model(model_name)
    messages_by_id: dict[str, list[dict[str, str]]] = {}
    examples_by_id: dict[str, list[Any]] = {}
    example_token_cache: dict[tuple[Any, ...], int] = {}
    for row in dataset:
        example_ids, messages = get_formatted_chat(
            tokenizer=tokenizer,
            language=row["language"],
            input_system=row["instruction"],
            input_user=row["input"],
            max_token_input=max_token_input,
            times_word_token=4,
            num_token_reserve=5,
            examples=examples,
            flag_openai=True,
            example_token_cache=example_token_cache,
        )
        sample_id = str(row["id"])
        messages_by_id[sample_id] = messages
        examples_by_id[sample_id] = list(example_ids)
    if return_example_ids:
        return messages_by_id, examples_by_id
    return messages_by_id


def split_system_messages(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Split OpenAI-style messages into provider system text and dialog."""
    systems = [
        str(message.get("content", ""))
        for message in messages
        if message.get("role") == "system"
    ]
    dialog = [
        {"role": message.get("role", "user"), "content": str(message.get("content", ""))}
        for message in messages
        if message.get("role") != "system"
    ]
    return "\n\n".join(systems), dialog


def gemini_contents_from_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    """Convert chat messages to a Gemini system instruction and content string."""
    system, dialog = split_system_messages(messages)
    chunks = []
    for message in dialog:
        role = message.get("role", "user").upper()
        chunks.append(f"{role}:\n{message.get('content', '')}")
    return system, "\n\n".join(chunks)
