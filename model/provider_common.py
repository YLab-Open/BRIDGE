"""Shared helpers for provider batch-format modules.

The Azure, GCP/Gemini, and DeepSeek batch helpers all need the same prompt-mode
validation, Direct-to-CoT instruction conversion, and example-file lookup. This
module keeps that behavior consistent across providers.
"""

from __future__ import annotations

from typing import Any, MutableSequence

from dataset.config import transform_instruction_to_cot


def apply_cot_instruction_transform(
    dataset: MutableSequence[dict[str, Any]],
    task_name: str,
) -> None:
    """
    Convert Direct task instructions to CoT instructions in place.

    Args:
        dataset: Mutable sequence of task records. Each record must include an
            `instruction` field and should include `id` for error reporting.
        task_name: Task name used only for clear error messages.

    Raises:
        KeyError: If a record is missing `instruction`.
        ValueError: If the instruction does not contain a known Direct template.
    """
    for dict_data in dataset:
        instruction_cot = transform_instruction_to_cot(
            dict_data["instruction"], flag_print=False
        )
        if not instruction_cot:
            sample_id = dict_data.get("id", dict_data.get("idx", "unknown"))
            raise ValueError(
                "No Direct instruction template was found for CoT conversion "
                f"in task={task_name}, sample_id={sample_id}."
            )
        dict_data["instruction"] = instruction_cot


def get_example_path(path_dir_raw: str, task_name: str, prompt_base: str) -> str:
    """
    Return the few-shot example path for one prompt base.

    Args:
        path_dir_raw: Dataset root containing `example/` and `example_cot/`.
        task_name: Task file stem.
        prompt_base: Parsed prompt base, either `direct` or `cot`.

    Returns:
        Path string for `<task_name>.example.json`.

    Raises:
        ValueError: If `prompt_base` is not `direct` or `cot`.
    """
    if prompt_base not in {"direct", "cot"}:
        raise ValueError("prompt_base must be `direct` or `cot`.")
    example_dir = "example_cot" if prompt_base == "cot" else "example"
    return f"{path_dir_raw}/{example_dir}/{task_name}.example.json"
