import pytest

pytest.importorskip("tiktoken")
pytest.importorskip("openai")

from model.azure import (
    build_chat_completion_batch_body,
    build_response_batch_body,
    default_query_mode_for_model,
    extract_text_from_chat_completion_body,
    extract_text_from_responses_api_output,
    get_azure_price,
    is_reasoning_model,
)
from model.azure_utils import build_messages, parse_custom_id
from model.provider_common import apply_cot_instruction_transform, get_example_path


def test_provider_common_builds_example_path():
    assert get_example_path("dataset_raw", "task-a", "direct") == (
        "dataset_raw/example/task-a.example.json"
    )
    assert get_example_path("dataset_raw", "task-a", "cot") == (
        "dataset_raw/example_cot/task-a.example.json"
    )
    with pytest.raises(ValueError):
        get_example_path("dataset_raw", "task-a", "directly")


def test_provider_common_applies_cot_transform():
    dataset = [
        {
            "id": "sample-1",
            "instruction": (
                "Return your answer in the following format. DO NOT GIVE ANY EXPLANATION:"
            ),
        }
    ]

    apply_cot_instruction_transform(dataset, "task-a")

    assert "Analysis:" in dataset[0]["instruction"]
    assert "Result:" in dataset[0]["instruction"]


def test_azure_message_order_with_examples():
    messages = build_messages(
        input_data={"instruction": "Instruction", "input": "Current input"},
        examples=[{"input": "Example input", "output": "Example output"}],
    )

    assert messages == [
        {"role": "system", "content": "Instruction"},
        {"role": "user", "content": "Example input"},
        {"role": "assistant", "content": "Example output"},
        {"role": "user", "content": "Current input"},
    ]


def test_azure_batch_body_token_keys():
    chat_body = build_chat_completion_batch_body(
        model_name="gpt-4o",
        messages=[{"role": "user", "content": "hello"}],
        max_token_output=7,
    )
    response_body = build_response_batch_body(
        model_name="gpt-5",
        messages=[{"role": "user", "content": "hello"}],
        max_token_output=11,
    )

    assert chat_body["max_completion_tokens"] == 7
    assert "max_output_tokens" not in chat_body
    assert response_body["max_output_tokens"] == 11
    assert "max_completion_tokens" not in response_body


def test_azure_result_text_extractors():
    chat_body = {"choices": [{"message": {"content": "chat text"}}]}
    response_body = {
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "response text"}],
            }
        ]
    }

    assert extract_text_from_chat_completion_body(chat_body) == "chat text"
    assert extract_text_from_responses_api_output(response_body) == "response text"


def test_azure_model_detection_and_cost_helpers():
    assert is_reasoning_model("gpt-5")
    assert default_query_mode_for_model("gpt-5") == "response"
    assert default_query_mode_for_model("gpt-4o") == "chat"
    assert get_azure_price("gpt-4o") == (2.5, 10.0)
    assert get_azure_price("gpt-4o-batch", flag_batch=True) == (1.25, 5.0)


def test_parse_custom_id():
    assert parse_custom_id("task|model|direct-5-shot|test|sample-1") == (
        "task",
        "model",
        "direct-5-shot",
        "test",
        "sample-1",
    )
    assert parse_custom_id("bad") is None
