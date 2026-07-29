import pytest

pytest.importorskip("tiktoken")
pytest.importorskip("openai")

from model.api.azure_common import (
    build_chat_completion_batch_body,
    build_response_batch_body,
    extract_text_from_chat_completion_body,
    extract_text_from_responses_api_output,
    get_azure_price,
    is_reasoning_model,
)
from model.api.config import effective_run_config, require_model_section
from model.api.prompts import custom_id
from model.api.results import QueryResult
from model.api_runner import (
    result_rows_from_query_results,
)


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
    assert get_azure_price("gpt-4o") == (2.5, 10.0)
    assert get_azure_price("gpt-4o-batch", flag_batch=True) == (1.25, 5.0)


def test_custom_id():
    assert custom_id(
        "task", "model", "direct-5-shot", "test", "sample-1"
    ) == (
        "task|model|direct-5-shot|test|sample-1"
    )


def test_api_model_registry_fails_fast_for_missing_model():
    with pytest.raises(ValueError, match="not registered"):
        require_model_section(
            api_model_cfg={"Azure": {"gpt-4o": {"deploy_name": "gpt-4o"}}},
            provider="azure",
            model_name="missing-model",
        )


def test_effective_run_config_applies_greedy_seed_to_azure_chat():
    cfg = effective_run_config(
        run_cfg={"chat_params": {"temperature": 0.7}, "max_token_output": 8},
        provider="azure",
        decoding_strategy="greedy",
        seed=123,
    )

    assert cfg["chat_params"]["temperature"] == 0
    assert cfg["chat_params"]["top_p"] == 1
    assert cfg["chat_params"]["seed"] == 123


def test_effective_run_config_requires_sampling_override():
    with pytest.raises(ValueError, match="sampling_chat_params"):
        effective_run_config(
            run_cfg={"chat_params": {}},
            provider="azure",
            decoding_strategy="sampling",
            seed=42,
        )


def test_api_result_rows_can_omit_saved_input():
    rows = result_rows_from_query_results(
        dataset=[
            {
                "task": "Task-A",
                "id": "1",
                "type": "ext",
                "language": "en",
                "split": "test",
                "instruction": "private instruction",
                "input": "private note",
                "output": "gold",
            }
        ],
        messages_by_id={"1": [{"role": "user", "content": "private note"}]},
        examples_by_id={"1": ["example-1"]},
        results=[QueryResult(custom_id="Task-A|model|direct|test|1", text="pred")],
        save_input=False,
    )

    assert rows == [
        {
            "task": "Task-A",
            "id": "1",
            "type": "ext",
            "language": "en",
            "split": "test",
            "output": "gold",
            "example": ["example-1"],
            "pred": "pred",
            "error": "",
        }
    ]
