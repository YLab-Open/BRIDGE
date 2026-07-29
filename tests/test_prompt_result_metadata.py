from types import SimpleNamespace

from dataset.dataset import GeneralDataset, GeneralTask
from dataset.process import format_chat
from model.api.prompts import build_messages_for_dataset
from model.api.results import QueryResult
from model.api_runner import result_rows_from_query_results
from model.inference import _unpack_prompt_batch


class CharacterTokenizer:
    """Small tokenizer with deterministic character-level token counts."""

    def encode(self, text, **_kwargs):
        return list(str(text))

    def decode(self, tokens, **_kwargs):
        return "".join(tokens)

    def apply_chat_template(
        self,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        return_dict=False,
        **_kwargs,
    ):
        del add_generation_prompt, return_dict
        text = "".join(str(message.get("content", "")) for message in messages)
        return list(text) if tokenize else text


class CountingCharacterTokenizer(CharacterTokenizer):
    def __init__(self):
        self.tokenized_chat_texts = []

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        text = "".join(str(message.get("content", "")) for message in messages)
        if tokenize:
            self.tokenized_chat_texts.append(text)
        return super().apply_chat_template(messages, tokenize=tokenize, **kwargs)


def _source_row():
    return {
        "task": "Task-A",
        "id": "sample-1",
        "type": "clf",
        "language": "en",
        "split": "test",
        "instruction": "S",
        "input": "original input",
        "output": "Yes",
    }


def _examples():
    return [
        {"id": f"example-{index}", "input": "abc", "output": "def"}
        for index in range(1, 6)
    ]


def test_local_prompt_metadata_saves_exact_input_and_actual_example_ids():
    row = _source_row()
    original_input = row["input"]
    args = SimpleNamespace(
        model_name="test-model",
        chat_kwargs={},
        max_token_input=44,
        max_token_output=8,
    )
    prompt_dataset = GeneralDataset(
        args=args,
        data=[row],
        tokenizer=CharacterTokenizer(),
        example=_examples(),
        return_metadata=True,
    )

    record = prompt_dataset[0]

    assert not hasattr(prompt_dataset, "dict_idx_cache")
    assert row["input"] == original_input
    assert "example" not in row
    assert record["input"] == original_input
    assert record["example"] == ["example-1", "example-2", "example-3"]

    dataloader = SimpleNamespace(dataset=prompt_dataset)
    assert _unpack_prompt_batch(dataloader, [record]) == [record["prompt"]]
    assert row["input"] == original_input
    assert row["input"] == record["input"]
    assert row["example"] == ["example-1", "example-2", "example-3"]


def test_format_chat_metadata_does_not_mutate_source_row():
    row = _source_row()

    prompt, metadata = format_chat(
        model_name="test-model",
        tokenizer=CharacterTokenizer(),
        data=row,
        max_token_input=44,
        examples=_examples(),
        return_metadata=True,
    )

    assert isinstance(prompt, str)
    assert metadata["input"] == "original input"
    assert metadata["example"] == ["example-1", "example-2", "example-3"]
    assert row == _source_row()


def test_general_dataset_direct_access_keeps_string_interface():
    row = _source_row()
    args = SimpleNamespace(
        model_name="test-model",
        chat_kwargs={},
        max_token_input=44,
        max_token_output=8,
    )

    item = GeneralDataset(
        args=args,
        data=[row],
        tokenizer=CharacterTokenizer(),
        example=_examples(),
    )[0]

    assert isinstance(item, str)
    assert row == _source_row()


def test_local_truncated_user_input_is_replaced_by_exact_prompt():
    row = _source_row()
    row["input"] = "x" * 100
    raw_input = row["input"]
    args = SimpleNamespace(
        model_name="test-model",
        chat_kwargs={},
        max_token_input=44,
        max_token_output=8,
    )
    prompt_dataset = GeneralDataset(
        args=args,
        data=[row],
        tokenizer=CharacterTokenizer(),
        return_metadata=True,
    )

    record = prompt_dataset[0]
    _unpack_prompt_batch(SimpleNamespace(dataset=prompt_dataset), [record])

    assert row["input"] == record["input"]
    assert row["input"] not in row["instruction"]
    assert row["input"] != raw_input
    assert raw_input not in row["input"]
    assert row["instruction"] == "S"
    assert row["example"] == []


def test_api_result_saves_exact_messages_as_input_and_examples(monkeypatch):
    from model.api import prompts

    monkeypatch.setattr(
        prompts,
        "tokenizer_for_model",
        lambda _model_name: CharacterTokenizer(),
    )
    row = _source_row()
    messages_by_id, examples_by_id = build_messages_for_dataset(
        dataset=[row],
        examples=_examples(),
        model_name="test-model",
        max_token_input=124,
        return_example_ids=True,
    )
    rows = result_rows_from_query_results(
        dataset=[row],
        messages_by_id=messages_by_id,
        examples_by_id=examples_by_id,
        results=[
            QueryResult(
                custom_id="Task-A|test-model|direct-5-shot|test|sample-1",
                text="No",
            )
        ],
        save_input=True,
    )

    assert rows[0]["input"] == messages_by_id["sample-1"]
    assert "prompt" not in rows[0]
    assert rows[0]["example"] == ["example-1", "example-2", "example-3"]


def test_compact_api_result_still_keeps_actual_example_ids():
    rows = result_rows_from_query_results(
        dataset=[_source_row()],
        messages_by_id={"sample-1": [{"role": "user", "content": "prompt"}]},
        examples_by_id={"sample-1": ["example-1", "example-2"]},
        results=[
            QueryResult(
                custom_id="Task-A|test-model|direct-5-shot|test|sample-1",
                text="No",
            )
        ],
        save_input=False,
    )

    assert "input" not in rows[0]
    assert "prompt" not in rows[0]
    assert rows[0]["example"] == ["example-1", "example-2"]


def test_instruction_transform_runs_once_per_unique_instruction(monkeypatch):
    from dataset import dataset as dataset_module

    task = GeneralTask.__new__(GeneralTask)
    task.dataset_train = [
        {"instruction": "same"},
        {"instruction": "same"},
    ]
    task.dataset_dev = [{"instruction": "different"}]
    task.dataset_test = []
    calls = []

    def fake_transform(instruction, flag_print=False):
        calls.append((instruction, flag_print))
        return f"{instruction}-direct"

    monkeypatch.setattr(
        dataset_module,
        "transform_instruction_to_direct",
        fake_transform,
    )

    task.setup_direct()

    assert calls == [("same", False), ("different", False)]
    assert [row["instruction"] for row in task.dataset_train] == [
        "same-direct",
        "same-direct",
    ]
    assert task.dataset_dev[0]["instruction"] == "different-direct"


def test_direct_cot_direct_setup_is_isolated_between_experiments():
    from dataset.config import get_template_direct

    instruction = f"Clinical task. {get_template_direct()[0]}\nAssessment:"
    source = {
        **_source_row(),
        "instruction": instruction,
    }
    task = GeneralTask.__new__(GeneralTask)
    task.args = SimpleNamespace(dir_dataset=[])
    task.name = "Task-A"
    task.dataset_train_base = [source]
    task.dataset_dev_base = []
    task.dataset_test_base = []

    task.setup(tokenizer=None, prompt_mode="direct")
    first_direct = task.dataset_train[0]["instruction"]
    task.setup(tokenizer=None, prompt_mode="cot")
    cot = task.dataset_train[0]["instruction"]
    task.setup(tokenizer=None, prompt_mode="direct")
    second_direct = task.dataset_train[0]["instruction"]

    assert first_direct == instruction
    assert "Analysis:" in cot
    assert second_direct == first_direct
    assert task.dataset_train_base[0]["instruction"] == instruction
    assert task.dataset_train[0] is not task.dataset_train_base[0]


def test_few_shot_token_counts_are_cached_across_dataset_rows():
    tokenizer = CountingCharacterTokenizer()
    args = SimpleNamespace(
        model_name="test-model",
        chat_kwargs={},
        max_token_input=44,
    )
    rows = [_source_row(), {**_source_row(), "id": "sample-2"}]
    prompt_dataset = GeneralDataset(
        args=args,
        data=rows,
        tokenizer=tokenizer,
        example=_examples(),
        return_metadata=True,
    )

    assert prompt_dataset[0]["example"] == [
        "example-1",
        "example-2",
        "example-3",
    ]
    assert prompt_dataset[1]["example"] == [
        "example-1",
        "example-2",
        "example-3",
    ]
    # The fourth example is checked and rejected; all four counts are reused
    # for the second patient instead of being tokenized again.
    assert tokenizer.tokenized_chat_texts.count("abcdef") == 4
