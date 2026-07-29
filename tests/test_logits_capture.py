import gzip
import json
from types import SimpleNamespace

import pytest
import yaml

from model.inference import (
    _append_partial_vllm_state,
    _load_partial_vllm_state,
    partial_result_path,
    save_result,
)
from model.logits import (
    JsonlSidecarWriter,
    _label_token_ids_for_prefix,
    collect_vllm_text_outputs,
    configure_logits_from_args,
    get_generation_trace_top_k,
    infer_label_target_from_generated_tokens,
    logit_output_paths,
    run_vllm_label_probability_probes,
)


class FakeTokenizer:
    id_to_text = {
        1: "Assessment:",
        2: " ",
        3: "Yes",
        4: " Yes",
        5: "No",
        6: " No",
        7: "1",
    }
    text_to_ids = {
        "Yes": [3],
        " Yes": [4],
        "No": [5],
        " No": [6],
        "1": [7],
        " 1": [2, 7],
    }

    def decode(self, token_ids, **_kwargs):
        return "".join(self.id_to_text[int(token_id)] for token_id in token_ids)

    def encode(self, text, add_special_tokens=False):
        if text in self.text_to_ids:
            return self.text_to_ids[text]
        return [ord(ch) for ch in text]

    def __len__(self):
        return 10


def test_generated_token_ids_determine_label_boundary():
    tokenizer = FakeTokenizer()

    merged_space = infer_label_target_from_generated_tokens(
        text="Assessment: Yes",
        token_ids=[1, 4],
        tokenizer=tokenizer,
        label_values=["No", "Yes"],
        answer_prefix="Assessment:",
    )
    assert merged_space["prefix_before_target_token"] == "Assessment:"
    assert merged_space["target_token_text"] == " Yes"
    assert merged_space["target_token_prefix"] == " "

    separate_space = infer_label_target_from_generated_tokens(
        text="Assessment: Yes",
        token_ids=[1, 2, 3],
        tokenizer=tokenizer,
        label_values=["No", "Yes"],
        answer_prefix="Assessment:",
    )
    assert separate_space["prefix_before_target_token"] == "Assessment: "
    assert separate_space["target_token_text"] == "Yes"
    assert separate_space["target_token_prefix"] == ""


def test_generated_label_target_requires_answer_prefix():
    tokenizer = FakeTokenizer()

    with pytest.raises(ValueError, match="answer_prefix"):
        infer_label_target_from_generated_tokens(
            text='Reasoning says "Yes" but no final answer.',
            token_ids=[3],
            tokenizer=tokenizer,
            label_values=["No", "Yes"],
            answer_prefix="Assessment:",
        )


def test_configure_logits_from_separate_yaml(tmp_path):
    path = tmp_path / "logits.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "enabled": True,
                "generation_trace": {"top_k_logprobs": 5},
                "tasks": {
                    "Task-A": {
                        "answer_prefix": "Assessment:",
                        "label_values": ["No", "Yes"],
                        "label_groups": {"negative": ["No"], "positive": ["Yes"]},
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(logits_config=str(path))

    cfg = configure_logits_from_args(args)

    assert args.logits_enabled is True
    assert args.vllm_max_logprobs == -1
    assert cfg["generation_trace"]["top_k_logprobs"] == 5
    assert cfg["generation_trace"]["top_k_logprobs_for_all_output_tokens"] == 5
    assert get_generation_trace_top_k(args) == 5
    assert cfg["tasks"]["Task-A"]["label_values"] == ["No", "Yes"]


def test_generation_logprobs_mode_requests_label_top_k():
    args = SimpleNamespace(
        logits_enabled=True,
        logits={
            "generation_trace": {
                "enabled": True,
                "top_k_logprobs_for_all_output_tokens": 0,
            },
            "label_probability": {
                "enabled": True,
                "mode": "generation_logprobs",
                "top_k_logits": 1000,
            },
        },
    )

    assert get_generation_trace_top_k(args) == 1000

    args.logits["label_probability"]["generation_logprobs_top_k"] = 5000
    assert get_generation_trace_top_k(args) == 5000


def test_label_token_id_overrides_are_merged_for_model_prefix():
    tokenizer = FakeTokenizer()
    target_cfg = {
        "label_values": ["No", "Yes"],
        "model_label_token_id_overrides": {
            "Mistral-Small-3.2-24B-Instruct-2506": {
                "prefixes": {
                    " ": {
                        False: [3501],
                        True: [13830],
                    }
                }
            }
        },
    }
    lp_cfg = {"label_token_prefixes": "auto", "scan_decoded_vocab": True}

    label_token_ids = _label_token_ids_for_prefix(
        tokenizer=tokenizer,
        target_cfg=target_cfg,
        label_probability_cfg=lp_cfg,
        target_prefix=" ",
        model_name="Mistral-Small-3.2-24B-Instruct-2506",
    )

    assert label_token_ids["No"] == [6, 3501]
    assert label_token_ids["Yes"] == [4, 13830]


def test_logit_output_paths_do_not_overlap_result_json(tmp_path):
    result_path = tmp_path / "Task-direct-greedy-42.result.json"
    paths = logit_output_paths(result_path)

    assert paths["logit_json"].name == "Task-direct-greedy-42.result.logit.json"
    assert paths["top_logits"].name == "Task-direct-greedy-42.result.logit.top_logits.jsonl.gz"
    assert paths["metadata"].name == "Task-direct-greedy-42.result.metadata.json"
    assert paths["logit_json"] != result_path


def test_generation_logprobs_scores_actual_generated_token():
    tokenizer = FakeTokenizer()
    args = SimpleNamespace(
        seed=42,
        decoding_strategy="greedy",
        max_token_output=16,
        vllm_logprobs_mode="raw_logits",
        logits_enabled=True,
        logits={
            "generation_trace": {
                "enabled": True,
                "top_k_logprobs_for_all_output_tokens": 0,
            },
            "label_probability": {
                "enabled": True,
                "mode": "generation_logprobs",
                "top_k_logits": 2,
                "save_top_logits": True,
                "label_token_prefixes": "auto",
                "scan_decoded_vocab": True,
                "fail_on_error": False,
            },
        },
    )
    target_cfg = {
        "answer_prefix": "Assessment:",
        "label_values": ["No", "Yes"],
        "label_groups": {"negative": ["No"], "positive": ["Yes"]},
    }
    logprobs = [
        {1: SimpleNamespace(logprob=10.0, rank=1)},
        {
            4: SimpleNamespace(logprob=5.0, rank=1),
            6: SimpleNamespace(logprob=3.0, rank=2),
        },
    ]
    outputs = [
        SimpleNamespace(
            outputs=[
                SimpleNamespace(
                    text="Assessment: Yes",
                    token_ids=[1, 4],
                    logprobs=logprobs,
                )
            ]
        )
    ]

    blocks = run_vllm_label_probability_probes(
        args=args,
        model=None,
        tokenizer=tokenizer,
        request_outputs=outputs,
        prompt_texts=["prompt"],
        target_cfg=target_cfg,
    )

    block = blocks[0]
    assert block["mode"] == "generation_logprobs"
    assert block["pred_label"] == "Yes"
    assert block["pred_target_token_id"] == 4
    assert block["generation_consistency"]["exact_token_match"] is True
    assert block["generation_consistency"]["top_logit_token_match"]["exact_token_match"] is True
    assert block["groups"]["positive"]["conditional_probability"] > block["groups"]["negative"]["conditional_probability"]


def test_collect_vllm_text_outputs_preserves_legacy_shape():
    outputs = [
        SimpleNamespace(outputs=[SimpleNamespace(text="A")]),
        SimpleNamespace(outputs=[SimpleNamespace(text="B")]),
    ]
    args = SimpleNamespace(decoding_strategy="greedy", sample_size=1)
    assert collect_vllm_text_outputs(args, outputs) == ["A", "B"]

    sampled = [SimpleNamespace(outputs=[SimpleNamespace(text="A1"), SimpleNamespace(text="A2")])]
    args = SimpleNamespace(decoding_strategy="sampling", sample_size=2)
    assert collect_vllm_text_outputs(args, sampled) == [["A1", "A2"]]


def test_save_result_writes_compact_separate_logit_file(tmp_path):
    result_path = tmp_path / "Task-direct-greedy-42.result.json"
    logit_path = tmp_path / "Task-direct-greedy-42.result.logit.json"
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    dataset = [
        {
            "id": "1",
            "task": "Task",
            "type": "clf",
            "language": "en",
            "split": "test",
            "instruction": "private",
            "input": "note",
            "output": "Assessment: Yes",
        }
    ]

    save_result(
        logger=logger,
        dataset=dataset,
        list_response=["Assessment: Yes"],
        path_file_save=result_path,
        save_input=True,
        list_generation=[{"outputs": [{"token_ids": [1, 4]}]}],
        list_logit=[{"pred_label": "Yes"}],
        path_file_logit_save=logit_path,
    )

    assert result_path.exists()
    assert logit_path.exists()
    assert "private" in result_path.read_text(encoding="utf-8")
    logit_text = logit_path.read_text(encoding="utf-8")
    assert "private" not in logit_text
    assert '"logit"' in logit_text


def test_jsonl_sidecar_append_preserves_line_references(tmp_path):
    path = tmp_path / "top_logits.jsonl.gz"
    with JsonlSidecarWriter(path) as writer:
        assert writer.write({"idx": 0})["line"] == 1
    with JsonlSidecarWriter(path, append=True) as writer:
        assert writer.write({"idx": 1})["line"] == 2

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["line"] for row in rows] == [1, 2]
    assert [row["idx"] for row in rows] == [0, 1]


def test_jsonl_sidecar_repairs_truncated_gzip_before_append(tmp_path):
    path = tmp_path / "top_logits.jsonl.gz"
    with JsonlSidecarWriter(path) as writer:
        assert writer.write({"idx": 0})["line"] == 1
        assert writer.write({"idx": 1})["line"] == 2

    compressed = path.read_bytes()
    path.write_bytes(compressed[:-6])

    with JsonlSidecarWriter(path, append=True) as writer:
        assert writer.write({"idx": 2})["line"] == 3

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    assert [row["line"] for row in rows] == [1, 2, 3]
    assert [row["idx"] for row in rows] == [0, 1, 2]


def test_load_partial_vllm_state_requires_exact_dataset_prefix(tmp_path):
    result_path = tmp_path / "Task-direct-greedy-42.result.json"
    logit_path = tmp_path / "Task-direct-greedy-42.result.logit.json"
    logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)
    dataset = [
        {"id": "a", "output": "Assessment: No"},
        {"id": "b", "output": "Assessment: Yes"},
    ]
    save_result(
        logger=logger,
        dataset=dataset[:1],
        list_response=["Assessment: No"],
        path_file_save=result_path,
        save_input=True,
        list_generation=[{"outputs": [{"token_ids": [1, 6]}]}],
        list_logit=[{"pred_label": "No"}],
        path_file_logit_save=logit_path,
    )
    args = SimpleNamespace(path_file_result=str(result_path))
    paths = {
        "logit_json": logit_path,
        "top_logits": tmp_path / "top_logits.jsonl.gz",
    }

    responses, generations, logits, count = _load_partial_vllm_state(
        args=args,
        logger=logger,
        dataset=dataset,
        logit_paths=paths,
        trace_enabled=True,
        label_probability_enabled=True,
    )
    assert count == 1
    assert responses == ["Assessment: No"]
    assert generations[0]["outputs"][0]["token_ids"] == [1, 6]
    assert logits == [{"pred_label": "No"}]

    dataset[0]["id"] = "wrong"
    with pytest.raises(ValueError, match="exact dataset-ID prefix"):
        _load_partial_vllm_state(
            args=args,
            logger=logger,
            dataset=dataset,
            logit_paths=paths,
            trace_enabled=True,
            label_probability_enabled=True,
        )


def test_partial_checkpoint_appends_only_new_rows_and_resumes_without_logits(
    tmp_path,
):
    result_path = tmp_path / "Task-direct-greedy-42.result.json"
    args = SimpleNamespace(
        path_file_result=str(result_path),
        save_input=True,
    )
    dataset = [
        {"id": "a", "output": "No", "input": "one"},
        {"id": "b", "output": "Yes", "input": "two"},
    ]

    saved = _append_partial_vllm_state(
        args=args,
        dataset=dataset,
        list_response=["No"],
        list_generation=None,
        list_logit=None,
        start=0,
    )
    saved = _append_partial_vllm_state(
        args=args,
        dataset=dataset,
        list_response=["No", "Yes"],
        list_generation=None,
        list_logit=None,
        start=saved,
    )

    checkpoint = partial_result_path(result_path)
    lines = checkpoint.read_text(encoding="utf-8").splitlines()
    assert saved == 2
    assert len(lines) == 2
    assert [json.loads(line)["result"]["id"] for line in lines] == ["a", "b"]

    responses, generations, logits, count = _load_partial_vllm_state(
        args=args,
        logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
        dataset=dataset,
        logit_paths={
            "logit_json": tmp_path / "missing.logit.json",
            "top_logits": tmp_path / "missing.top_logits.jsonl.gz",
        },
        trace_enabled=False,
        label_probability_enabled=False,
    )
    assert count == 2
    assert responses == ["No", "Yes"]
    assert generations is None
    assert logits is None
