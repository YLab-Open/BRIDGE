from types import SimpleNamespace

import pytest

from evaluation.probability import (
    compute_binary_metrics_for_rule,
    compute_binary_probability_metrics,
    compute_binary_rule_metrics_with_ci,
    compute_probability_summary_with_ci,
    extract_label_from_text,
    infer_label_token_probe_target,
    paired_bootstrap_score_metric_difference,
    paired_delong_auroc_difference,
    score_label_groups_from_logits,
    sweep_binary_probability_metrics,
    top_token_rows_from_logits,
)


class FakeTokenizer:
    def __init__(self):
        self.id_to_text = {
            0: "0",
            1: "1",
            2: "2",
            3: "3",
            4: "4",
            5: "x",
        }

    def decode(self, token_ids):
        return "".join(self.id_to_text[int(token_id)] for token_id in token_ids)


def test_extract_label_from_assessment_output():
    info = extract_label_from_text(
        "Assessment: 1\n",
        label_values=["0", "1", "2", "3", "4"],
        answer_prefix="Assessment:",
    )
    assert info["label"] == "1"
    assert info["prefix_before_label"] == "Assessment: "


def test_extract_label_requires_configured_answer_prefix():
    with pytest.raises(ValueError, match="answer_prefix"):
        extract_label_from_text(
            'The label is "Yes".',
            label_values=["No", "Yes"],
            answer_prefix="Assessment:",
        )


def test_extract_label_requires_label_immediately_after_answer_prefix():
    with pytest.raises(ValueError, match="immediately after"):
        extract_label_from_text(
            "Assessment: the answer is Yes",
            label_values=["No", "Yes"],
            answer_prefix="Assessment:",
        )


def test_token_aligned_probe_target_can_include_leading_space_token():
    class SpaceLabelTokenizer(FakeTokenizer):
        def __init__(self):
            self.id_to_text = {10: "Assessment", 11: ":", 12: " Yes"}

        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            if text == "Assessment: Yes":
                return [10, 11, 12]
            raise AssertionError(text)

    info = infer_label_token_probe_target(
        "Assessment: Yes",
        tokenizer=SpaceLabelTokenizer(),
        label_values=["No", "Yes"],
        answer_prefix="Assessment:",
    )

    assert info["label"] == "Yes"
    assert info["prefix_before_label"] == "Assessment: "
    assert info["prefix_before_target_token"] == "Assessment:"
    assert info["target_token_text"] == " Yes"
    assert info["target_token_prefix"] == " "


def test_score_label_groups_from_logits_uses_probability_valid_aggregation():
    logits = {
        0: SimpleNamespace(logprob=0.0, rank=5),
        1: SimpleNamespace(logprob=1.0, rank=4),
        2: SimpleNamespace(logprob=2.0, rank=3),
        3: SimpleNamespace(logprob=3.0, rank=2),
        4: SimpleNamespace(logprob=4.0, rank=1),
        5: SimpleNamespace(logprob=5.0, rank=0),
    }
    score = score_label_groups_from_logits(
        logits,
        tokenizer=FakeTokenizer(),
        label_token_ids={str(i): [i] for i in range(5)},
        label_groups={"negative": ["0", "1"], "positive": ["2", "3", "4"]},
    )

    assert score["labels"]["1"]["logit"] == pytest.approx(1.0)
    assert score["groups"]["negative"]["logit"] > 1.0
    assert score["groups"]["positive"]["conditional_probability"] > 0.9
    total_conditional = (
        score["groups"]["negative"]["conditional_probability"]
        + score["groups"]["positive"]["conditional_probability"]
    )
    assert total_conditional == pytest.approx(1.0)


def test_top_token_rows_append_label_tokens_outside_saved_top_k():
    logits = {
        0: SimpleNamespace(logprob=10.0, rank=1),
        1: SimpleNamespace(logprob=9.0, rank=2),
        2: SimpleNamespace(logprob=1.0, rank=50),
    }

    rows = top_token_rows_from_logits(
        logits,
        tokenizer=FakeTokenizer(),
        top_k=1,
        include_token_ids=[2],
    )

    assert [row["token_id"] for row in rows] == [0, 2]
    assert rows[0]["selection"] == "top_k"
    assert rows[1]["selection"] == "label_token"


def test_compute_binary_probability_metrics_thresholds():
    rows = [
        {"y_true": 0, "score": 0.1},
        {"y_true": 0, "score": 0.4},
        {"y_true": 1, "score": 0.6},
        {"y_true": 1, "score": 0.9},
    ]
    metrics = compute_binary_probability_metrics(rows, thresholds=[0.5, 0.7])

    assert metrics["auroc"] == pytest.approx(1.0)
    assert metrics["auprc"] == pytest.approx(1.0)
    assert metrics["thresholds"]["0.5"]["tp"] == 2
    assert metrics["thresholds"]["0.5"]["tn"] == 2
    assert metrics["thresholds"]["0.7"]["recall"] == pytest.approx(0.5)


def test_extended_binary_probability_metrics_and_positive_gt_negative_rule():
    rows = [
        {"y_true": 0, "score": 0.2, "positive_score": 0.2, "negative_score": 0.8},
        {"y_true": 1, "score": 0.8, "positive_score": 0.8, "negative_score": 0.2},
    ]
    metrics = compute_binary_metrics_for_rule(rows, positive_gt_negative=True)
    assert metrics["Accuracy"] == pytest.approx(1.0)
    assert metrics["F1 (Pos)"] == pytest.approx(1.0)
    assert metrics["PPV"] == pytest.approx(1.0)
    assert metrics["NPV"] == pytest.approx(1.0)

    swept = sweep_binary_probability_metrics(rows, thresholds=[0.5])
    assert [row["rule"] for row in swept] == ["threshold", "positive>negative"]


def test_probability_bootstrap_ci_helpers_are_reproducible():
    rows = [
        {"id": "a", "y_true": 0, "score": 0.1, "positive_score": 0.1, "negative_score": 0.9},
        {"id": "b", "y_true": 0, "score": 0.4, "positive_score": 0.4, "negative_score": 0.6},
        {"id": "c", "y_true": 1, "score": 0.6, "positive_score": 0.6, "negative_score": 0.4},
        {"id": "d", "y_true": 1, "score": 0.9, "positive_score": 0.9, "negative_score": 0.1},
    ]

    summary = compute_probability_summary_with_ci(rows, n_bootstrap=20, seed=7)
    assert summary["auroc"] == pytest.approx(1.0)
    assert summary["auroc_ci_low"] == pytest.approx(1.0)
    assert summary["bootstrap_n"] == 20

    rule = compute_binary_rule_metrics_with_ci(
        rows,
        threshold=0.5,
        n_bootstrap=20,
        seed=7,
    )
    assert rule["Accuracy"] == pytest.approx(1.0)
    assert rule["Accuracy ci_low"] == pytest.approx(1.0)


def test_paired_bootstrap_score_metric_difference_aligns_by_id():
    rows_a = [
        {"id": "a", "y_true": 0, "score": 0.1},
        {"id": "b", "y_true": 0, "score": 0.8},
        {"id": "c", "y_true": 1, "score": 0.4},
        {"id": "d", "y_true": 1, "score": 0.9},
    ]
    rows_b = [
        {"id": "d", "y_true": 1, "score": 0.95},
        {"id": "c", "y_true": 1, "score": 0.85},
        {"id": "b", "y_true": 0, "score": 0.2},
        {"id": "a", "y_true": 0, "score": 0.1},
    ]

    diff = paired_bootstrap_score_metric_difference(
        rows_a,
        rows_b,
        metric="auroc",
        n_bootstrap=20,
        seed=7,
    )
    assert diff["b"] > diff["a"]
    assert diff["delta_b_minus_a"] > 0
    assert diff["n"] == 4


def test_paired_delong_auroc_difference_aligns_by_id():
    rows_a = [
        {"id": "a", "y_true": 0, "score": 0.10},
        {"id": "b", "y_true": 0, "score": 0.60},
        {"id": "c", "y_true": 0, "score": 0.55},
        {"id": "d", "y_true": 1, "score": 0.45},
        {"id": "e", "y_true": 1, "score": 0.70},
        {"id": "f", "y_true": 1, "score": 0.80},
    ]
    rows_b = [
        {"id": "f", "y_true": 1, "score": 0.95},
        {"id": "e", "y_true": 1, "score": 0.85},
        {"id": "d", "y_true": 1, "score": 0.75},
        {"id": "c", "y_true": 0, "score": 0.40},
        {"id": "b", "y_true": 0, "score": 0.30},
        {"id": "a", "y_true": 0, "score": 0.20},
    ]

    diff = paired_delong_auroc_difference(rows_a, rows_b)

    assert diff["test"] == "DeLong"
    assert diff["b"] > diff["a"]
    assert diff["delta_b_minus_a"] > 0
    assert 0.0 <= diff["p_two_sided_delong"] <= 1.0
    assert diff["n"] == 6
