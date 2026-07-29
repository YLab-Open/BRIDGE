from dataset.config import get_metrics_ext, get_metrics_ext_subject
from metric.extraction import print_metrics_ext_subject


def test_get_metrics_ext_subject_is_subject_only():
    assert get_metrics_ext_subject() == [
        "precision_subject",
        "recall_subject",
        "f1_subject",
        "num_failed_ratio",
    ]
    assert get_metrics_ext() == ["f1_subject", "f1_event", "num_failed_ratio"]


def test_print_metrics_ext_subject_defaults_to_subject_metrics():
    dict_model_metrics = {
        "model-a": {
            "precision_subject": {"mean": 10.0, "ci": [9.0, 11.0]},
            "recall_subject": {"mean": 20.0, "ci": [19.0, 21.0]},
            "f1_subject": {"mean": 30.0, "ci": [29.0, 31.0]},
            "f1_event": {"mean": 40.0, "ci": [39.0, 41.0]},
            "num_failed_ratio": {"mean": 0.0, "ci": [0.0, 0.0]},
        }
    }

    assert print_metrics_ext_subject(dict_model_metrics, list_model=["model-a"]) == (
        "10.00 [9.00, 11.00]; "
        "20.00 [19.00, 21.00]; "
        "30.00 [29.00, 31.00]; "
        "0.00 [0.00, 0.00];"
    )


def test_medication_format_parser_accepts_common_model_variants(tmp_path):
    import json
    from types import SimpleNamespace

    import pytest

    pytest.importorskip("regex")
    from dataset.extraction import Task_ext_Medication_format

    path_dataset = tmp_path / "dataset"
    path_dataset.mkdir()
    (path_dataset / "Medication.SFT.json").write_text(
        json.dumps(
            [
                {
                    "task": "Medication",
                    "language": "en",
                    "type": "ext",
                    "id": 0,
                    "split": "test",
                    "instruction": "",
                    "input": "",
                    "output": "medication: lasix, status: Active;",
                }
            ]
        ),
        encoding="utf-8",
    )
    args = SimpleNamespace(dir_dataset=[str(path_dataset)], dir_result=str(tmp_path))
    task = Task_ext_Medication_format(args=args, task="Medication")

    preds = task.get_pred(
        [
            {
                "pred": "\n".join(
                    [
                        "Levofloxacin: Active;",
                        "Lasix, status: Discontinued;",
                        "- prochlorperazine: ..., status: Active;",
                        "2. ___: Discontinued;",
                        "medication: Haldol, status: PRN;",
                        "medication: sodium chloride bolus 1,000 ml, status: Active;",
                    ]
                )
            }
        ]
    )

    assert preds == [
        [
            {"medication": "levofloxacin", "status": "active"},
            {"medication": "lasix", "status": "discontinued"},
            {"medication": "prochlorperazine", "status": "active"},
            {"medication": "haldol", "status": "prn"},
            {"medication": "sodium chloride bolus 1,000 ml", "status": "active"},
        ]
    ]
