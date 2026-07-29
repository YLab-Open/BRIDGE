import json

import yaml

from model.metadata import result_metadata_paths, write_result_metadata


def test_metadata_writes_redacted_config_snapshot_and_file_hashes(tmp_path):
    result_path = tmp_path / "Task-direct-greedy-42.result.json"
    result_path.write_text('[{"id": "1", "pred": "Yes"}]', encoding="utf-8")
    task_path = tmp_path / "Task.SFT.json"
    task_path.write_text("[]", encoding="utf-8")

    paths = write_result_metadata(
        result_path=result_path,
        run_type="api_azure",
        n_records=1,
        effective_config={
            "model": "gpt-4o",
            "api_key": "must-not-leak",
            "api": {"endpoint": "must-not-leak-either"},
        },
        source_files={"task_data": task_path},
        experiment={"seed": 42},
    )

    assert paths == result_metadata_paths(result_path)
    snapshot = yaml.safe_load(paths["config"].read_text(encoding="utf-8"))
    assert snapshot["api_key"] == "***"
    assert snapshot["api"]["endpoint"] == "***"
    metadata_text = paths["metadata"].read_text(encoding="utf-8")
    assert "must-not-leak" not in metadata_text
    metadata = json.loads(metadata_text)
    assert metadata["schema_version"] == 3
    assert metadata["n_records"] == 1
    assert metadata["files"]["artifacts"]["result"]["sha256"]
    assert metadata["files"]["source"]["task_data"]["sha256"]
