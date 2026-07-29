import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dataset.task import get_cls_for_all_tasks
from evaluation.bridge import DEFAULT_CONFIG_PATH, config_from_yaml
from evaluation.result_utils import collect_result_inventory, find_result_files
from evaluation.task import (
    DEFAULT_CONFIG_PATH as DEFAULT_TASK_CONFIG_PATH,
)
from evaluation.task import (
    validate_task_config,
)
from evaluation.task_registry import select_task_groups
from util.io import load_json, write_json
from util.local_config import (
    DEFAULT_API_KEY_PATH,
    DEFAULT_MODEL_MAPPING_PATH,
    load_api_key_config,
    load_model_path_mapping,
    resolve_model_path,
)
from util.misc import set_default_args
from util.runtime import (
    load_yaml_config,
    merge_config_into_args,
    normalize_experiments,
    normalize_runtime_args,
    parse_gpu_list,
    parse_path_list,
    parse_prompt_mode,
    validate_args,
)


@pytest.mark.parametrize(
    ("prompt_mode", "expected"),
    [
        ("direct", ("direct", None)),
        ("cot", ("cot", None)),
        ("direct-5-shot", ("direct", 5)),
        ("cot-5-shot", ("cot", 5)),
    ],
)
def test_parse_prompt_mode_valid(prompt_mode, expected):
    assert parse_prompt_mode(prompt_mode) == expected


@pytest.mark.parametrize("prompt_mode", ["directly", "cot-shot", "direct-0-shot"])
def test_parse_prompt_mode_invalid(prompt_mode):
    with pytest.raises(ValueError):
        parse_prompt_mode(prompt_mode)


@pytest.mark.parametrize(
    ("gpu_value", "expected"),
    [
        ("0,1,2", [0, 1, 2]),
        ("0", [0]),
        ([0, "2"], [0, 2]),
    ],
)
def test_parse_gpu_list_valid(gpu_value, expected):
    assert parse_gpu_list(gpu_value) == expected


def test_parse_gpu_list_invalid():
    with pytest.raises(ValueError):
        parse_gpu_list("0,gpu")


def test_parse_path_list_accepts_csv_and_yaml_list():
    assert parse_path_list("dataset_raw, dataset_raw_test", "dir_dataset") == [
        "dataset_raw",
        "dataset_raw_test",
    ]
    assert parse_path_list([" result ", "result_extra"], "dir_result") == [
        "result",
        "result_extra",
    ]
    with pytest.raises(ValueError, match="dir_result"):
        parse_path_list([" ", ""], "dir_result")


def test_normalize_experiments_casts_seed_to_int():
    assert normalize_experiments(
        [["direct", "GREEDY", "42"], ("cot-2-shot", "sampling", 43)]
    ) == [["direct", "greedy", 42], ["cot-2-shot", "sampling", 43]]


def test_load_yaml_config_rejects_duplicate_keys(tmp_path):
    path_config = tmp_path / "config.yaml"
    path_config.write_text("engine: vllm\nengine: hf\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate key"):
        load_yaml_config(path_config)


def test_utility_split_uses_direct_modules(tmp_path):
    path_json = tmp_path / "nested" / "data.json"
    write_json({"ok": True}, path_json)

    assert load_json(path_json) == {"ok": True}
    assert set_default_args(SimpleNamespace(), "missing", "default") == (
        "default",
        True,
    )


def test_merge_config_into_args_preserves_cli_priority():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default=None)
    parser.add_argument("--batch_size", default=1, type=int)
    args = parser.parse_args(["--batch_size", "8"])

    merge_config_into_args(
        args,
        parser,
        {"engine": "vllm", "batch_size": 4, "save_input": False},
    )

    assert args.engine == "vllm"
    assert args.batch_size == 8
    assert args.save_input is False
    assert args._manual_config_keys == {"engine", "batch_size", "save_input"}


def test_validate_and_normalize_runtime_args(tmp_path):
    model_file = tmp_path / "dict_model_path.json"
    model_file.write_text(json.dumps({"model-a": str(tmp_path)}), encoding="utf-8")
    args = SimpleNamespace(
        tasks=["ADE-Identification"],
        experiments=[["direct", "greedy", "42"], ["cot-2-shot", "sampling", 43]],
        mode="test",
        engine="vllm",
        dir_dataset="dataset_raw,dataset_raw_test",
        dir_result="result,result_extra",
        batch_size=2,
        model_file=str(model_file),
        gpus="0,1",
        max_token_all=4096,
        max_token_input=3072,
        max_token_output=1024,
    )

    validate_args(args)
    normalize_runtime_args(args)

    assert args.gpus == [0, 1]
    assert args.num_workers == 4
    assert args.dir_dataset == ["dataset_raw", "dataset_raw_test"]
    assert args.dir_result == ["result", "result_extra"]
    assert args.dir_result_save == "result"
    assert args.experiments == [["direct", "greedy", 42], ["cot-2-shot", "sampling", 43]]


def test_public_task_names_are_registered():
    root = Path(__file__).resolve().parents[1]
    config = load_yaml_config(root / "configs" / "BRIDGE.example.yaml")
    _, all_tasks = get_cls_for_all_tasks()

    assert "ADE-Drug dosage" in all_tasks
    assert "ADE-Identification" in all_tasks
    assert "BrainMRI-AIS" in all_tasks
    assert "1" + "-1.ADE-ADE identification" not in all_tasks
    assert "1" + "-3.ADE-Drug dosage" not in all_tasks
    assert set(config["tasks"]).issubset(all_tasks)


def test_model_config_vllm_casts_string_seed(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.repetition_penalty = kwargs.get("repetition_penalty", 1.0)

    monkeypatch.setitem(
        sys.modules,
        "vllm",
        SimpleNamespace(SamplingParams=FakeSamplingParams),
    )
    from model.init import model_config_vllm

    args = SimpleNamespace(
        decoding_strategy="greedy",
        seed="42",
        max_token_output=16,
        model_name="model-a",
    )
    logger = SimpleNamespace(info=lambda *_: None, warning=lambda *_: None)

    sampling_params = model_config_vllm(args, logger)

    assert sampling_params.kwargs["seed"] == 42
    assert args.seed == 42


def test_normalize_runtime_args_accepts_yaml_path_lists(tmp_path):
    args = SimpleNamespace(
        gpus=[0],
        dir_dataset=[" dataset_raw ", "dataset_raw_test"],
        dir_result=[" result ", "result_extra"],
    )

    normalize_runtime_args(args)

    assert args.gpus == [0]
    assert args.num_workers == 2
    assert args.dir_dataset == ["dataset_raw", "dataset_raw_test"]
    assert args.dir_result == ["result", "result_extra"]
    assert args.dir_result_save == "result"


def test_normalize_runtime_args_respects_explicit_num_workers(tmp_path):
    model_file = tmp_path / "dict_model_path.json"
    model_file.write_text(json.dumps({"model-a": str(tmp_path)}), encoding="utf-8")
    args = SimpleNamespace(
        tasks=["task"],
        experiments=[["direct", "greedy", 42]],
        mode="test",
        engine="vllm",
        dir_dataset="dataset_raw",
        dir_result="result",
        batch_size=1,
        model_file=str(model_file),
        gpus="0,1",
        num_workers=0,
    )

    validate_args(args)
    normalize_runtime_args(args)

    assert args.num_workers == 0


def test_validate_args_rejects_negative_num_workers(tmp_path):
    model_file = tmp_path / "dict_model_path.json"
    model_file.write_text(json.dumps({"model-a": str(tmp_path)}), encoding="utf-8")
    args = SimpleNamespace(
        tasks=["task"],
        experiments=[["direct", "greedy", 42]],
        mode="test",
        engine="vllm",
        dir_dataset="dataset_raw",
        dir_result="result",
        batch_size=1,
        model_file=str(model_file),
        gpus="0",
        num_workers=-1,
    )

    with pytest.raises(ValueError, match="num_workers"):
        validate_args(args)


def test_validate_args_rejects_bad_prompt_mode(tmp_path):
    model_file = tmp_path / "dict_model_path.json"
    model_file.write_text(json.dumps({"model-a": str(tmp_path)}), encoding="utf-8")
    args = SimpleNamespace(
        tasks=["task"],
        experiments=[["directly", "greedy", 42]],
        mode="test",
        engine="vllm",
        dir_dataset="dataset_raw",
        dir_result="result",
        batch_size=1,
        model_file=str(model_file),
    )

    with pytest.raises(ValueError, match="Invalid prompt_mode"):
        validate_args(args)


def test_load_api_key_config_validates_provider_fields(tmp_path):
    path_config = tmp_path / "API.key.yaml"
    path_config.write_text(
        "\n".join(
            [
                "AZURE-OPENAI:",
                "  endpoint: https://example.openai.azure.com/",
                "  api_version: 2025-04-01-preview",
                "  api_key: test-key",
                "OPENAI:",
                "  api_key: openai-key",
            ]
        ),
        encoding="utf-8",
    )

    azure_config = load_api_key_config(path_config, "AZURE-OPENAI")
    openai_config = load_api_key_config(path_config, "OPENAI")

    assert azure_config["endpoint"].startswith("https://")
    assert azure_config["api_version"] == "2025-04-01-preview"
    assert openai_config == {"api_key": "openai-key"}


def test_load_api_key_config_rejects_missing_provider_field(tmp_path):
    path_config = tmp_path / "API.key.yaml"
    path_config.write_text(
        "AZURE-OPENAI:\n  endpoint: https://example.openai.azure.com/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required field"):
        load_api_key_config(path_config, "AZURE-OPENAI")


def test_model_mapping_loader_and_resolver(tmp_path):
    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    path_mapping = tmp_path / "dict_model_path.json"
    path_mapping.write_text(
        json.dumps({"model-a": str(model_dir), "model-b": "/not/checked"}),
        encoding="utf-8",
    )

    mapping = load_model_path_mapping(path_mapping)

    assert mapping["model-a"] == str(model_dir)
    assert resolve_model_path("model-a", path_mapping) == str(model_dir)
    assert resolve_model_path("model-b", path_mapping, must_exist=False) == "/not/checked"


def test_model_mapping_loader_rejects_invalid_entries(tmp_path):
    path_mapping = tmp_path / "dict_model_path.json"
    path_mapping.write_text(json.dumps({"model-a": ""}), encoding="utf-8")

    with pytest.raises(ValueError, match="non-empty string"):
        load_model_path_mapping(path_mapping)


def test_default_local_config_paths_use_configs_folder():
    assert DEFAULT_API_KEY_PATH.as_posix() == "configs/API.key.yaml"
    assert DEFAULT_MODEL_MAPPING_PATH.as_posix() == "configs/dict_model_path.json"


def test_default_model_mapping_falls_back_to_legacy_root_file(tmp_path, monkeypatch):
    model_dir = tmp_path / "model-a"
    model_dir.mkdir()
    path_mapping = tmp_path / "dict_model_path.json"
    path_mapping.write_text(json.dumps({"model-a": str(model_dir)}), encoding="utf-8")

    monkeypatch.chdir(tmp_path)

    assert load_model_path_mapping()["model-a"] == str(model_dir)
    assert resolve_model_path("model-a") == str(model_dir)


def test_default_evaluation_config_path_exists_and_parses():
    assert DEFAULT_CONFIG_PATH.name == "evaluate.BRIDGE.yaml"
    assert DEFAULT_CONFIG_PATH.exists()

    config = config_from_yaml(DEFAULT_CONFIG_PATH)

    assert config.dir_result == ["result"]
    assert config.prompt_modes == ["direct"]
    assert set(config.task_types) == {"clf", "ext", "gen"}


def test_default_task_evaluation_config_path_exists_and_requires_custom_tasks():
    assert DEFAULT_TASK_CONFIG_PATH.name == "evaluate.task.yaml"
    assert DEFAULT_TASK_CONFIG_PATH.exists()

    config = config_from_yaml(DEFAULT_TASK_CONFIG_PATH)

    assert config.tasks
    assert config.custom_tasks
    assert any(
        task_name in task_mapping
        for task_mapping in config.custom_tasks.values()
        for task_name in config.tasks
    )
    validate_task_config(config)


def test_task_entry_rejects_bridge_config_without_custom_tasks():
    config = config_from_yaml(DEFAULT_CONFIG_PATH)

    with pytest.raises(ValueError, match="custom_tasks"):
        validate_task_config(config)


def test_result_inventory_searches_multiple_result_roots(tmp_path):
    task = "Task-A"
    model_a = "model-a"
    model_b = "model-b"
    root_a = tmp_path / "result_a"
    root_b = tmp_path / "result_b"
    paths = [
        root_a / task / model_a / f"{task}-direct-greedy-42.result.json",
        root_b / task / model_a / f"{task}-direct-greedy-43.result.json",
        root_b / task / model_b / f"{task}-cot-greedy-42.result.json",
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")

    direct_files = find_result_files(
        dir_result=[root_a, root_b],
        task_names=[task],
        prompt_modes=["direct"],
    )
    assert direct_files == sorted(paths[:2])

    inventory = collect_result_inventory(
        dir_result=f"{root_a},{root_b}",
        task_names=[task],
        model_names=[model_a],
        prompt_modes=["direct"],
    )
    assert [item["path"] for item in inventory] == [str(path) for path in sorted(paths[:2])]


def test_general_evaluation_searches_multiple_result_roots(tmp_path):
    pytest.importorskip("tqdm")
    from dataset.dataset import GeneralEvaluation

    task = "Task-A"
    dataset_root = tmp_path / "dataset_raw"
    dataset_root.mkdir()
    (dataset_root / f"{task}.SFT.json").write_text(
        json.dumps(
            [
                {
                    "id": "1",
                    "split": "test",
                    "language": "en",
                    "type": "classification",
                    "instruction": "Instruction",
                    "input": "Input",
                    "output": "Answer: yes",
                }
            ]
        ),
        encoding="utf-8",
    )

    root_a = tmp_path / "result_a"
    root_b = tmp_path / "result_b"
    result_paths = [
        root_a / task / "model-a" / f"{task}-direct-greedy-42.result.json",
        root_b / task / "model-a" / f"{task}-direct-greedy-43.result.json",
        root_b / task / "model-b" / f"{task}-direct-greedy-42.result.json",
    ]
    for path in result_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("[]", encoding="utf-8")

    args = SimpleNamespace(
        dir_dataset=[str(dataset_root)],
        dir_result=[str(root_a), str(root_b)],
        seed=42,
        sample_eval_index=0,
    )
    evaluation = GeneralEvaluation(args=args, task=task)

    assert evaluation.search_result_by_model("model-a", "direct") == {
        "model-a": [str(result_paths[0]), str(result_paths[1])]
    }
    assert evaluation.search_result_by_model(prompt_mode="direct") == {
        "model-a": [str(result_paths[0]), str(result_paths[1])],
        "model-b": [str(result_paths[2])],
    }


def test_evaluation_registry_accepts_yaml_custom_tasks(tmp_path, monkeypatch):
    pytest.importorskip("regex")
    module_dir = tmp_path / "custom_pkg"
    module_dir.mkdir()
    (module_dir / "__init__.py").write_text("", encoding="utf-8")
    (module_dir / "tasks.py").write_text(
        "\n".join(
            [
                "class CustomExtTask:",
                "    pass",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    selected = select_task_groups(
        task_types=["ext"],
        tasks=["Project-Task"],
        custom_tasks={"ext": {"Project-Task": "custom_pkg.tasks:CustomExtTask"}},
    )

    assert selected["ext"]["registry"]["Project-Task"].__name__ == "CustomExtTask"


def test_main_api_auto_evaluation_defaults_to_all_task_types(monkeypatch):
    import evaluation.runner
    import main_api

    captured = {}

    def fake_run_config(config):
        captured["task_types"] = config.task_types
        captured["tasks"] = config.tasks

    monkeypatch.setattr(evaluation.runner, "run_config", fake_run_config)

    main_api.run_evaluation(
        run_cfg={
            "dir_dataset": ["dataset_raw"],
            "dir_result": ["result"],
            "tasks": ["Task-A"],
        },
        model_names=["model-a"],
        prompt_modes=["direct"],
    )

    assert captured == {
        "task_types": ["clf", "ext", "gen"],
        "tasks": ["Task-A"],
    }


def test_api_experiment_dry_run_does_not_create_output_dirs(tmp_path, monkeypatch):
    from model import api_runner

    monkeypatch.setattr(
        api_runner,
        "load_task_dataset",
        lambda **_: ([{"id": "1", "language": "en"}], []),
    )
    monkeypatch.setattr(
        api_runner,
        "build_messages_for_dataset",
        lambda **_: {"1": [{"role": "user", "content": "hello"}]},
    )

    result = api_runner.run_api_experiment(
        provider_name="azure",
        model_name="ad-hoc-model",
        run_cfg={
            "mode": "test",
            "dir_dataset": [str(tmp_path / "dataset")],
            "dir_result": [str(tmp_path / "result")],
            "chat_params": {},
            "require_model_registered": False,
            "save_trace": True,
        },
        api_model_cfg={},
        api_key_cfg={},
        task_name="Task-A",
        prompt_mode="direct",
        decoding_strategy="greedy",
        seed=42,
        dry_run=True,
    )

    assert result.dry_run
    assert result.result_path == (
        tmp_path
        / "result"
        / "Task-A"
        / "ad-hoc-model"
        / "Task-A-direct-greedy-42.result.json"
    )
    assert not (tmp_path / "result").exists()


def test_documentation_file_naming_convention():
    """Keep public documentation English-only and consistently named."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "README.md").exists()
    legacy_root_readme = "README" + ".zh"
    legacy_azure_zh = "README" + "_AZURE_CN.md"
    assert not (root / legacy_root_readme).exists()
    assert not (root / "README-zh.md").exists()
    assert (root / "USAGE.md").exists()
    assert not (root / "USAGE-zh.md").exists()

    documented_dirs = [
        "configs",
        "dataset",
        "model",
        "metric",
        "util",
        "evaluation",
        "notebooks",
        "notebooks/evaluation",
        "notebooks/inference",
        "scripts",
        "tests",
        "data_contamination",
        "fig",
    ]
    for dirname in documented_dirs:
        path_dir = root / dirname
        assert (path_dir / "README.md").exists()
        assert not (path_dir / "README-zh.md").exists()
        assert not (path_dir / legacy_root_readme).exists()

    assert (root / "configs" / "API_USAGE.md").exists()
    assert not (root / "configs" / "API_USAGE-zh.md").exists()
    assert not (root / "model" / legacy_azure_zh).exists()


def test_root_entry_files_are_kept_small_after_reorganization():
    """Root keeps core entry points while configs/notebooks stay in folders."""
    root = Path(__file__).resolve().parents[1]

    assert not (root / "BRIDGE.yaml").exists()
    assert not (root / "API.key.example.yaml").exists()
    assert not (root / "dict_model_path.example.json").exists()
    assert not (root / "Inference task.Local.ipynb").exists()
    assert not (root / "Inference task.Azure.ipynb").exists()
    assert not (root / ".vscode").exists()
    assert not (root / "run_time.sh").exists()
    assert not (root / "scripts" / "run_time.sh").exists()
    assert not (root / "scripts" / "run.sh").exists()
    assert not (root / "scripts" / "run_multiple_model.sh").exists()

    assert (root / "configs" / "BRIDGE.example.yaml").exists()
    assert not (root / "configs" / "BRIDGE.yaml").exists()
    assert (root / "run.sh").exists()
    assert (root / "run_multiple_model.sh").exists()
    assert (root / "notebooks" / "inference" / "Inference task.Local.ipynb").exists()
    assert (root / "notebooks" / "inference" / "Inference task.Azure.ipynb").exists()
