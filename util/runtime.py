import argparse
import re
from pathlib import Path

import yaml


PROMPT_MODE_PATTERN = re.compile(r"^(direct|cot)(?:-([1-9]\d*)-shot)?$")


def parse_gpu_list(gpu_value) -> list[int]:
    """
    Convert a GPU list from YAML/CLI into integer device IDs.

    Accepts either a YAML list such as `[0, 1]` or a CLI string such as `0,1`.
    """
    if isinstance(gpu_value, list):
        return [int(x) for x in gpu_value]
    try:
        return [int(x) for x in str(gpu_value).split(",") if x != ""]
    except ValueError as e:
        raise ValueError(f"Invalid GPU list format: {gpu_value}") from e


def parse_path_list(path_value, field_name: str) -> list[str]:
    """
    Normalize a path field from YAML/CLI into a non-empty list of strings.

    Accepts a comma-separated string or a YAML list. This is used for fields
    such as `dir_dataset` and `dir_result`.
    """
    if isinstance(path_value, str):
        paths = path_value.split(",")
    elif isinstance(path_value, (list, tuple)):
        paths = path_value
    else:
        paths = [path_value]

    normalized = [str(item).strip() for item in paths if str(item).strip()]
    if not normalized:
        raise ValueError(f"{field_name} must contain at least one path.")
    return normalized


def parse_prompt_mode(prompt_mode: str) -> tuple[str, int | None]:
    """
    Parse and validate one prompt mode.

    Supported forms are:
    - direct
    - cot
    - direct-N-shot
    - cot-N-shot
    """
    if not isinstance(prompt_mode, str):
        raise ValueError(f"prompt_mode must be a string, got {type(prompt_mode).__name__}")
    match = PROMPT_MODE_PATTERN.fullmatch(prompt_mode.strip())
    if not match:
        raise ValueError(
            "Invalid prompt_mode: "
            f"{prompt_mode!r}. Expected direct, cot, direct-N-shot, or cot-N-shot."
        )
    prompt_base, num_example = match.groups()
    return prompt_base, int(num_example) if num_example is not None else None


class UniqueKeyLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate keys instead of silently overriding."""


def _construct_mapping_unique_keys(loader, node, deep=False):
    """Construct one YAML mapping and raise if the same key appears twice."""
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise ValueError(f"Duplicate key in YAML config: {key}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_mapping_unique_keys,
)


def load_yaml_config(path_config: str | None) -> dict:
    """Load a YAML config file as a mapping and reject duplicate keys."""
    if not path_config:
        return {}

    cfg_path = Path(path_config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

    with cfg_path.open(encoding="utf-8") as f:
        loaded = yaml.load(f, Loader=UniqueKeyLoader)
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML mapping: {cfg_path}")
    return loaded


def merge_config_into_args(args, parser, dict_cfg: dict):
    """
    Merge YAML values into argparse args.

    CLI values keep priority for parser-defined arguments. YAML-only settings are
    copied directly to args so model and runtime helpers can consume them.
    """
    args._config_keys = set(dict_cfg.keys())
    args._manual_config_keys = {
        key for key, value in dict_cfg.items() if value is not None
    }

    parser_defaults = {
        action.dest: parser.get_default(action.dest)
        for action in parser._actions
        if action.dest != argparse.SUPPRESS
    }
    for key, value in dict_cfg.items():
        if key in parser_defaults:
            if getattr(args, key, None) == parser_defaults[key]:
                setattr(args, key, value)
        else:
            setattr(args, key, value)
    return args


def validate_args(args):
    """Validate runtime config before loading models or datasets."""
    required_fields = [
        "tasks",
        "experiments",
        "mode",
        "engine",
        "dir_dataset",
        "dir_result",
    ]
    missing = [field for field in required_fields if not hasattr(args, field)]
    if missing:
        raise ValueError(f"Missing required config field(s): {missing}")

    if args.engine is None:
        raise ValueError("engine must be set in YAML or passed by CLI.")
    if args.engine not in {"vllm", "hf"}:
        raise ValueError("engine must be one of: vllm, hf")
    if args.mode not in {"train", "dev", "test", "all"}:
        raise ValueError("mode must be one of: train, dev, test, all")

    if not isinstance(args.tasks, list) or len(args.tasks) == 0:
        raise ValueError("tasks must be a non-empty list.")
    if not all(isinstance(task, str) and task.strip() for task in args.tasks):
        raise ValueError("Every task entry must be a non-empty string.")

    if not isinstance(args.experiments, list) or len(args.experiments) == 0:
        raise ValueError("experiments must be a non-empty list.")
    valid_decoding = {"greedy", "sampling"}
    for idx, item in enumerate(args.experiments):
        if not isinstance(item, (list, tuple)) or len(item) != 3:
            raise ValueError(
                f"experiments[{idx}] must be [prompt_mode, decoding_strategy, seed]."
            )
        prompt_mode, decoding_strategy, seed = item
        try:
            parse_prompt_mode(prompt_mode)
        except ValueError as e:
            raise ValueError(f"Invalid prompt_mode in experiments[{idx}]: {prompt_mode}") from e
        if str(decoding_strategy).lower() not in valid_decoding:
            raise ValueError(
                f"Invalid decoding_strategy in experiments[{idx}]: {decoding_strategy}"
            )
        int(seed)

    if int(args.batch_size) <= 0:
        raise ValueError("batch_size must be a positive integer.")

    if hasattr(args, "save_every_n_batches") and args.save_every_n_batches is not None:
        if int(args.save_every_n_batches) <= 0:
            raise ValueError("save_every_n_batches must be a positive integer or null.")

    token_fields = ["max_token_all", "max_token_input", "max_token_output"]
    if all(hasattr(args, field) for field in token_fields):
        if int(args.max_token_input) + int(args.max_token_output) > int(
            args.max_token_all
        ):
            raise ValueError(
                "max_token_input + max_token_output cannot exceed max_token_all."
            )

    if not Path(args.model_file).exists():
        raise FileNotFoundError(f"Model mapping file not found: {args.model_file}")


def normalize_runtime_args(args):
    """Normalize path, GPU, and worker-count values after config validation."""
    args.gpus = parse_gpu_list(args.gpus)
    args.num_workers = len(args.gpus) * 2
    args.dir_dataset = parse_path_list(args.dir_dataset, "dir_dataset")
    args.dir_result = parse_path_list(args.dir_result, "dir_result")
    args.dir_result_save = args.dir_result[0]
    return args


def log_task(args, logger, task_name, task, str_time_start_task):
    """Log task metadata and runtime parameters at the start of a task."""
    logger.info(f"Model: {args.model_name}: {args.model_path}")
    logger.info(f"Task: {task_name}")
    logger.info(
        f"Size: train={len(task.dataset_train)}, dev={len(task.dataset_dev)}, "
        f"test={len(task.dataset_test)}"
    )

    logger.info("\n" + "-" * 30 + "\n")
    logger.info("Experiment parameters:")
    for key, value in args.__dict__.items():
        if key not in ["tasks", "temperature", "top_p", "top_k"]:
            logger.info(f" - {key}: {value}")
    logger.info("\n" + "-" * 30 + "\n")

    logger.info(f"Start on: {str_time_start_task}")
    logger.info("\n" + "=" * 50 + "\n")


def get_task_dataloader(task, mode, batch_size, num_workers):
    """Return the selected split dataset and dataloader for a task."""
    if mode == "test":
        return task.dataloader_test(batch_size, num_workers)
    if mode == "dev":
        return task.dataloader_dev(batch_size, num_workers)
    if mode == "train":
        return task.dataloader_train(batch_size, num_workers)
    if mode == "all":
        return task.dataloader_all(batch_size, num_workers)
    raise ValueError(f"Invalid mode: {mode}. Choose from train, dev, test, or all.")
