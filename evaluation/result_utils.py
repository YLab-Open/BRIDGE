from pathlib import Path
from typing import Iterable

from .common import read_json
from util.runtime import parse_path_list

RESULT_SUFFIX = ".result.json"
DECODING_MARKERS = ("greedy", "sampling")


def parse_result_file(path_file: str | Path) -> dict:
    """
    Parse the standard BRIDGE result path.

    Expected layout:
    `<dir_result>/<task>/<model>/<task>-<prompt_mode>-<decoding>-<seed>.result.json`
    """
    path_file = Path(path_file)
    task_name = path_file.parent.parent.name
    model_name = path_file.parent.name
    file_name = path_file.name
    if not file_name.endswith(RESULT_SUFFIX):
        raise ValueError(f"Not a result JSON file: {path_file}")

    stem = file_name[: -len(RESULT_SUFFIX)]
    prefix = f"{task_name}-"
    rest = stem[len(prefix) :] if stem.startswith(prefix) else stem

    prompt_mode = None
    decoding_strategy = None
    seed = None
    for marker in DECODING_MARKERS:
        split_token = f"-{marker}-"
        if split_token in rest:
            prompt_mode, seed = rest.rsplit(split_token, 1)
            decoding_strategy = marker
            break

    return {
        "task": task_name,
        "model": model_name,
        "prompt_mode": prompt_mode,
        "decoding_strategy": decoding_strategy,
        "seed": seed,
        "path": str(path_file),
    }


def find_result_files(
    dir_result: str | Path | Iterable[str | Path] = "result",
    task_names: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    prompt_modes: Iterable[str] | None = None,
) -> list[Path]:
    """Find result files across one or more result roots."""
    task_filter = set(task_names or [])
    model_filter = set(model_names or [])
    prompt_filter = set(prompt_modes or [])

    matched = []
    for dir_result_one in parse_path_list(dir_result, "dir_result"):
        for path_file in Path(dir_result_one).glob(f"*/*/*{RESULT_SUFFIX}"):
            info = parse_result_file(path_file)
            if task_filter and info["task"] not in task_filter:
                continue
            if model_filter and info["model"] not in model_filter:
                continue
            if prompt_filter and info["prompt_mode"] not in prompt_filter:
                continue
            matched.append(path_file)
    return sorted(matched)


def load_result_records(path_file: str | Path) -> list[dict]:
    """Load one result JSON file."""
    data = read_json(path_file)
    if not isinstance(data, list):
        raise ValueError(f"Result file must contain a list: {path_file}")
    return data


def collect_result_inventory(
    dir_result: str | Path | Iterable[str | Path] = "result",
    task_names: Iterable[str] | None = None,
    model_names: Iterable[str] | None = None,
    prompt_modes: Iterable[str] | None = None,
    count_rows: bool = False,
) -> list[dict]:
    """Return a lightweight inventory of result files for analysis notebooks."""
    records = []
    for path_file in find_result_files(
        dir_result=dir_result,
        task_names=task_names,
        model_names=model_names,
        prompt_modes=prompt_modes,
    ):
        info = parse_result_file(path_file)
        if count_rows:
            info["rows"] = len(load_result_records(path_file))
        records.append(info)
    return records
