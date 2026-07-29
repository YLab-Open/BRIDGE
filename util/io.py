from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def load_json(path_file: str | Path) -> Any:
    """Load and return parsed JSON content."""
    with Path(path_file).open("r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path_file: str | Path) -> List[Dict[str, Any]]:
    """Load a JSONL file and return non-empty lines as dictionaries."""
    with Path(path_file).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(data: Any, path_file: str | Path, indent: int = 2):
    """Write JSON data with parent-directory creation."""
    path_file = Path(path_file)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def write_jsonl(data: List[Dict[str, Any]], path_file: str | Path):
    """Write a list of dictionaries to a JSONL file."""
    path_file = Path(path_file)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def write_jsonl_iter(rows: Iterable[Dict[str, Any]], path_file: str | Path):
    """Write dictionaries from any iterable to a JSONL file."""
    path_file = Path(path_file)
    path_file.parent.mkdir(parents=True, exist_ok=True)
    with path_file.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dict_to_json(dict_saved, path_saved):
    """Write a dictionary to JSON and print the saved path."""
    write_json(dict_saved, path_saved, indent=2)
    print(f"Save to {path_saved}")


def json_to_dict(path_saved):
    """Load a JSON object and print the loaded path."""
    dict_saved = load_json(path_saved)
    print(f"Loaded {path_saved}")
    return dict_saved


def list_to_txt(list_saved, path_saved):
    """Write one list item per line."""
    path_saved = Path(path_saved)
    path_saved.parent.mkdir(parents=True, exist_ok=True)
    with path_saved.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(list_saved):
            if idx != (len(list_saved) - 1):
                f.write(str(item) + "\n")
            else:
                f.write(str(item))
    print(f"Save to {path_saved}")


def list_to_txt_nested(list_nested, path_saved, flag_print=True):
    """Write a nested list as tab-separated rows."""
    path_saved = Path(path_saved)
    path_saved.parent.mkdir(parents=True, exist_ok=True)
    with path_saved.open("w", encoding="utf-8") as f:
        for idx, list_flat in enumerate(list_nested):
            for item in list_flat:
                f.write(str(item) + "\t")
            if idx != (len(list_nested) - 1):
                f.write("\n")
    if flag_print:
        print(f"Save to {path_saved}")


def txt_to_list(path_saved, flag_print=True):
    """Read non-empty text lines into a list."""
    list_loaded = []
    with Path(path_saved).open("r", encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()):
                list_loaded.append(line.strip())
    if flag_print:
        print(f"Loaded {path_saved}")
    return list_loaded


def txt_to_list_nested(path_saved, token_split=" "):
    """Read text lines into a nested list split by token_split."""
    with Path(path_saved).open("r", encoding="utf-8") as f:
        list_loaded = []
        for line in f.readlines():
            line = line.strip()
            list_line = []
            if len(line):
                for item in line.split(token_split):
                    list_line.append(item)
            list_loaded.append(list_line)
    print(f"Loaded {path_saved}")
    return list_loaded


def load_dict(dict_path):
    """Load an integer-key dictionary from a tab-separated text file."""
    vocab = {}
    with Path(dict_path).open("r", encoding="utf-8") as f:
        for line in f:
            key, value = line.strip("\n").split("\t")
            vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    """Write an integer-key dictionary to a tab-separated text file."""
    dict_path = Path(dict_path)
    dict_path.parent.mkdir(parents=True, exist_ok=True)
    with dict_path.open("w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))


def get_keywords(path_keyword):
    """
    Load keyword patterns from a file.

    Lines starting with # are treated as bare keywords and wrapped with a simple
    boundary pattern to preserve the previous keyword-file behavior.
    """
    list_keywords = []
    prefix_suffix = "( |^|$)"
    with Path(path_keyword).open(encoding="utf-8") as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            if line.startswith("#"):
                line = line.strip().split("#")[-1] + prefix_suffix
                print(line)
                list_keywords.append(line)
            else:
                list_keywords.append(line.strip())
    print(f"Got {len(list_keywords)} keywords")
    return list_keywords
