import json
from pathlib import Path


def test_public_inference_notebook_sources_compile():
    """Keep public inference notebooks portable and free of saved run output."""
    root = Path(__file__).resolve().parents[1]
    notebook_dir = root / "notebooks" / "inference"
    notebook_names = [
        "Inference task.Local.ipynb",
        "Inference task.Local.probability.ipynb",
        "Inference task.Azure.ipynb",
    ]

    for notebook_name in notebook_names:
        notebook_path = notebook_dir / notebook_name
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        for cell_index, cell in enumerate(notebook["cells"]):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            compile(source, f"{notebook_path}:cell-{cell_index}", "exec")
            assert "/" + "PHShome/" not in source
            assert "/" + "netapp" not in source
            assert cell.get("execution_count") is None
            assert cell.get("outputs", []) == []
