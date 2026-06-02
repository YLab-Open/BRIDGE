import json
from pathlib import Path


def test_evaluation_notebook_dry_run_cells_execute(monkeypatch):
    """Execute maintained evaluation notebooks with RUN_EVALUATION disabled."""
    root = Path(__file__).resolve().parents[1]
    path_notebooks = root / "notebooks" / "evaluation"
    notebooks = sorted(path_notebooks.glob("*.ipynb"))
    assert notebooks

    monkeypatch.chdir(path_notebooks)
    for nb_path in notebooks:
        namespace = {"__name__": "__notebook_test__"}
        nb = json.loads(nb_path.read_text(encoding="utf-8"))
        for cell in nb["cells"]:
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, str(nb_path), "exec"), namespace)
