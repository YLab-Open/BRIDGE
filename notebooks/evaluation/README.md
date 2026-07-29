# Evaluation Notebooks

This folder contains compact notebooks that demonstrate how to reuse the
maintained evaluation API.

## Files

- `evaluate_classification.ipynb`: loads `evaluation/evaluate.BRIDGE.yaml`,
  restricts evaluation to classification tasks, and shows result inventory.
- `evaluate_extraction.ipynb`: same pattern for extraction tasks.
- `evaluate_generation.ipynb`: same pattern for generation tasks. Generation
  metrics require optional dependencies such as `bert-score` and `evaluate`.

## Pattern

Each notebook imports:

```python
from evaluation.config import config_from_yaml
from evaluation.runner import evaluate_all
from evaluation.result_utils import collect_result_inventory
```

The notebooks keep actual evaluation behind a `RUN_EVALUATION` flag so they can
be opened safely on machines without complete result folders.

Each notebook locates the repository root by walking upward until `main.py` is
found, so it can be run from the repository root or from this notebook folder.

`tests/test_evaluation_notebooks.py` executes these notebooks in dry-run mode,
so imports and example cells are covered by the lightweight test suite.
