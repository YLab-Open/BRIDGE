# Metric Layer

This folder contains task-type metric implementations used by BRIDGE evaluation.

## Files

- `classification.py`: accuracy, precision, recall, F1, ordered-label metrics,
  and multi-label metrics.
- `extraction.py`: entity/event extraction metrics, subject-field matching, and
  QA-style extraction metrics.
- `generation.py`: BLEU, ROUGE, METEOR, BERTScore, and language-specific
  tokenization helpers.

## Usage

Task adapters in `dataset/classification.py`, `dataset/extraction.py`, and
`dataset/generation.py` call these metric functions after converting raw model
outputs into normalized predictions.

Example:

```python
from metric.classification import calc_metrics_clf

scores = calc_metrics_clf(list_pred=[1, 0, 1], list_label=[1, 1, 1], average="macro")
```

Generation metrics depend on optional packages such as `bert-score` and
`evaluate`; classification and extraction metrics use lighter dependencies.
