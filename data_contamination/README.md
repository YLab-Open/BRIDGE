# Data Contamination Utilities

This folder contains utilities for n-gram based data-contamination checks.

## Files

- `utils.py`: current helper functions for text normalization, model loading,
  dataset loading, and n-gram overlap evaluation with HF or vLLM backends.
- `full/`: legacy/full experiment scripts and generated artifacts. This folder
  is ignored by git and should be treated as historical or local experiment
  material.

## Typical Use

Create `configs/dict_model_path.json` from the public template, then customize
`BRIDGE_data_contamination.yaml`. Validate the launchers without loading a model:

```bash
bash data_contamination/run_data_contamination.sh --dry-run
bash data_contamination/run_data_contamination_mul_model.sh --dry-run
```

Remove `--dry-run` to start the single- or multi-model job. Override the default
GPU list, model, or config without editing the launchers:

```bash
CONTAMINATION_GPUS=0 \
CONTAMINATION_MODEL_NAME=Llama-3.1-8B-Instruct \
bash data_contamination/run_data_contamination.sh
```

These utilities are designed for offline analysis rather than the main BRIDGE
inference/evaluation path. Keep new reusable contamination helpers in `utils.py`;
put one-off local outputs under ignored result/log folders.

## Maintenance Notes

Do not import these helpers from the main inference path unless the dependency
cost is intentional. Contamination checks often require model loading and large
local data files, so they should remain opt-in analysis utilities.
