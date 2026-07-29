# Inference Notebooks

This folder contains interactive inference examples.

## Files

- `Inference task.Local.ipynb`: local vLLM debugging workflow.
- `Inference task.Local.probability.ipynb`: local vLLM workflow with generated
  Yes/No token logits and logit-margin inspection.
- `Inference task.Azure.ipynb`: Azure OpenAI workflow through the same unified
  API runner used by `main_api.py` and `run_API.sh`.

All notebooks locate the repository root as `PROJECT_ROOT` and
read local-only files from `configs/` by default.
