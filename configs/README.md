# Runtime Configs

This folder contains runtime configuration templates and local-only config
locations.

## Files

- `BRIDGE.example.yaml`: public local inference template for `main.py`.
- `BRIDGE.yaml`: ignored machine-local copy used by default by `main.py`.
- `logits.example.yaml`: optional local vLLM logits-capture config used through
  `main.py --logits_config`.
- `README.logits.md`: logits YAML reference.
- `README.result.md`: ordinary result JSON format.
- `README.result.logits.md`: logits result sidecar format.
- `API_USAGE.md`: commercial API run configuration and trace/evaluation workflow.
- `API.provider.azure.yaml`, `API.provider.claude.yaml`, `API.provider.gemini.yaml`: provider API run configs.
- `API.model.yaml`: commercial API model/deployment capability registry.
- `API.key.example.yaml`: public provider credential template.
- `dict_model_path.example.json`: public local model-path mapping template.

Create local files from the templates:

```bash
cp configs/BRIDGE.example.yaml configs/BRIDGE.yaml
cp configs/API.key.example.yaml configs/API.key.yaml
cp configs/dict_model_path.example.json configs/dict_model_path.json
```

`configs/BRIDGE.yaml`, `configs/API.key.yaml`, and
`configs/dict_model_path.json` are ignored by git.
For compatibility, `util.local_config` can still read old root-level
`API.key.yaml` and `dict_model_path.json` files when the new `configs/` files do
not exist.

For API runs, keep tasks, models, paths, and provider parameters in YAML. Use
`run_API.sh` only as a launcher:

```bash
bash run_API.sh --provider azure --config configs/API.provider.azure.yaml --model gpt-4o --foreground
```

For logits-enabled local vLLM runs, keep regular task/model/runtime settings in
`BRIDGE.yaml` or a project-specific BRIDGE YAML, and put logits-specific options
and task label groups in a separate file:

```bash
python main.py \
  --model_name MyLocalModel \
  --config configs/BRIDGE.yaml \
  --logits_config configs/logits.example.yaml
```

This keeps normal inference configuration reusable while allowing one logits YAML
to define probability targets for multiple tasks.

For long vLLM runs, the local YAML can enable append-only checkpoints:

```yaml
save_every_n_batches: 20
resume_partial_results: false
```

Set `resume_partial_results: true` only when resuming an interrupted run with
the same task/model/experiment settings. Completed local and API experiments
always write `*.result.config.yaml` and `*.result.metadata.json` next to the
standard result file; API snapshots are redacted.

See `README.result.md` and `README.result.logits.md` in this folder for the
exported result file formats.
