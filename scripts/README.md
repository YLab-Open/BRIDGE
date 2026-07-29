# Run Scripts

This folder contains delayed shell launchers for local inference jobs.
The primary launchers are kept at the repository root for convenience.

## Files

- `run_timing.sh`: submit `run.sh` after a delay.
- `run_multiple_model_timing.sh`: submit `run_multiple_model.sh` after a delay.

Run scripts from the repository root:

```bash
bash run.sh
bash run_multiple_model.sh
bash scripts/run_timing.sh 45
bash scripts/run_multiple_model_timing.sh 45
```

Edit `run.sh` and `run_multiple_model.sh` for model names and GPU IDs. Config
paths can also be overridden with `BRIDGE_CONFIG_FILE` and
`BRIDGE_MODEL_FILE`.

Use `--dry-run` to validate paths and print commands without launching
inference. In a fresh clone, dry-run automatically uses the committed example
config and model mapping when the ignored local copies do not exist:

```bash
bash run.sh --dry-run
bash run_multiple_model.sh --dry-run
```
