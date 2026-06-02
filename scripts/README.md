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

Edit `run.sh` and `run_multiple_model.sh` directly for model names, GPU IDs,
config paths, and log locations.

Use `--dry-run` to validate paths and print the command without launching
inference:

```bash
bash run.sh --dry-run
bash run_multiple_model.sh --dry-run
```
