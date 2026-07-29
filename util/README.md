# Utility Layer

This folder contains shared runtime, local-config, I/O, logging, statistics, and
text preprocessing helpers.

## Files

- `runtime.py`: YAML loading with duplicate-key detection, CLI/YAML merge,
  runtime validation, prompt-mode parsing, GPU parsing, dataloader selection, and
  task logging.
- `local_config.py`: loaders for local-only files such as `configs/API.key.yaml` and
  `configs/dict_model_path.json`.
- `io.py`: JSON/JSONL, iterable JSONL, text-file, and simple dictionary file helpers.
- `logging.py`: logger setup and the legacy terminal `ProgressBar`.
- `stats.py`: confidence interval/statistical helpers.
- `text.py`: text/list cleanup helpers.
- `misc.py`: small argument, memory-size, and dictionary helpers.
- `seed.py`: shared seed setup for local inference and evaluation bookkeeping.
- `preprocess.py`: punctuation normalization and simple EHR-note cleanup.

## Local-Only Configs

Real credential and path files are ignored by git:

```bash
cp configs/API.key.example.yaml configs/API.key.yaml
cp configs/dict_model_path.example.json configs/dict_model_path.json
```

Use `util.local_config.load_api_key_config()` for API credentials and
`util.local_config.resolve_model_path()` for local model paths. These helpers
validate structure without printing secrets.

## Runtime Validation

`util.runtime.parse_prompt_mode()` accepts only:

- `direct`
- `cot`
- `direct-N-shot`
- `cot-N-shot`

The YAML loader rejects duplicate keys, and `validate_args()` catches missing
required runtime fields before model loading starts.
