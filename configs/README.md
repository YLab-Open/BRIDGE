# Runtime Configs

This folder contains runtime configuration templates and local-only config
locations.

## Files

- `BRIDGE.yaml`: default local inference config for `main.py`.
- `API.key.example.yaml`: public provider credential template.
- `dict_model_path.example.json`: public local model-path mapping template.

Create local files from the templates:

```bash
cp configs/API.key.example.yaml configs/API.key.yaml
cp configs/dict_model_path.example.json configs/dict_model_path.json
```

`configs/API.key.yaml` and `configs/dict_model_path.json` are ignored by git.
For compatibility, `util.local_config` can still read old root-level
`API.key.yaml` and `dict_model_path.json` files when the new `configs/` files do
not exist.
