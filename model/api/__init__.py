"""Commercial API inference internals.

Public callers should keep importing `run_api_from_config` from
`model.api_runner`. The submodules here keep configuration, prompt preparation,
provider calls, and result data structures separated for maintainability.

Provider implementations live in `model.api.provider`:

- `azure.py`: Azure OpenAI single-query and Batch API.
- `claude.py`: Anthropic Messages API.
- `gemini.py`: Google Gemini generate_content API.
- `factory.py`: provider-name dispatch used by `model.api_runner`.
"""
