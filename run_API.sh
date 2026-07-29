#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=False
export PYTHONUNBUFFERED=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

usage() {
    cat <<'USAGE'
Usage:
  bash run_API.sh [options]

Options:
  --provider azure|claude|gemini   Provider override. Default: azure
  --config PATH                    Run YAML. Default follows provider.
  --model NAME                     Run one model instead of YAML models.
  --api-model-config PATH          Model registry YAML. Default: configs/API.model.yaml
  --api-key-file PATH              Credential YAML. Default: configs/API.key.yaml
  --evaluate                       Force evaluation after inference.
  --no-evaluate                    Skip evaluation even if YAML enables it.
  --allow-partial                  Keep partial outputs when sample API calls fail.
  --fail-on-error                  Fail when any sample API call fails. Default unless YAML says otherwise.
  --dry-run                        Validate config and print planned result paths without writing result/trace files.
  --foreground                     Run in foreground instead of nohup background.
  -h, --help                       Show this help.

Recommended workflow:
  Put tasks, models, paths, and provider parameters in the YAML file.
  Use this script only to choose the YAML, run mode, and short-lived overrides.
USAGE
}

provider="${API_PROVIDER:-azure}"
config_file=""
model_name="${API_MODEL_NAME:-}"
api_model_config="${REPO_ROOT}/configs/API.model.yaml"
api_key_file="${REPO_ROOT}/configs/API.key.yaml"
dry_run=false
foreground=false
force_evaluate=false
skip_evaluate=false
allow_partial=false
fail_on_error=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --provider)
            provider="${2:?Missing value for --provider}"
            shift 2
            ;;
        --config)
            config_file="${2:?Missing value for --config}"
            shift 2
            ;;
        --model|--model_name|--model-name)
            model_name="${2:?Missing value for --model}"
            shift 2
            ;;
        --api-model-config|--api_model_config)
            api_model_config="${2:?Missing value for --api-model-config}"
            shift 2
            ;;
        --api-key-file|--api_key_file)
            api_key_file="${2:?Missing value for --api-key-file}"
            shift 2
            ;;
        --evaluate)
            force_evaluate=true
            shift
            ;;
        --no-evaluate)
            skip_evaluate=true
            shift
            ;;
        --allow-partial)
            allow_partial=true
            shift
            ;;
        --fail-on-error)
            fail_on_error=true
            shift
            ;;
        --dry-run)
            dry_run=true
            foreground=true
            shift
            ;;
        --foreground)
            foreground=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage >&2
            exit 1
            ;;
    esac
done

case "$provider" in
    azure|claude|gemini) ;;
    *)
        echo "Invalid provider: $provider" >&2
        echo "Valid values: azure, claude, gemini" >&2
        exit 1
        ;;
esac

if [[ -z "$config_file" ]]; then
    case "$provider" in
        azure) config_file="${REPO_ROOT}/configs/API.provider.azure.yaml" ;;
        claude) config_file="${REPO_ROOT}/configs/API.provider.claude.yaml" ;;
        gemini) config_file="${REPO_ROOT}/configs/API.provider.gemini.yaml" ;;
    esac
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi
if [[ ! -f "$api_model_config" ]]; then
    echo "API model config not found: $api_model_config" >&2
    exit 1
fi
if [[ ! -f "$api_key_file" ]]; then
    echo "Optional API key file not found: $api_key_file"
    echo "The runner will try provider environment variables instead."
fi

dir_log="${REPO_ROOT}/log_API"
if [[ "$dry_run" == false ]]; then
    mkdir -p "$dir_log"
fi

now="$(date +"%m-%d_%H-%M-%S")"
log_model_label="${model_name:-yaml-models}"
log_file="${dir_log}/${provider}.${log_model_label}.${now}.log"

cd "$REPO_ROOT"

cmd=(
    python -u main_api.py
    --provider "$provider"
    --config "$config_file"
    --api_model_config "$api_model_config"
    --api_key_file "$api_key_file"
)

if [[ -n "$model_name" ]]; then
    cmd+=(--model "$model_name")
fi
if [[ "$force_evaluate" == true ]]; then
    cmd+=(--evaluate)
fi
if [[ "$skip_evaluate" == true ]]; then
    cmd+=(--no-evaluate)
fi
if [[ "$allow_partial" == true ]]; then
    cmd+=(--allow-partial)
fi
if [[ "$fail_on_error" == true ]]; then
    cmd+=(--fail-on-error)
fi
if [[ "$dry_run" == true ]]; then
    cmd+=(--dry-run)
fi

echo "Command:"
printf ' %q' "${cmd[@]}"
echo

if [[ "$foreground" == true ]]; then
    echo "Running in foreground."
    "${cmd[@]}"
    exit 0
fi

nohup "${cmd[@]}" > "$log_file" 2>&1 &
pid=$!

echo "API job started."
echo "Provider: $provider"
if [[ -n "$model_name" ]]; then
    echo "Model: $model_name"
else
    echo "Models: from $config_file"
fi
echo "PID: $pid"
echo "Log: $log_file"
