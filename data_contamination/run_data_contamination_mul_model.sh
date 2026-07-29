#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

models=(
    "Llama-3.1-8B-Instruct"
    "Qwen3-30B-A3B-Instruct-2507"
    "gpt-oss-20b"
)

gpus="${CONTAMINATION_GPUS:-0,1,2,3}"
config_file="${CONTAMINATION_CONFIG_FILE:-${SCRIPT_DIR}/BRIDGE_data_contamination.yaml}"
model_file="${BRIDGE_MODEL_FILE:-${REPO_ROOT}/configs/dict_model_path.json}"
export CUDA_VISIBLE_DEVICES="$gpus"

if [[ "$dry_run" == true && ! -f "$model_file" ]]; then
    model_file="${REPO_ROOT}/configs/dict_model_path.example.json"
fi

if [[ ! -f "$config_file" ]]; then
    echo "Config file not found: $config_file" >&2
    exit 1
fi
if [[ ! -f "$model_file" ]]; then
    echo "Model mapping file not found: $model_file" >&2
    echo "Create it from configs/dict_model_path.example.json before running inference." >&2
    exit 1
fi

if [[ "$dry_run" == false ]]; then
    mkdir -p "${SCRIPT_DIR}/log"
fi

cd "$SCRIPT_DIR"
total_models=${#models[@]}
i=0
for model_name in "${models[@]}"; do
    i=$((i + 1))
    now="$(date +"%m-%d_%H-%M")"
    log_file="${SCRIPT_DIR}/log/${model_name}.${now}.log"
    cmd=(
        python data_contamination.py
        --model_name "$model_name"
        --config "$config_file"
    )

    echo "[$i/$total_models] $model_name"
    if [[ "$dry_run" == true ]]; then
        printf 'Dry run command:'
        printf ' %q' "${cmd[@]}"
        echo
        continue
    fi

    nohup "${cmd[@]}" > "$log_file" 2>&1
    echo "Finished $model_name; log: $log_file"
done
