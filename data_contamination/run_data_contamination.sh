#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

model_name="${CONTAMINATION_MODEL_NAME:-Llama-3.1-8B-Instruct}"
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

cmd=(
    python data_contamination.py
    --model_name "$model_name"
    --config "$config_file"
)

if [[ "$dry_run" == true ]]; then
    echo "Dry run. Command:"
    printf ' %q' "${cmd[@]}"
    echo
    echo "GPUs: $gpus"
    echo "Model mapping: $model_file"
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/log"
now="$(date +"%m-%d_%H-%M")"
log_file="${SCRIPT_DIR}/log/${model_name}.${now}.log"

cd "$SCRIPT_DIR"
nohup "${cmd[@]}" > "$log_file" 2>&1 &
echo "Job started, log in $log_file"
