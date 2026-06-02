#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/BRIDGE_data_contamination.yaml"
LOG_DIR="${SCRIPT_DIR}/log"

models=(
    "Llama-3.1-8B-Instruct"
    "Llama-3.1-70B-Instruct"
)

gpus=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpus

mkdir -p "$LOG_DIR"
cd "$SCRIPT_DIR"

total_models=${#models[@]}
i=0
for model_name in "${models[@]}"; do
    i=$((i + 1))
    now=$(date +"%m-%d_%H-%M-%S")
    log_file="$LOG_DIR/${model_name}.${now}.log"

    cmd=(
        python data_contamination.py
        --model_name "$model_name"
        --config "$CONFIG_FILE"
    )

    echo "[$i/${total_models}] $model_name"
    if [[ "$dry_run" == true ]]; then
        printf 'Dry run command:'
        printf ' %q' "${cmd[@]}"
        echo
        echo "Log file: $log_file"
        continue
    fi

    nohup "${cmd[@]}" > "$log_file" 2>&1
done
