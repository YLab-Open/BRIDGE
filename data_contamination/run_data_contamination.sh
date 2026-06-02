#!/usr/bin/env bash
set -euo pipefail

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/BRIDGE_data_contamination.yaml"
LOG_DIR="${SCRIPT_DIR}/log"

model_name=Llama-3.1-8B-Instruct
gpus=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpus

mkdir -p "$LOG_DIR"

cmd=(
    python data_contamination.py
    --model_name "$model_name"
    --config "$CONFIG_FILE"
)

now=$(date +"%m-%d_%H-%M-%S")
log_file="$LOG_DIR/${model_name}.${now}.log"

cd "$SCRIPT_DIR"

if [[ "$dry_run" == true ]]; then
    echo "Dry run. Command:"
    printf ' %q' "${cmd[@]}"
    echo
    echo "Log file: $log_file"
    exit 0
fi

nohup "${cmd[@]}" > "$log_file" 2>&1 &
echo "Job started, log in $log_file"
