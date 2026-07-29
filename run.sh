#!/usr/bin/env bash
set -euo pipefail
export TOKENIZERS_PARALLELISM=False

dry_run=false
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run=true
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

# -------- Path to YAML config --------
CONFIG_FILE="${BRIDGE_CONFIG_FILE:-${REPO_ROOT}/configs/BRIDGE.yaml}"
MODEL_FILE="${BRIDGE_MODEL_FILE:-${REPO_ROOT}/configs/dict_model_path.json}"

# -------- Model Name --------
model_name=gemma-3-12b-it

# -------- GPU VISIBILITY --------
gpus=0,1,2,3
# gpus=4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpus

# -------- Log output --------
dir_log="${REPO_ROOT}/log"

if [[ "$dry_run" == true ]]; then
    if [[ ! -f "$CONFIG_FILE" ]]; then
        CONFIG_FILE="${REPO_ROOT}/configs/BRIDGE.example.yaml"
    fi
    if [[ ! -f "$MODEL_FILE" ]]; then
        MODEL_FILE="${REPO_ROOT}/configs/dict_model_path.example.json"
    fi
else
    mkdir -p "$dir_log"
fi

# -------- Sanity checks --------
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi
if [[ ! -f "$MODEL_FILE" ]]; then
    echo "Model mapping file not found: $MODEL_FILE"
    echo "Create it from configs/dict_model_path.example.json before running inference."
    exit 1
fi

# -------- Run --------
now=$(date +"%m-%d_%H-%M-%S")
log_file="$dir_log/${model_name}.${now}.log"
cd "$REPO_ROOT"

cmd=(
    python main.py
    --model_name "$model_name"
    --gpus "$gpus"
    --config "$CONFIG_FILE"
    --model_file "$MODEL_FILE"
)

if [[ "$dry_run" == true ]]; then
    echo "Dry run. Command:"
    printf ' %q' "${cmd[@]}"
    echo
    echo "Log file: $log_file"
    exit 0
fi

nohup "${cmd[@]}" > "$log_file" 2>&1 &

# -------- Log output --------
echo "Job started, log in $log_file"
