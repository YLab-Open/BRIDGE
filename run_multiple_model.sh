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
CONFIG_FILE="${REPO_ROOT}/configs/BRIDGE.yaml"
MODEL_FILE="${REPO_ROOT}/configs/dict_model_path.json"
MODEL_FILE_EXAMPLE="${REPO_ROOT}/configs/dict_model_path.example.json"

# -------- Model Names --------
models=(
    "Llama-3.1-8B-Instruct"
    "Llama-3.1-70B-Instruct"
)

# -------- GPU VISIBILITY --------
gpus=0,1,2,3
export CUDA_VISIBLE_DEVICES=$gpus

# -------- Log output --------
dir_log="${REPO_ROOT}/log"
mkdir -p "$dir_log"

# -------- Sanity checks --------
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Config file not found: $CONFIG_FILE"
    exit 1
fi
if [[ ! -f "$MODEL_FILE" && "$dry_run" == true && -f "$MODEL_FILE_EXAMPLE" ]]; then
    MODEL_FILE="$MODEL_FILE_EXAMPLE"
elif [[ ! -f "$MODEL_FILE" ]]; then
    echo "Model mapping file not found: $MODEL_FILE"
    echo "Create it from configs/dict_model_path.example.json before running inference."
    exit 1
fi

echo "========================================"
echo "GPUs: ${gpus}"
echo "Config file: ${CONFIG_FILE}"
echo "Model file: ${MODEL_FILE}"
echo "Log dir: ${dir_log}"
echo "Dry run: ${dry_run}"
echo "========================================"

# -------- Run --------
total_models=${#models[@]}

echo "========================================"
echo "Total models to run: ${total_models}"
echo "Models:"
echo "========================================"

i=0
for model_name in "${models[@]}"; do
    i=$((i + 1))
    echo "[$i/${total_models}] $model_name"
done

echo "========================================"
echo "Start running models"
echo "========================================"

i=0
cd "$REPO_ROOT"
for model_name in "${models[@]}"; do
    i=$((i + 1))

    now=$(date +"%m-%d_%H-%M-%S")
    log_file="$dir_log/${model_name}.${now}.log"

    echo "----------------------------------------"
    echo "Job started: $model_name"
    echo "Progress: $i/${total_models}"
    echo "Start time: $now"
    echo "Log file: $log_file"
    echo "----------------------------------------"

    cmd=(
        python main.py
        --model_name "$model_name"
        --gpus "$gpus"
        --config "$CONFIG_FILE"
        --model_file "$MODEL_FILE"
    )

    if [[ "$dry_run" == true ]]; then
        printf 'Dry run command:'
        printf ' %q' "${cmd[@]}"
        echo
        continue
    fi

    nohup "${cmd[@]}" > "$log_file" 2>&1

    end_time=$(date +"%m-%d_%H-%M-%S")
    echo "Job finished: $model_name"
    echo "Progress: $i/${total_models}"
    echo "End time: $end_time"
    echo "Log file: $log_file"
done

echo "========================================"
echo "All jobs finished."
echo "========================================"
