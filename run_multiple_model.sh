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

# -------- Model Names --------
models=(
    # "gemma-4-26B-A4B-it"
    # "gemma-4-31B-it"
    # "Qwen3-30B-A3B-Thinking-2507"
    # "Qwen3-30B-A3B-Instruct-2507"
    # "Llama-3.3-70B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "medgemma-27b-text-it"
    # "gemma-3-12b-it"
    # "Baichuan-M1-14B-Instruct"
    # "DeepSeek-R1"
    # "DeepSeek-R1-Distill-Llama-8B"
    # "DeepSeek-R1-Distill-Llama-70B"
    # "DeepSeek-R1-Distill-Qwen-1.5B"
    # "DeepSeek-R1-Distill-Qwen-7B"
    # "DeepSeek-R1-Distill-Qwen-14B"
    # "DeepSeek-R1-Distill-Qwen-32B"
    # "gemma-2-9b-it"
    # "gemma-2-27b-it"
    # "gemma-3-1b-it"
    # "gemma-3-4b-it"
    # "gemma-3-12b-it"
    # "gemma-3-27b-it"
    "Llama-3.1-8B-Instruct"
    "Llama-3.1-70B-Instruct"
    # "Llama-3.2-1B-Instruct"
    # "Llama-3.2-3B-Instruct"
    # "Llama-3.3-70B-Instruct"
    # "Llama-4-Scout-17B-16E-Instruct"
    # "Llama-3.1-Nemotron-70B-Instruct-HF"
    # "meditron-7b"
    # "meditron-70b"
    # "MeLLaMA-13B-chat"
    # "MeLLaMA-70B-chat"
    # "Llama3-OpenBioLLM-8B"
    # "Llama3-OpenBioLLM-70B"
    # "MMed-Llama-3-8B"
    # "Llama-3.1-8B-UltraMedical"
    # "Llama-3-70B-UltraMedical"
    # "Ministral-8B-Instruct-2410"
    # "Mistral-Small-Instruct-2409"
    # "Mistral-Small-24B-Instruct-2501"
    # "Mistral-Small-3.1-24B-Instruct-2503"
    # "Mistral-Large-Instruct-2411"
    # "BioMistral-7B"
    # "Phi-3.5-mini-instruct"
    # "Phi-3.5-MoE-instruct"
    # "Phi-4"
    # "Qwen2.5-1.5B-Instruct"
    # "Qwen2.5-3B-Instruct"
    # "Qwen2.5-7B-Instruct"
    # "Qwen2.5-72B-Instruct"
    # "QwQ-32B-Preview"
    # "QWQ-32B"
    # "Athene-V2-Chat"
    # "Yi-1.5-9B-Chat-16K"
    # "Yi-1.5-34B-Chat-16K"
    "gpt-oss-20b"
    # "gpt-oss-120b"
    # "Qwen2.5-72B-Instruct"
    # "Llama-3.1-8B-Instruct"
    # "Llama-3.1-70B-Instruct"
    "medgemma-27b-it"
    "Baichuan-M2-32B"
    # "HuatuoGPT-o1-72B"
    # "Qwen3-Next-80B-A3B-Thinking"
    # "Qwen3-Next-80B-A3B-Instruct"
    # "Qwen3-235B-A22B-Thinking-2507"
    # "Qwen3-235B-A22B-Instruct-2507"
    "Qwen3-30B-A3B-Thinking-2507"
    "Qwen3-30B-A3B-Instruct-2507"
    "gemma-3-12b-it"
)

# -------- GPU VISIBILITY --------
gpus=4,5,6,7
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

# nohup bash run_multiple_model.sh > log/run_multiple_model.log 2>&1 &
# PID: 547851

# ps -u $USER -o pid,ppid,stat,etime,user,cmd --sort=etime
