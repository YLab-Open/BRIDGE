models=(
    "Qwen2.5-72B-Instruct"
    "Llama-3.1-70B-Instruct"
)

# -------- GPU VISIBILITY --------
gpus=4,5,6,7
# 0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpus

# -------- Path to YAML config --------
CONFIG_FILE="BRIDGE_data_contamination.yaml"

# -------- RUN --------
for model_name in "${models[@]}"; do
    # -------- Log output --------
    echo "Running $model_name at $now ($((++i))/${#models[@]})"
    # -------- Get time --------
    now=$(date +"%m-%d_%H-%M")
    nohup python data_contamination.py \
    --model_name $model_name \
    --config "$CONFIG_FILE" \
    > log/${model_name}.${now}.log 2>&1
done