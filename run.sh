#!/usr/bin/env bash
# export TOKENIZERS_PARALLELISM=False

# -------- Path to YAML config --------
CONFIG_FILE="BRIDGE.yaml"

# -------- GPU VISIBILITY --------
gpus=0,1,2,3
# 0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpus

# -------- Run --------
now=$(date +"%m-%d_%H-%M")
nohup python main.py \
    --model_name "$model_name" \
    --gpus "$gpus" \
    --config "$CONFIG_FILE" \
    > log/${model_name}.${now}.log 2>&1 &

# -------- Log output --------
echo "Job started, log in log/${model_name}.${now}.log"