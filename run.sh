#!/usr/bin/env bash
export TOKENIZERS_PARALLELISM=False

# -------- Path to YAML config --------
CONFIG_FILE="BRIDGE.yaml"

# -------- Model Name --------
model_name=gpt-oss-20b
# Supported models:
# Baichuan-M1-14B-Instruct
# DeepSeek-R1
# DeepSeek-R1-Distill-Llama-8B
# DeepSeek-R1-Distill-Llama-70B
# DeepSeek-R1-Distill-Qwen-1.5B
# DeepSeek-R1-Distill-Qwen-7B
# DeepSeek-R1-Distill-Qwen-14B
# DeepSeek-R1-Distill-Qwen-32B
# gemma-2-9b-it
# gemma-2-27b-it
# gemma-3-1b-it
# gemma-3-4b-it
# gemma-3-12b-it
# gemma-3-27b-it
# Llama-3.1-8B-Instruct
# Llama-3.1-70B-Instruct
# Llama-3.2-1B-Instruct
# Llama-3.2-3B-Instruct
# Llama-3.3-70B-Instruct
# Llama-4-Scout-17B-16E-Instruct
# Llama-3.1-Nemotron-70B-Instruct-HF
# meditron-7b
# meditron-70b
# MeLLaMA-13B-chat
# MeLLaMA-70B-chat
# Llama3-OpenBioLLM-8B
# Llama3-OpenBioLLM-70B
# MMed-Llama-3-8B
# Llama-3.1-8B-UltraMedical
# Llama-3-70B-UltraMedical
# Ministral-8B-Instruct-2410
# Mistral-Small-Instruct-2409
# Mistral-Small-24B-Instruct-2501
# Mistral-Small-3.1-24B-Instruct-2503
# Mistral-Large-Instruct-2411
# BioMistral-7B
# Phi-3.5-mini-instruct
# Phi-3.5-MoE-instruct
# Phi-4
# Qwen2.5-1.5B-Instruct
# Qwen2.5-3B-Instruct
# Qwen2.5-7B-Instruct
# Qwen2.5-72B-Instruct
# Qwen2.5-14B-Instruct
# Qwen3-0.6B
# Qwen3-1.7B
# Qwen3-4B
# Qwen3-8B
# Qwen3-14B
# Qwen3-32B
# Qwen3-30B-A3B
# Qwen3-235B-A22B
# QwQ-32B-Preview
# QWQ-32B
# Athene-V2-Chat
# Yi-1.5-9B-Chat-16K
# Yi-1.5-34B-Chat-16K
# gpt-oss-20b
# gpt-oss-120b

# -------- GPU VISIBILITY --------
gpus=4
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