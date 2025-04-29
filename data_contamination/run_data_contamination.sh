model_name=Llama-3.1-8B-Instruct
# # Size: Small
# "Phi-4"
# "BioMistral-7B"
# "Ministral-8B-Instruct-2410"
# "MMed-Llama-3-8B"
# # Size: Mid
# "gemma-2-27b-it"
# "QwQ-32B-Preview"
# # Size: Large
# "Llama-3.1-70B-Instruct"
# "Llama-3.3-70B-Instruct"
# "Llama-3.1-Nemotron-70B-Instruct-HF"
# "meditron-70b"
# "MeLLaMA-70B-chat"
# "Llama3-OpenBioLLM-70B"
# "Llama-3-70B-UltraMedical"
# "Mistral-Large-Instruct-2411"
# "Phi-3.5-MoE-instruct"
# "Qwen2.5-72B-Instruct"
# "Athene-V2-Chat"
# "gemma-3-27b-it"
# "Llama-4-Scout-17B-16E-Instruct"
# "Mistral-Small-3.1-24B-Instruct-2503"

# -------- GPU VISIBILITY --------
gpus=4,5,6,7
# 0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=$gpus

# -------- Path to YAML config --------
CONFIG_FILE="BRIDGE_data_contamination.yaml"

# -------- RUN --------
now=$(date +"%m-%d_%H-%M")
nohup python data_contamination.py \
    --model_name $model_name \
    --config "$CONFIG_FILE" \
    > log/${model_name}.${now}.log 2>&1 &

# -------- Log output --------
echo "Job started, log in log/${model_name}.${now}.log"