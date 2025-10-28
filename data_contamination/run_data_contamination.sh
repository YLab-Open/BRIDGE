model_name=Qwen2.5-72B-Instruct

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