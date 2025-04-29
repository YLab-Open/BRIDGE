import os
import json
import regex
import torch
import random
import numpy as np
import transformers
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    transformers.set_seed(seed)
    print(f"seed everything: {seed}")


def text_normalize(str_text):
    str_text = regex.sub(r"(\s*\n\s*)+", "\n", str_text)
    str_text = regex.sub(r"[ \t]+", " ", str_text)

    return str_text


list_model_small = [
    "gemma-2-9b-it",
    "Llama-3.1-8B-Instruct",
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "meditron-7b",
    "MeLLaMA-13B-chat",
    "Llama3-OpenBioLLM-8B",
    "MMed-Llama-3-8B",
    "Llama-3.1-8B-UltraMedical",
    "Ministral-8B-Instruct-2410",
    "Mistral-Small-Instruct-2409",
    "BioMistral-7B",
    "Phi-3.5-mini-instruct",
    "Qwen2.5-1.5B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Qwen2.5-7B-Instruct",
    "Yi-1.5-9B-Chat-16K",
]
list_model_mid = ["Phi-4", "gemma-2-27b-it", "QwQ-32B-Preview", "Yi-1.5-34B-Chat-16K"]


def get_max_token(model_name, model_path):
    # length setting
    if "Qwen" in model_name or "Athene" in model_name.lower():
        path_file_config = os.path.join(model_path, "tokenizer_config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        max_token_all = dict_config["model_max_length"]
    elif "BioMistral-7B" == model_name:
        max_token_all = 2048
    else:
        path_file_config = os.path.join(model_path, "config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        if "max_position_embeddings" in dict_config:
            max_token_all = dict_config["max_position_embeddings"]
        elif "text_config" in dict_config:
            max_token_all = dict_config["text_config"]["max_position_embeddings"]
        else:
            max_token_all = 100 * 1024

    max_token_all = 100 * 1024 if max_token_all > 100 * 1024 else max_token_all

    return max_token_all


def load_model(model_name):
    print(f"Loaded model: {model_name} with hf")
    with open(
        "../dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    model_path = dict_model_path[model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        padding_side="left",
        trust_remote_code=True,
    )

    model.eval()

    # tokenizer setting
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print("Tokenizer: Now pad_token_id is:", tokenizer.pad_token_id)
    else:
        print("Tokenizer: pad_token_id is already set:", tokenizer.pad_token_id)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        print("Model: Now pad_token_id is:", model.generation_config.pad_token_id)
    else:
        print(
            "Model: pad_token_id is already set:", model.generation_config.pad_token_id
        )

    max_token_all = get_max_token(model_name, model_path)

    return (model, tokenizer, max_token_all)


def load_model_vllm(model_name, seed=42, ngram=5):
    print(f"Loaded model: {model_name} with vllm")

    # model path
    with open(
        "../dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    model_path = dict_model_path[model_name]
    max_token_all = get_max_token(model_name, model_path)
    num_gpus = os.environ["CUDA_VISIBLE_DEVICES"].count(",") + 1
    # model init
    if "mistral" in model_name.lower() and "biomistral" not in model_name.lower():
        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_token_all,
            gpu_memory_utilization=0.95,
            tokenizer_mode="mistral",
            load_format="mistral",
            config_format="mistral",
        )
    else:
        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus,
            dtype="bfloat16",
            trust_remote_code=True,
            max_model_len=max_token_all,
            gpu_memory_utilization=0.95,
        )
    # sampling params
    sampling_params = SamplingParams(seed=seed, temperature=0, max_tokens=ngram)

    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")

    return model, tokenizer, sampling_params


def get_data_from_dir(path_dir_data="../dataset_raw/"):
    """
    Load the task data from the specified directory. The directory should contain
    JSON files with the task data. The function will return a dictionary where
    the keys are the task names and the values are the corresponding data.
    Args:
        path_dir_data (str): Path to the directory containing the task data files.
    Returns:
        dict_task_data (dict): Dictionary containing the task data. The keys are
            the task names and the values are the corresponding data.
    """
    # Check if the directory exists
    if not os.path.exists(path_dir_data):
        print(f"Directory {path_dir_data} does not exist.")
        return {}
    list_file_json = [
        os.path.join(path_dir_data, file)
        for file in os.listdir(path_dir_data)
        if file.endswith(".json") and os.path.isfile(os.path.join(path_dir_data, file))
    ]
    dict_task_data = {}
    for file in list_file_json:
        task_name = os.path.basename(file).split(".SFT")[0]
        with open(file, "r", encoding="utf-8") as f:
            dict_task_data[task_name] = json.load(f)
            print(
                f"Loaded task: {task_name} with {len(dict_task_data[task_name])} samples"
            )

    print(f"Loaded {len(dict_task_data)} tasks")

    return dict_task_data


def get_data_from_task(tasks, path_dir_data="../dataset_raw/"):
    """
    Load the task data from the specified directory. The directory should contain
    JSON files with the task data. The function will return a dictionary where
    the keys are the task names and the values are the corresponding data.
    Args:
        path_dir_data (str): Path to the directory containing the task data files.
    Returns:
        dict_task_data (dict): Dictionary containing the task data. The keys are
            the task names and the values are the corresponding data.
    """
    # Check if the directory exists
    if not os.path.exists(path_dir_data):
        print(f"Directory {path_dir_data} does not exist.")
        return {}
    list_file_json = [
        os.path.join(path_dir_data, file)
        for file in os.listdir(path_dir_data)
        if file.endswith(".json") and os.path.isfile(os.path.join(path_dir_data, file))
    ]
    dict_task_data = {}
    for file in list_file_json:
        task_name = os.path.basename(file).split(".SFT")[0]
        if task_name in tasks:
            with open(file, "r", encoding="utf-8") as f:
                dict_task_data[task_name] = json.load(f)
                print(
                    f"Loaded task: {task_name} with {len(dict_task_data[task_name])} samples"
                )
        else:
            # print(f"Skip task: {task_name} with {len(dict_task_data[task_name])} samples")
            pass

    print(f"Loaded {len(dict_task_data)} tasks")

    return dict_task_data


def calculate_my_n_gram_accuracy_vllm(
    n,
    k,
    dataset,
    model,
    tokenizer,
    sampling_params,
    path_file_output,
    token_gap=5,
    min_prompt_start=10,
    max_token_all=3072,
):
    """
    Calculate n-gram accuracy using a vLLM model interface. For each text sample,
    we select multiple starting positions, build a prompt, let the model generate
    an n-gram, and compare it to the reference n-gram.

    Args:
        n (int): Size of the n-gram to predict.
        k (int): Number of main starting points to use for each sample.
        dataset (list[str]): List of text samples.
        model: vLLM model interface with a .generate() method.
        tokenizer: Corresponding tokenizer with .tokenize() and .convert_tokens_to_string().
        sampling_params: Sampling params for the model.generate() call.
        path_file_output (str): Output path to save detailed JSON results.
        token_gap (int): Gap (in tokens) between consecutive starting points. Default 5.
        min_prompt_start (int): Minimum starting token index. Default 10.
        max_token_all (int): Maximum token positions to consider from each sample. Default 3072.

    Returns:
        dict:
            {
                "n_grams": list_dict_sample,  # Detailed info about each sample
                "mean_n_grams": float,        # Mean accuracy across valid positions
            }
    """

    # Preprocess the dataset, merge the redundant spaces, \n, \t
    # dataset = [text_normalize(text) for text in dataset]

    # Prepare the dataset
    dataset_tokenized = [
        tokenizer.tokenize(text, add_special_tokens=False) for text in dataset
    ]
    list_dict_sample = [
        {"idx": i, "sample": text, "n_gram_results": []}
        for i, text in enumerate(dataset)
    ]

    # Initialize results and accuracy tracking
    accuracies = np.zeros((len(dataset), 2))  # [correct, total]

    list_input = []
    list_output_text, list_output_token, list_idx_pos = [], [], []

    # If need to extend the starting positions
    # preset_positions = [40, 80, 160, 320, 640]

    for idx_data, tokens in enumerate(dataset_tokenized):
        # Get the starting positions between min_prompt_start and max_token_all
        # len_tokens = len(tokens)
        # if min_prompt_start + n > len_tokens:
        #     continue
        # Select starting points using linspace
        # list_pos_start = np.linspace(
        #     min_prompt_start,
        #     min(len_tokens, max_token_all) - n,
        #     num=k,
        #     dtype=int,
        # )
        # list_pos_start = list(set(list_pos_start))

        if len(tokens) < min_prompt_start + (k - 1) * token_gap + n:
            continue

        list_pos_start = [min_prompt_start + i * token_gap for i in range(k)]

        # Extend the starting positions with preset positions
        # for pos_start in preset_positions:
        #     if pos_start + n <= len(tokens):
        #         list_pos_start.append(pos_start)

        # Build prompts and record the original n-gram IDs
        for pos_start in list_pos_start:
            prefix_text = tokenizer.convert_tokens_to_string(tokens[:pos_start])
            output_text = tokenizer.convert_tokens_to_string(
                tokens[pos_start : pos_start + n]
            )
            output_token = tokenizer.convert_tokens_to_ids(
                tokens[pos_start : pos_start + n]
            )
            list_input.append(prefix_text)
            list_output_text.append(output_text)
            list_output_token.append(output_token)
            list_idx_pos.append((idx_data, pos_start))

    # Generate responses
    list_response_generator = model.generate(
        list_input,
        sampling_params,
        use_tqdm=True,
    )
    # Extract the predicted texts
    list_pred_token = [
        list(response.outputs[0].token_ids[-n:]) for response in list_response_generator
    ]
    list_pred_text = [response.outputs[0].text for response in list_response_generator]

    for i, (idx_data, pos_start) in enumerate(list_idx_pos):
        # Extract the predicted and original n-grams
        pred_token = list_pred_token[i]
        pred_text = list_pred_text[i]
        output_token = list_output_token[i]
        output_text = list_output_text[i]
        acc_binary = pred_token == output_token
        acc_prop = sum(p == o for p, o in zip(pred_token, output_token)) / len(
            pred_token
        )
        # Record detailed results
        list_dict_sample[idx_data]["n_gram_results"].append(
            {
                "start_index": int(pos_start),
                "prompt": list_input[i],
                "original_ids": output_token,
                "original_text": output_text,
                "predicted_ids": pred_token,
                "predicted_text": pred_text,
                "accuracy_binary": acc_binary,
                "accuracy_proportion": acc_prop,
            }
        )
        if acc_binary:
            accuracies[idx_data][0] += 1  # correct
        accuracies[idx_data][1] += 1  # total

    list_acc = [correct / total for correct, total in accuracies if total > 0]
    mean_acc = np.mean(list_acc) if list_acc else 0.0

    # Save the detailed results
    with open(path_file_output, "w", encoding="utf-8") as f:
        json.dump(list_dict_sample, f, indent=4, ensure_ascii=False)

    return list_dict_sample, mean_acc


def calculate_my_n_gram_accuracy_hf_batch(
    n,
    k,
    dataset,
    model,
    tokenizer,
    model_name,
    path_file_output,
    token_gap=5,
    batch_size=32,
    min_prompt_start=10,
    max_token_all=3072,
):
    """
    Calculate n-gram accuracy using Hugging Face model in batches.
    For each sample, select multiple starting positions (using a series based on token_gap
    and additional preset positions), build prompts, let the model generate an n-gram,
    and then compare the generated n-gram with the reference.

    Args:
        n (int): Size of the n-gram to predict.
        k (int): Number of main starting points (using min_prompt_start + i*token_gap).
        dataset (list[str]): List of text samples.
        model: Hugging Face model with a .generate() method.
        tokenizer: Corresponding tokenizer.
        sampling_params (dict): Sampling parameters for model.generate() call.
        path_file_output (str): Output path for saving detailed JSON results.
        token_gap (int): Gap between consecutive starting positions. default 5.
        min_prompt_start (int): Minimum starting token index. default 10.
        max_token_all (int): Maximum token positions to consider from each sample. default 3072.
        batch_size (int): Batch size.

    Returns:
        dict:
            {
                "n_grams": list_dict_sample,  # Detailed info about each sample
                "mean_n_grams": float,        # Mean accuracy across valid positions
            }
    """
    generate_kwargs = {
        "max_new_tokens": n,
        "do_sample": False,
        "temperature": None,
        "top_k": None,
        "top_p": None,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tokenize dataset once to avoid redundancy
    dataset_tokenized = [
        tokenizer.tokenize(text)
        for text in dataset
        # , add_special_tokens=False
    ]

    # Initialize results and accuracy tracking
    list_dict_sample = [
        {"idx": i, "sample": text, "n_gram_results": []}
        for i, text in enumerate(dataset)
    ]

    accuracies = np.zeros((len(dataset), 2))  # Stores [correct, total] per sample

    list_input = []
    list_output_text, list_output_token, list_idx_pos = [], [], []

    # If need to extend the starting positions
    # preset_positions = [40, 80, 160, 320, 640]

    # Iterate over all samples
    for idx_data, tokens in enumerate(dataset_tokenized):
        # If the sample is too short, skip
        if len(tokens) < min_prompt_start + (k - 1) * token_gap + n:
            continue

        # Generate starting positions based on token_gap
        list_pos_start = [min_prompt_start + i * token_gap for i in range(k)]

        # Extend the starting positions with preset positions
        # for pos in preset_positions:
        #     if pos + n <= len(tokens):
        #         list_pos_start.append(pos)

        # deduplicate and sort the starting positions
        list_pos_start = list(set(list_pos_start))
        # Build prompts and record the original n-gram IDs
        for pos_start in list_pos_start:
            prefix_text = tokenizer.convert_tokens_to_string(tokens[:pos_start])
            output_text = tokenizer.convert_tokens_to_string(
                tokens[pos_start : pos_start + n]
            )
            output_token = tokenizer.convert_tokens_to_ids(
                tokens[pos_start : pos_start + n]
            )
            list_input.append(prefix_text)
            list_output_text.append(output_text)
            list_output_token.append(output_token)
            list_idx_pos.append((idx_data, pos_start))

    # Generate responses in batches
    num_batches = (len(list_input) + batch_size - 1) // batch_size
    list_pred_token = []
    list_pred_text = []
    for i in tqdm(range(num_batches), desc="Generating n-gram predictions"):
        batch_inputs = list_input[i * batch_size : (i + 1) * batch_size]
        # Tokenize the batch and generate predictions
        batch_encoded = tokenizer(
            batch_inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_token_all - n,
        ).to(device)
        with torch.no_grad():
            outputs = model.generate(**batch_encoded, **generate_kwargs)
        # Decode the predictions and store them
        for j in range(outputs.shape[0]):
            pred_ids = outputs[j][-n:].tolist()
            pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
            list_pred_token.append(pred_ids)
            list_pred_text.append(pred_text)

    # Iterate over the predictions and compare them to the original n-grams
    for i, (sample_idx, pos_start) in enumerate(list_idx_pos):
        pred_token = list_pred_token[i]
        pred_text = list_pred_text[i]
        orig_token = list_output_token[i]
        orig_text = list_output_text[i]
        acc_binary = pred_token == orig_token
        acc_prop = sum(p == o for p, o in zip(pred_token, orig_token)) / len(pred_token)
        list_dict_sample[sample_idx]["n_gram_results"].append(
            {
                "start_index": int(pos_start),
                "prompt": list_input[i],
                "original_ids": orig_token,
                "original_text": orig_text,
                "predicted_ids": pred_token,
                "predicted_text": pred_text,
                "accuracy_binary": acc_binary,
                "accuracy_proportion": acc_prop,
            }
        )
        # Update the accuracy tracking
        if acc_binary:
            accuracies[sample_idx][0] += 1
        accuracies[sample_idx][1] += 1

    # Calculate the mean accuracy across all samples
    list_acc = [correct / total for correct, total in accuracies if total > 0]
    mean_acc = np.mean(list_acc) if list_acc else 0.0

    # Save the detailed results
    with open(path_file_output, "w", encoding="utf-8") as f:
        json.dump(list_dict_sample, f, indent=4, ensure_ascii=False)

    return list_dict_sample, mean_acc
