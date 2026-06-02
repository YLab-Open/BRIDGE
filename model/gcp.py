import os
import json
import glob
from dataset.process import get_formatted_chat_gemini
from util.runtime import parse_prompt_mode
from model.provider_common import apply_cot_instruction_transform, get_example_path


def create_gcp_batch_data(
    task_name,
    model_name,
    prompt_mode,
    split="test",
    path_dir_raw="../dataset_raw",
    path_dir_batch="gcp/input",
    path_dir_mapping="gcp/mapping",
    temperature=0,
    top_p=0,
    max_token_input=1048576,  # Gemini's context window
    max_token_output=8192,
    seed=42,
):
    """
    Create a list of dictionary for batch data in GCP Gemini format

    Input:
        task_name: string, the task name
        model_name: string, the model name
        prompt_mode: string, the prompt mode (direct/cot/few-shot)
        split: string, the split of the data
        path_dir_raw: string, path to raw dataset
        path_dir_batch: string, path to save batch files
        temperature: float, sampling temperature
        top_p: float, nucleus sampling parameter
        max_token_input: int, maximum input tokens
        max_token_output: int, maximum output tokens
        seed: int, random seed

    Output:
        list_dict_data_batch: list of dictionary in GCP batch format
    """

    if "flash" in model_name:
        max_token_input = 1048576
        max_token_output = 8192
    elif "pro" in model_name:
        max_token_input = 2097152
        max_token_output = 8192
    else:
        raise ValueError(f"Model {model_name} is not supported")

    # Load data
    path_file_task = f"{path_dir_raw}/{task_name}.SFT.json"
    with open(path_file_task, "r", encoding="utf-8") as f:
        list_dict_data = json.load(f)

    # Filter the data
    list_dict_data = [
        dict_data for dict_data in list_dict_data if dict_data["split"] == split
    ]
    print(f"Task: {task_name}")
    print(f" - {split.capitalize()} split: {len(list_dict_data)} samples")
    print(f" - Model: {model_name}")
    print(f" - Prompt: {prompt_mode}")

    prompt_base, num_example = parse_prompt_mode(prompt_mode)

    # Prepare instruction based on prompt mode
    if prompt_base == "direct":
        pass
    elif prompt_base == "cot":
        apply_cot_instruction_transform(list_dict_data, task_name)
        print(f" - Transform the instruction to the CoT format")
        if list_dict_data:
            print(f" - Instruction: \n{list_dict_data[0]['instruction']}")

    # Prepare examples if using few-shot
    if num_example is not None:
        path_file_example = get_example_path(path_dir_raw, task_name, prompt_base)
        with open(path_file_example, "r", encoding="utf-8") as f:
            list_dict_example = json.load(f)
        examples = list_dict_example[:num_example]
        print(f" - Prepare {num_example} examples")
    else:
        examples = []
        print(" - No example")

    # Get formatted data
    list_dict_data_batch, input_to_id_dict = prepare_gcp_batch_data(
        task_name=task_name,
        model_name=model_name,
        prompt_mode=prompt_mode,
        split=split,
        list_dict_data=list_dict_data,
        examples=examples,
        temperature=temperature,
        top_p=top_p,
        max_token_input=max_token_input,
        max_token_output=max_token_output,
        seed=seed,
    )

    # Save data
    path_dir_gcp_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}"
    if not os.path.exists(path_dir_gcp_batch):
        os.makedirs(path_dir_gcp_batch)
    path_file_gcp_batch = f"{path_dir_gcp_batch}/{task_name}.batch.jsonl"
    save_gcp_batch_file(list_dict_data_batch, path_file_gcp_batch)
    print(f" - Save {len(list_dict_data_batch)} to {path_file_gcp_batch}")

    # Save input_to_id_dict (the mapping from input to custom id) as a JSON file
    path_dir_input_to_id_dict = f"{path_dir_mapping}/{model_name}/{prompt_mode}"
    if not os.path.exists(path_dir_input_to_id_dict):
        os.makedirs(path_dir_input_to_id_dict)
    path_file_input_to_id_dict = (
        f"{path_dir_input_to_id_dict}/{task_name}.input_to_id.json"
    )
    with open(path_file_input_to_id_dict, "w", encoding="utf-8") as f:
        json.dump(input_to_id_dict, f, ensure_ascii=False, indent=4)
    print(
        f" - Save {len(input_to_id_dict)} lines of input_to_id_dict to {path_file_input_to_id_dict}"
    )

    return list_dict_data_batch


def prepare_gcp_batch_data(
    task_name,
    model_name,
    prompt_mode,
    split,
    list_dict_data,
    examples=None,
    temperature=0,
    top_p=0,
    max_token_input=1048576,
    max_token_output=8192,
    seed=42,
    num_token_reserve=5,
):
    """
    Prepare data in GCP Gemini batch prediction format

    Input:
        task_name: string, the task name
        prompt_mode: string, the prompt mode
        list_dict_data: list of dictionary, including input, instruction, and split
        temperature: float, the temperature for the model
        top_p: float, the top_p for the model
        frequency_penalty: float, the frequency_penalty for the model
        presence_penalty: float, the presence_penalty for the model
        max_token_outputs: int, the max_token_outputs for the model

    Output:
        list_dict_data_batch: list of dictionary, including input, instruction, and model name

    """
    from vertexai.preview import tokenization

    examples = examples or []
    parse_prompt_mode(prompt_mode)
    if not list_dict_data:
        print(" - No data to prepare")
        return [], {}

    try:
        tokenizer = tokenization.get_tokenizer_for_model(model_name)
        print(f" - Loading tokenizer of {model_name}")
    except:
        if "pro" in model_name:
            tokenizer = tokenization.get_tokenizer_for_model("gemini-1.5-pro-002")
            print(
                f"Cannot find tokenizer for {model_name}, use gemini-1.5-pro-002's tokenizer instead"
            )
        elif "flash" in model_name:
            tokenizer = tokenization.get_tokenizer_for_model("gemini-1.5-flash-002")
            print(
                f"Cannot find tokenizer for {model_name}, use gemini-1.5-flash-002's tokenizer instead"
            )
        else:
            raise ValueError(f"Model {model_name} is not supported")

    if "flash" in model_name:
        max_token_input = 1048576
        max_token_output = 8192
    elif "pro" in model_name:
        max_token_input = 2097152
        max_token_output = 8192
    else:
        raise ValueError(f"Model {model_name} is not supported")

    print(f" - Max token input: {max_token_input}")
    print(f" - Max token output: {max_token_output}")

    language = list_dict_data[0]["language"]

    # Prepare id prefix
    exp_name = f"{task_name}|{model_name}|{prompt_mode}|{split}"

    list_dict_data_batch = []
    input_to_id_dict = {}
    duplicate_count = 0
    for idx, dict_data in enumerate(list_dict_data):
        example_ids, list_message = get_formatted_chat_gemini(
            # example_ids, list_message = get_formatted_chat_ins_ex(
            tokenizer=tokenizer,
            language=language,
            input_system=dict_data["instruction"],
            input_user=dict_data["input"],
            max_token_input=max_token_input,
            times_word_token=4,
            num_token_reserve=5,
            examples=examples,
        )

        # Create the batch instance in GCP format
        dict_data_batch = {"request": {"contents": list_message}}
        list_dict_data_batch.append(dict_data_batch)

        custom_id = f"{exp_name}|{dict_data['id']}"
        mapping_key_list = []
        for item in list_message:
            if item["role"] == "user":
                mapping_key_list.append(item["parts"][0]["text"])
        mapping_key = "".join(mapping_key_list)
        if mapping_key in input_to_id_dict:
            duplicate_count += 1
            print(f" - Duplicated case {duplicate_count}:")
            print(f" - Duplicated mapping key: {mapping_key}")
            print(f" - Old id: {input_to_id_dict[mapping_key]}")
            print(f" - New id: {custom_id}\n")
            # raise ValueError("Duplicated mapping key")
        input_to_id_dict[mapping_key] = custom_id

    return list_dict_data_batch, input_to_id_dict


def save_gcp_batch_file(list_dict_data_batch, path_file_batch):
    """
    Save the batch data to a JSONL file for GCP batch prediction

    Input:
        list_dict_data_batch: list of dictionaries in GCP format
        path_file_batch: string, path to save the batch file
    """
    if "jsonl" not in path_file_batch:
        print("Please use the jsonl format")
        return None

    with open(path_file_batch, "w", encoding="utf-8") as f:
        for dict_data_batch in list_dict_data_batch:
            f.write(json.dumps(dict_data_batch, ensure_ascii=False) + "\n")


def merge_gcp_batch_data(
    model_name,
    prompt_mode,
    path_dir_raw="gcp/input",
    path_dir_merged="gcp/input/merged",
    max_lines=None,
    max_size_mb=None,
):
    """
    Merge multiple GCP batch files into larger files while respecting size limits
    """
    path_dir_raw = f"{path_dir_raw}/{model_name}/{prompt_mode}"
    os.makedirs(path_dir_merged, exist_ok=True)

    list_path_file = [
        os.path.join(path_dir_raw, file)
        for file in os.listdir(path_dir_raw)
        if file.endswith(".jsonl")
    ]

    chunk_index = 0
    current_lines = []
    current_size = 0
    if max_lines is not None and max_size_mb is not None:
        max_size_bytes = max_size_mb * 1024 * 1024

    for task_file in list_path_file:
        with open(task_file, "r", encoding="utf-8") as infile:
            file_lines = [line.rstrip("\n") for line in infile]

        file_size = sum(len(line.encode("utf-8")) for line in file_lines)

        if (
            max_lines is not None
            and max_size_mb is not None
            and (
                (len(current_lines) + len(file_lines) > max_lines)
                or (current_size + file_size > max_size_bytes)
            )
        ):
            # Write current chunk
            output_file = os.path.join(
                path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
            )
            with open(output_file, "w", encoding="utf-8") as outfile:
                outfile.write("\n".join(current_lines))
            print(
                f"Created: {output_file} with {len(current_lines)} lines "
                f"and size {current_size / (1024 * 1024):.2f} MB"
            )

            chunk_index += 1
            current_lines = []
            current_size = 0

        current_lines.extend(file_lines)
        current_size += file_size

    # Write final chunk if there are remaining lines
    if current_lines:
        output_file = os.path.join(
            path_dir_merged, f"{model_name}.{prompt_mode}.chunk_{chunk_index}.jsonl"
        )
        with open(output_file, "w", encoding="utf-8") as outfile:
            outfile.write("\n".join(current_lines))
        print(
            f"Created: {output_file} with {len(current_lines)} lines "
            f"and size {current_size / (1024 * 1024):.2f} MB"
        )


def merge_gcp_mapping(
    model_name,
    prompt_mode_list=["direct", "cot", "direct-5-shot"],
    path_dir_raw_mapping="gcp/mapping",
    path_dir_merged_mapping="gcp/mapping/merged",
):
    """
    Merge multiple GCP mapping files into a single dictionary
    """
    if not os.path.exists(path_dir_merged_mapping):
        os.makedirs(path_dir_merged_mapping)
    duplicate_count = 0
    all_input_to_id_dict = {}
    for prompt_mode in prompt_mode_list:
        # Merge per-task input_to_id mapping files for this prompt mode.
        all_input_to_id_dict_path = glob.glob(
            f"{path_dir_raw_mapping}/{model_name}/{prompt_mode}/*.input_to_id.json"
        )
        for input_to_id_dict_path in all_input_to_id_dict_path:
            with open(input_to_id_dict_path, "r", encoding="utf-8") as f:
                input_to_id_dict = json.load(f)
            for key, value in input_to_id_dict.items():
                if key in all_input_to_id_dict:
                    duplicate_count += 1
                    print(f" - Duplicated case {duplicate_count}:")
                    print(f" - Duplicated mapping key: {key}")
                    print(f" - Old id: {all_input_to_id_dict[key]}")
                    print(f" - New id: {value}\n")
                    # raise ValueError("Duplicated mapping key")
                all_input_to_id_dict[key] = value
    path_file_input_to_id_dict = os.path.join(
        path_dir_merged_mapping, f"{model_name}.input_to_id.json"
    )
    with open(path_file_input_to_id_dict, "w", encoding="utf-8") as f:
        json.dump(all_input_to_id_dict, f, ensure_ascii=False, indent=4)
    print(
        f" - Save {len(all_input_to_id_dict)} lines of merged input_to_id_dict for {model_name} to {path_file_input_to_id_dict}"
    )


def parse_gcp_batch_result(path_file_result_batch, path_dir_mapping):
    """
    Parse the results from GCP batch prediction output

    Input:
        path_file_result_batch: string, path to the result file

    Output:
        list_dict_id_result: list of dictionaries with id, output and error info
    """
    with open(path_file_result_batch, "r", encoding="utf-8") as f:
        list_dict_result = [json.loads(line) for line in f.readlines()]

    model_name = "-".join(
        list_dict_result[0]["response"]["modelVersion"].split("-")[:-1]
    )
    mapping_dict_path = os.path.join(path_dir_mapping, f"{model_name}.input_to_id.json")
    with open(mapping_dict_path, "r", encoding="utf-8") as f:
        mapping_dict = json.load(f)
    list_dict_id_result = []
    for dict_result in list_dict_result:
        part_list = dict_result["request"]["contents"]
        mapping_key_list = []
        for part in part_list:
            if part["role"] == "user":
                mapping_key_list.append(part["parts"][0]["text"])
        mapping_key = "".join(mapping_key_list)
        try:
            dict_id_result = {
                "id": mapping_dict[mapping_key],
                "output": (
                    dict_result["response"]["candidates"][0]["content"]["parts"][0][
                        "text"
                    ][:-1]
                    if dict_result["response"]["candidates"][0]["content"]["parts"][0][
                        "text"
                    ][-1]
                    == "\n"
                    else dict_result["response"]["candidates"][0]["content"]["parts"][
                        0
                    ]["text"]
                ),
            }
        except Exception as e:
            print(f"Error - {mapping_dict[mapping_key]}: {e}")
            print(f"The raw response from the model is:\n{dict_result['response']}")
            dict_id_result = {
                "id": mapping_dict[mapping_key],
                "output": "",
            }
        list_dict_id_result.append(dict_id_result)

    return list_dict_id_result


def process_gcp_result_to_task_result(
    path_dir_gcp_result="gcp/output",
    path_dir_task_save="result",
    path_dir_raw="../dataset_raw",
    path_dir_batch="gcp/input",
    path_dir_mapping="gcp/mapping/merged",
    decoding_strategy="greedy",
    seed=42,
):
    """
    Process the result from the gcp batch files

    Input:
        list_path_file: list of string, the list of path file
        path_dir_result: string, the path of the result directory
        path_dir_save: string, the path of the save directory

    Output:
        None

    """
    # Get the list of path file, which is the result file of gcp batch job
    list_path_file = [
        os.path.join(path_dir_gcp_result, path_file)
        for path_file in os.listdir(path_dir_gcp_result)
        if path_file.endswith(".jsonl")
    ]

    # Parse the result, as the result contains multiple tasks
    for path_file_result in list_path_file:

        # Read the result file
        list_dict_id_result = parse_gcp_batch_result(path_file_result, path_dir_mapping)

        # Split the whole result into different tasks
        dict_task_data_all = {}
        for dict_result in list_dict_id_result:
            # Get the task information: task, model_name, prompt_mode
            task, model_name, prompt_mode, split, id = dict_result["id"].split("|")

            if task not in dict_task_data_all:
                dict_task_data_all[task] = {
                    "model_name": model_name,
                    "prompt_mode": prompt_mode,
                    "split": split,
                    "list_dict_result": [],
                }

            dict_task_data_all[task]["list_dict_result"].append(dict_result)

        # Construct the result for data file, which integrates the raw data, batch data, and result
        for task, dict_task_data in dict_task_data_all.items():
            model_name = dict_task_data["model_name"]
            prompt_mode = dict_task_data["prompt_mode"]
            split = dict_task_data["split"]
            print(f"Task: {task}")
            print(f" - Model Name: {dict_task_data['model_name']}")
            print(f" - Prompt Mode: {dict_task_data['prompt_mode']}")
            print(f" - Split: {dict_task_data['split']}")
            print("----------------------------------------")

            # Merge the raw data, batch data, and result
            list_dict_data = integrate_task_data_batch_result(
                task=task,
                model_name=model_name,
                prompt_mode=prompt_mode,
                split=split,
                dict_task_data=dict_task_data,
                path_dir_raw=path_dir_raw,
                path_dir_batch=path_dir_batch,
            )

            # Save the result
            model_name = model_name.replace("-batch", "")
            path_file_save = os.path.join(
                path_dir_task_save,
                f"{task}/{model_name}/{task}-{prompt_mode}-{decoding_strategy}-{seed}.result.json",
            )
            os.makedirs(os.path.dirname(path_file_save), exist_ok=True)
            with open(path_file_save, "w", encoding="utf-8") as f:
                json.dump(list_dict_data, f, ensure_ascii=False, indent=4)
            print(f" - Saved: {path_file_save}")
            print("========================================")


def integrate_task_data_batch_result(
    task,
    model_name,
    prompt_mode,
    split,
    dict_task_data,
    path_dir_raw="../dataset_raw",
    path_dir_batch="gcp/input",
):
    """
    Integrate the task data with the batch result.

    Input:
        task: string, the task name
        model_name: string, the model name
        prompt_mode: string, the prompt mode
        split: string, the split of the data
        dict_task_data: dictionary, including model_name, prompt_mode, split, and list_dict_result

    Output:
        list_dict_data: list of dictionary, including id, input, pred, token_input, and token_output

    """
    # Read the raw data file
    path_file_raw = f"{path_dir_raw}/{task}.SFT.json"
    with open(path_file_raw, "r") as f:
        list_dict_data = json.load(f)
    # Filter the data by split
    list_dict_data = [
        dict_data for dict_data in list_dict_data if dict_data["split"] == split
    ]
    print(f" - Num of data: {len(list_dict_data)}")

    # Read the batch data file
    path_file_batch = f"{path_dir_batch}/{model_name}/{prompt_mode}/{task}.batch.jsonl"
    with open(path_file_batch, "r") as f:
        list_dict_batch = [json.loads(line) for line in f]
    print(f" - Num of batch: {len(list_dict_batch)}")

    assert len(list_dict_data) == len(list_dict_batch)

    # Assign the input and output to the data
    for dict_data, dict_batch in zip(list_dict_data, list_dict_batch):
        dict_data["input"] = dict_batch["request"]["contents"]

    # Assign the output to the data
    dict_id_result = {
        dict_result["id"].split("|")[-1]: dict_result
        for dict_result in dict_task_data["list_dict_result"]
    }
    print(f" - Num of result: {len(dict_id_result)}")
    count_match = 0
    for dict_data in list_dict_data:
        id_data = str(dict_data["id"])
        if id_data in dict_id_result:
            dict_result = dict_id_result[id_data]
            dict_data["pred"] = dict_result["output"]
            count_match += 1
        else:
            print(f" - Missing: {id_data}")
            dict_data["pred"] = ""

    print(f" - Matched: {count_match}")

    if count_match == len(list_dict_data):
        print(" - All matched.")
    else:
        print(f" - Lost {len(list_dict_data) - count_match} samples.")

    return list_dict_data


def get_gemini_price(model_name, flag_batch=False):
    """
    Get the price of the gemini model

    Input:
        model_name: string, the model name
        flag_batch: boolean, the flag for batch

    Output:
        cost_input_1m: float, the cost of input per 1M tokens
        cost_output_1m: float, the cost of output per 1M tokens

    """
    dict_model_price = {
        "gemini-2.0-flash": [0.1, 0.4],
        "gemini-1.5-pro": [1.25, 5],
        "gemini-1.5-flash": [0.075, 0.3],
    }

    cost_input_1m, cost_output_1m = dict_model_price[model_name]
    if flag_batch:
        return cost_input_1m / 2, cost_output_1m / 2
    else:
        return cost_input_1m, cost_output_1m


def get_MedLM_price(model_name):
    """
    Get the price of the MedLM model

    Input:
        model_name: string, the model name

    Output:
        cost_input_1m: float, the cost of input per 1M characters
        cost_output_1m: float, the cost of output per 1M characters

    """
    dict_model_price = {
        "MedLM-large": [4, 12],
        "MedLM-medium": [4, 12],
    }

    cost_input_1m, cost_output_1m = dict_model_price[model_name]
    return cost_input_1m, cost_output_1m


def cost_calculation_gemini(
    num_token_input_all,
    num_token_output_all,
    model_name,
    flag_batch=False,
):
    """
    Calculate the cost of the gemini model

    Input:
        num_token_input_all: int, the number of input tokens
        num_token_output_all: int, the number of output tokens
        model_name: string, the model name
        flag_batch: boolean, the flag for batch

    Output:
        cost_total: float, the total cost

    """

    cost_input_1m, cost_output_1m = get_gemini_price(model_name, flag_batch)
    cost_input = num_token_input_all * cost_input_1m / 10**6
    cost_output = num_token_output_all * cost_output_1m / 10**6
    cost_total = cost_input + cost_output

    return cost_total


def cost_calculation_MedLM(
    num_character_input_all,
    num_character_output_all,
    model_name,
    flag_batch=False,
):
    """
    Calculate the cost of the MedLM model

    Input:
        num_token_input_all: int, the number of input characters
        num_token_output_all: int, the number of output characters
        model_name: string, the model name
        flag_batch: boolean, the flag for batch

    Output:
        cost_total: float, the total cost

    """

    cost_input_1m, cost_output_1m = get_MedLM_price(model_name, flag_batch)
    cost_input = num_character_input_all * cost_input_1m / 10**6
    cost_output = num_character_output_all * cost_output_1m / 10**6
    cost_total = cost_input + cost_output

    return cost_total
