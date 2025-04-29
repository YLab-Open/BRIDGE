import os
import json
import time
import yaml
import argparse
import setproctitle
from pathlib import Path

from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="Llama-3.1-8B-Instruct",
        help="The name of the LLM model, only one model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file containing tasks, exp_configs, and other arguments.",
    )

    args = parser.parse_args()

    # ---------- Load YAML & merge ----------
    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path.resolve()}")

        with open(cfg_path, encoding="utf-8") as f:
            dict_cfg = yaml.safe_load(f)

        # Write the YAML key value back to argparse.Namespace
        # Note: YAML has a lower priority than CLI
        for k, v in dict_cfg.items():
            if getattr(args, k, None) == parser.get_default(k):
                # Only set if not already set by CLI
                setattr(args, k, v)

    # Initialize the model
    if args.inference_mode == "vllm":
        # Load the model with vllm
        model, tokenizer, sampling_params = load_model_vllm(args.model_name)
    elif args.inference_mode == "hf":
        # Load the model with huggingface
        model, tokenizer, max_token_all = load_model(args.model_name)
    else:
        raise ValueError("Invalid inference mode. Choose 'vllm' or 'hf'.")

    print("=" * 20, "Experiment setting", "=" * 20)
    for key, value in args.__dict__.items():
        print(f"{key}: {value}")
    print("=" * 20, "Experiment setting", "=" * 20)

    # Get the data
    dict_task_data = get_data_from_task(
        tasks=args.tasks, path_dir_data=args.path_dir_data
    )
    print("The number of tasks: ", len(dict_task_data))

    dict_task_result = {}
    path_dir_result = os.path.join(args.path_dir_save, args.model_name)
    os.makedirs(path_dir_result, exist_ok=True)

    # File for summary of the results: all tasks for this model
    path_file_summary = os.path.join(path_dir_result, f"ngram.{args.model_name}.json")

    # Record the time of the start
    start_time = time.time()
    str_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    print("Start time: ", str_start_time)

    for idx_task, (task, data) in enumerate(dict_task_data.items()):
        print("\n" + "=" * 20, "Start: ", task, "=" * 20 + "\n")

        data_input = [dict_data["input"] for dict_data in data][:64]

        print(f"Data size: {len(data_input)}")

        # Set the process title
        setproctitle.setproctitle(
            f"Benchmark: Leakage - {args.ngram}-gram - {args.model_name} ({idx_task + 1}/{len(dict_task_data)})"
        )

        path_dir_result_task = os.path.join(path_dir_result, task)
        os.makedirs(path_dir_result_task, exist_ok=True)

        dict_ngram_acc = {}

        # Calculate the n-gram accuracy
        path_file_result_task = os.path.join(
            path_dir_result_task, f"{args.ngram}-gram-{args.model_name}.json"
        )
        if args.inference_mode == "vllm":
            list_dict_sample, mean_acc = calculate_my_n_gram_accuracy_vllm(
                n=args.ngram,
                k=args.k,
                dataset=data_input,
                model=model,
                tokenizer=tokenizer,
                sampling_params=sampling_params,
                path_file_output=path_file_result_task,
            )
        else:
            list_dict_sample, mean_acc = calculate_my_n_gram_accuracy_hf_batch(
                n=args.ngram,
                k=args.k,
                dataset=data_input,
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                path_file_output=path_file_result_task,
                batch_size=args.batch_size,
            )

        dict_ngram_acc[f"{args.ngram}-gram"] = mean_acc
        dict_task_result[task] = dict_ngram_acc
        print(f"\nAccuracy: {mean_acc}\n")

        if idx_task % 2 == 0 or idx_task == len(dict_task_data) - 1:
            # If the file exists, load the previous file and merge the result into current result
            if os.path.exists(path_file_summary):
                print("The result file exists, load the previous file and merge.")
                with open(path_file_summary, "r", encoding="utf-8") as fp:
                    result_before = json.load(fp)
                # Merge the result
                for key, value in result_before.items():
                    if key in dict_task_result and dict_task_result[key]:
                        pass
                    else:
                        dict_task_result[key] = value
            else:
                print("The result file does not exist, create a new one.")

            # Save the result
            with open(path_file_summary, "w", encoding="utf-8") as fp:
                json.dump(dict_task_result, fp, ensure_ascii=False, indent=2)

        print("\n" + "=" * 20, "End: ", task, "=" * 20 + "\n")

    # Record the time of the end
    end_time = time.time()
    str_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_time))
    print("End time: ", str_end_time)
    cost_time_hour = (end_time - start_time) / 60
    print("Time cost: ", cost_time_hour, " mins.")

    print("\n" + "=" * 20, "All Done.", "=" * 20 + "\n")
