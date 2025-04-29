import os
import json
import yaml
import argparse
import datetime
import setproctitle
from pathlib import Path

from util.tool import init_logger
from dataset.dataset import GeneralTask
from model.init import load_model, seed_everything
from model.inference import run_hf, run_vllm


def parse_gpu_list(gpu_str: str) -> list[int]:
    """Convert '0,1,2' -> [0,1,2] and validate."""
    try:
        return [int(x) for x in gpu_str.split(",") if x != ""]
    except ValueError as e:
        raise ValueError(f"Invalid --gpus format: {gpu_str}") from e


def save_result(args, logger, list_dict_data, list_response, save_input=True):
    """Save predictions into json file."""
    list_to_save = []
    for dict_data, pred in zip(list_dict_data, list_response):
        if save_input:
            # Save the original data and the response
            dict_data["pred"] = pred
            list_to_save.append(dict_data)
        else:
            # Only save the response, not the original instruction and input
            dict_result = {
                "task name": dict_data["task name"],
                "language": dict_data["language"],
                "task type": dict_data["task type"],
                "idx": dict_data["idx"],
                "output": dict_data["output"],
                "pred": pred,
            }
            list_to_save.append(dict_result)

    # Save the results to a json file
    Path(args.path_file_result).parent.mkdir(parents=True, exist_ok=True)
    with open(args.path_file_result, "w", encoding="utf-8") as f:
        json.dump(list_to_save, f, indent=4, ensure_ascii=False)

    logger.info(f"Save: {args.path_file_result}")
    logger.info("\n" + "-" * 30 + "\n")
    return list_to_save


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tasks",
        type=list,
        default=["ADE-ADE identification"],
        help="The list of tasks to run.",
    )
    parser.add_argument(
        "--inference_mode",
        type=str,
        default="vllm",
        help="The inference mode for the model.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Meta-Llama-3.1-70B-Instruct",
        help="The name of the LLM model, only one model.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="The gpus for the model.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file containing tasks, exp_configs, and other arguments.",
    )

    args = parser.parse_args()

    args.gpus = parse_gpu_list(args.gpus)

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

    # ---------- Experiment config ----------
    args.tasks.sort()
    list_exp_config = [tuple(item) for item in args.experiments]

    num_exp_all = len(args.tasks) * len(list_exp_config)

    # Set the name of the experiment
    setproctitle.setproctitle(f"Benchmark: {args.model_name}-{num_exp_all} runs.")

    # Get the model path
    with open(
        "dict_model_path.json",
        "r",
        encoding="utf-8",
    ) as f:
        dict_model_path = json.load(f)
    if args.model_name not in dict_model_path:
        raise KeyError(f"Model {args.model_name} not found in dict_model_path.json")
    args.model_path = dict_model_path[args.model_name]

    # Initialize the model and tokenizer
    tokenizer, model = load_model(args)

    # Record the time of the start
    time_start = datetime.datetime.now()
    # Time format: YYYY-MM-DD HH-MM-SS
    str_time_start = time_start.strftime("%Y-%m-%d-%H-%M-%S")

    # For different tasks
    for idx_task, task_name in enumerate(args.tasks):
        print(f"Task: {task_name}")
        # Set the task name
        args.task_name = task_name

        # Initialize the task and dataset
        task = GeneralTask(args=args, task_name=task_name)

        # Define the directory for the result
        args.path_dir_result = Path("result") / task_name / args.model_name
        args.path_dir_result.mkdir(parents=True, exist_ok=True)

        # Initialize the logger
        time_start_task = datetime.datetime.now()
        # time: YYYY-MM-DD HH-MM-SS
        str_time_start_task = time_start_task.strftime("%Y-%m-%d-%H-%M-%S")
        args.path_file_log = os.path.join(
            args.path_dir_result, f"{str_time_start_task}.log"
        )
        logger = init_logger(args.path_file_log)

        # Record data and model
        logger.info(f"Model: {args.model_name}: {args.model_path}")
        logger.info(f"Task: {task_name}")
        logger.info(
            f"Size: train={len(task.dataset_train)}, val={len(task.dataset_val)}, "
            f"test={len(task.dataset_test)}"
        )

        # Log the experiment parameters
        logger.info("\n" + "-" * 30 + "\n")
        logger.info("Experiment parameters:")
        for key, value in args.__dict__.items():
            if key not in ["tasks", "temperature", "top_p", "top_k"]:
                logger.info(f" - {key}: {value}")
        logger.info("\n" + "-" * 30 + "\n")

        logger.info(f"Start on: {str_time_start_task}")
        logger.info("\n" + "=" * 50 + "\n")

        # For different experiments, such as inference strategy or decoding parameter
        for idx_exp, (prompt_mode, decoding_strategy, seed) in enumerate(
            list_exp_config
        ):
            num_exp = idx_task * len(list_exp_config) + (idx_exp + 1)

            # Setup the inference strategy: "direct", "cot", "direct-n-shot"
            args.prompt_mode = prompt_mode
            task.setup(tokenizer, prompt_mode)

            # Initialize the dataloader
            dataloader = task.dataloader_test()

            # Set the seed
            args.seed = seed
            seed_everything(args.seed)

            # Set the decoding parameters
            args.decoding_strategy = decoding_strategy.lower()

            # Set experiment name and result path
            args.name_exp = (
                f"{task_name}-{args.prompt_mode}-{args.decoding_strategy}-{args.seed}"
            )
            args.path_file_result = os.path.join(
                args.path_dir_result, f"{args.name_exp}.result.json"
            )

            # Set the thread name with the name of experiment
            setproctitle.setproctitle(
                f"Benchmark: {args.model_name} {args.name_exp}-({num_exp}/{num_exp_all})"
            )

            # Log the experiment name
            logger.info(f"Name of experiment: {args.name_exp}")

            # Run: inference
            if args.inference_mode == "vllm":
                list_response = run_vllm(
                    args=args, logger=logger, dataloader=dataloader, model=model
                )
            else:
                list_response = run_hf(
                    args=args,
                    logger=logger,
                    dataloader=dataloader,
                    model=model,
                    tokenizer=tokenizer,
                )

            # Save results
            list_dict_data = save_result(
                args, logger, task.dataset_test, list_response, save_input=True
            )

            logger.info(f"Finished: {args.name_exp}")
            logger.info("\n" + "=" * 50 + "\n")

    # Record the time of the end
    time_end = datetime.datetime.now()
    str_time_end = time_end.strftime("%Y-%m-%d-%H-%M-%S")
    logger.info(f"End on: {str_time_end}")
    time_used = time_end - time_start
    logger.info(f"Time used: {time_used} seconds")
    logger.info("\n" + "=" * 50 + "\n")
