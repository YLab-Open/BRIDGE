import argparse
import datetime
import os
from pathlib import Path

import setproctitle

from util.logging import init_logger
from util.runtime import (
    get_task_dataloader,
    load_yaml_config,
    log_task,
    merge_config_into_args,
    normalize_runtime_args,
    validate_args,
)
from util.seed import seed_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="Meta-Llama-3.1-70B-Instruct",
        help="The name of the LLM model, only one model.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        default="configs/dict_model_path.json",
        help="Path to a JSON file containing model name to model path mapping.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="The gpus for the model.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default=None,
        help="Inference engine: vllm or hf.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/BRIDGE.yaml",
        help="Path to a YAML config file containing tasks, exp_configs, and other arguments.",
    )
    parser.add_argument(
        "--logits_config",
        type=str,
        default=None,
        help="Optional YAML config for synchronous logits/probability capture.",
    )

    args = parser.parse_args()

    # ---------- Load YAML & merge ----------
    dict_cfg = load_yaml_config(args.config)
    merge_config_into_args(args, parser, dict_cfg)
    validate_args(args)
    normalize_runtime_args(args)

    from dataset.dataset import GeneralTask
    from model.inference import (
        remove_partial_result,
        run_hf,
        run_vllm,
        save_result,
    )
    from model.init import init_model
    from model.logits import configure_logits_from_args, logit_output_paths
    from model.metadata import write_result_metadata

    configure_logits_from_args(args)

    # ---------- Experiment config ----------
    args.tasks.sort()
    list_exp_config = [tuple(item) for item in args.experiments]

    num_exp_all = len(args.tasks) * len(list_exp_config)

    # Initialize the model and tokenizer
    tokenizer, model = init_model(args)

    # Confirm whether the save_input was set in the config file
    if not hasattr(args, "save_input"):
        args.save_input = True
        print("Not assign save_input, set to True by default")
    else:
        print(f"Manually set save_input to: {args.save_input}")

    # Set the name of the experiment
    setproctitle.setproctitle(f"Benchmark: {args.model_name}-{num_exp_all} runs.")

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
        args.task_data_path = task.path_file_data

        # Define the directory for the result
        args.path_dir_result = Path(args.dir_result_save) / task_name / args.model_name
        args.path_dir_result.mkdir(parents=True, exist_ok=True)

        # Initialize the logger
        time_start_task = datetime.datetime.now()
        # time: YYYY-MM-DD HH-MM-SS
        str_time_start_task = time_start_task.strftime("%Y-%m-%d-%H-%M-%S")
        args.path_file_log = os.path.join(
            args.path_dir_result, f"{str_time_start_task}.log"
        )
        logger = init_logger(args.path_file_log)

        # Log the task and experiment information
        log_task(args, logger, task_name, task, str_time_start_task)

        # For different experiments, such as inference strategy or decoding parameter
        for idx_exp, (prompt_mode, decoding_strategy, seed) in enumerate(
            list_exp_config
        ):
            num_exp = idx_task * len(list_exp_config) + (idx_exp + 1)

            # Setup the inference strategy: "direct", "cot", "direct-n-shot"
            args.prompt_mode = prompt_mode
            task.setup(tokenizer, prompt_mode)

            # Initialize the selected split dataloader.
            dataset, dataloader = get_task_dataloader(
                task=task,
                mode=args.mode,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
            )
            logger.info(f"Running in {args.mode} mode, loading {args.mode} dataloader.")
            logger.info(f"Dataset size: {len(dataset)} samples")
            logger.info(f"DataLoader size: {len(dataloader)} batches")

            # Set the seed
            args.seed = seed_inference(seed)
            logger.info(f"Inference seed: {args.seed}")

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
            if getattr(args, "logits_enabled", False):
                logit_paths = logit_output_paths(args.path_file_result)
                logger.info(f"Result path: {args.path_file_result}")
                logger.info(f"Logit sidecar path: {logit_paths['logit_json']}")
                logger.info(f"Metadata path: {logit_paths['metadata']}")

            # Run: inference
            if args.engine == "vllm":
                list_response = run_vllm(
                    args=args,
                    logger=logger,
                    dataset=dataset,
                    dataloader=dataloader,
                    model=model,
                    tokenizer=tokenizer,
                )
                list_generation = None
                list_logit = None
                if isinstance(list_response, tuple):
                    list_response, list_generation, list_logit = list_response
            else:
                list_response = run_hf(
                    args=args,
                    logger=logger,
                    dataloader=dataloader,
                    model=model,
                    tokenizer=tokenizer,
                )
                list_generation = None
                list_logit = None

            # Save results
            logit_paths = logit_output_paths(args.path_file_result)
            save_result(
                logger=logger,
                dataset=dataset,
                list_response=list_response,
                path_file_save=args.path_file_result,
                save_input=args.save_input,
                list_generation=list_generation,
                list_logit=list_logit,
                path_file_logit_save=logit_paths["logit_json"],
            )
            remove_partial_result(args.path_file_result)
            effective_config = {
                key: value
                for key, value in vars(args).items()
                if not key.startswith("_") and not callable(value)
            }
            metadata_paths = write_result_metadata(
                result_path=args.path_file_result,
                run_type=f"local_{args.engine}",
                n_records=len(list_response),
                effective_config=effective_config,
                source_files={
                    "main_config": args.config,
                    "logits_config": getattr(args, "logits_config", None),
                    "model_registry": args.model_file,
                    "task_data": args.task_data_path,
                },
                related_files={
                    "log": args.path_file_log,
                    "logit": (
                        logit_paths["logit_json"]
                        if list_generation is not None or list_logit is not None
                        else None
                    ),
                    "top_logits": (
                        logit_paths["top_logits"]
                        if getattr(args, "logits_enabled", False)
                        else None
                    ),
                },
                experiment={
                    "model_name": args.model_name,
                    "model_path": args.model_path,
                    "task_name": task_name,
                    "prompt_mode": prompt_mode,
                    "decoding_strategy": args.decoding_strategy,
                    "seed": args.seed,
                    "split": args.mode,
                },
                extra={"logits": getattr(args, "logits", {})},
            )
            logger.info(f"Metadata path: {metadata_paths['metadata']}")
            logger.info(f"Config snapshot: {metadata_paths['config']}")

            logger.info(f"Finished: {args.name_exp}")
            logger.info("\n" + "=" * 50 + "\n")

    # Record the time of the end
    time_end = datetime.datetime.now()
    str_time_end = time_end.strftime("%Y-%m-%d-%H-%M-%S")
    logger.info(f"End on: {str_time_end}")
    time_used = time_end - time_start
    logger.info(f"Time used: {time_used} seconds")
    logger.info("\n" + "=" * 50 + "\n")
