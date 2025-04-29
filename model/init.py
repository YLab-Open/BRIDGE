import os
import json
import torch
import random
import numpy as np
import transformers
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def load_config(args):
    args.num_workers = len(args.gpus) * 2

    # load the config for context window
    if "Qwen" in args.model_name or "Athene" in args.model_name.lower():
        path_file_config = os.path.join(args.model_path, "tokenizer_config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        args.max_token_all = dict_config["model_max_length"]
    elif "BioMistral-7B" == args.model_name:
        args.max_token_all = 2048
    else:
        path_file_config = os.path.join(args.model_path, "config.json")
        with open(path_file_config, "r", encoding="utf-8") as f:
            dict_config = json.load(f)
        if "max_position_embeddings" in dict_config:
            args.max_token_all = dict_config["max_position_embeddings"]
        elif "text_config" in dict_config:
            args.max_token_all = dict_config["text_config"]["max_position_embeddings"]
        else:
            args.max_token_all = 128 * 1000

    # Set the max token all
    args.max_token_all = (
        128 * 1000 if args.max_token_all > 128 * 1000 else args.max_token_all
    )

    # Several model only support the short context window
    if args.model_name in [
        # Only 2048 tokens
        "meditron-7b",
        "BioMistral-7B",
        # Only 4096 tokens
        "meditron-70b",
        "MeLLaMA-13B-chat",
        "MeLLaMA-70B-chat",
        # Only 8192 tokens
        # "MMed-Llama-3-8B",
        # "gemma-2-9b-it",
        # "gemma-2-27b-it",
        # "Llama3-OpenBioLLM-8B",
        # "Llama3-OpenBioLLM-70B",
        # "MMed-Llama-3-8B",
    ]:
        args.max_token_output = 512
    else:
        args.max_token_output = int(args.max_token_all * 0.125)

    # Set the max token output
    args.max_token_output = (
        3 * 1024 if args.max_token_output > 3 * 1024 else args.max_token_output
    )

    # Set the max token input
    args.max_token_input = int(args.max_token_all - args.max_token_output)


def load_model(args):
    # load the config for model and decoding
    load_config(args)

    # load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )
    if args.inference_mode == "vllm":
        if (
            "mistral" in args.model_name.lower()
            and "biomistral" not in args.model_name.lower()
        ):
            model = LLM(
                model=args.model_path,
                tensor_parallel_size=len(args.gpus),
                dtype="bfloat16",
                trust_remote_code=True,
                max_model_len=args.max_token_all,
                gpu_memory_utilization=0.93,
                tokenizer_mode="mistral",
                load_format="mistral",
                config_format="mistral",
            )
        else:
            model = LLM(
                model=args.model_path,
                tensor_parallel_size=len(args.gpus),
                dtype="bfloat16",
                max_model_len=args.max_token_all,
                gpu_memory_utilization=0.93,
                trust_remote_code=True,
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return tokenizer, model


def model_config_hf(args, logger, model, tokenizer):
    # Decoding strategy
    logger.info(f"Decoding strategy (HF): {args.decoding_strategy}")
    # Set the decoding strategy and parameters
    if args.decoding_strategy == "greedy":
        model.generation_config.do_sample = False
        model.generation_config.temperature = 0
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    else:
        model.generation_config.do_sample = True
        model.generation_config.temperature = args.temperature
        model.generation_config.top_p = args.top_p
        model.generation_config.top_k = args.top_k
    # Print the decoding strategy
    logger.info(f"Do_sample: {model.generation_config.do_sample}")
    logger.info(f"Temperature: {model.generation_config.temperature}")
    logger.info(f"Top_p: {model.generation_config.top_p}")
    logger.info(f"Top_k: {model.generation_config.top_k}")
    logger.info(f"Max tokens all: {args.max_token_all}")
    logger.info(f"Max tokens input: {args.max_token_input}")
    logger.info(f"Max tokens output: {args.max_token_output}")

    # Tokenizer setting
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        logger.info(f"pad_token_id is already set: {tokenizer.pad_token_id}")
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.eos_token_id
    else:
        logger.info(
            f"pad_token_id is already set: {model.generation_config.pad_token_id}"
        )

    return tokenizer, model


def model_config_vllm(args, logger):
    # set the decoding strategy and parameters
    logger.info(f"Decoding strategy (vLLM): {args.decoding_strategy}")
    # Set the decoding strategy and parameters
    if args.decoding_strategy == "greedy":
        sampling_params = SamplingParams(
            seed=args.seed, temperature=0, max_tokens=args.max_token_output
        )
        # Print the decoding strategy
        logger.info(f"Do_sample: False")
        logger.info(f"Temperature: 0")
        logger.info(f"Top_p: None")
        logger.info(f"Top_k: None")
        logger.info(f"Max tokens all: {args.max_token_all}")
        logger.info(f"Max tokens input: {args.max_token_input}")
        logger.info(f"Max tokens output: {args.max_token_output}")

    else:
        sampling_params = SamplingParams(
            seed=args.seed,
            temperature=args.temperature,
            max_tokens=args.max_token_output,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        # Print the decoding strategy
        logger.info(f"Do_sample: True")
        logger.info(f"Temperature: {args.temperature}")
        logger.info(f"Top_p: {args.top_p}")
        logger.info(f"Top_k: {args.top_k}")
        logger.info(f"Max tokens all: {args.max_token_all}")
        logger.info(f"Max tokens input: {args.max_token_input}")
        logger.info(f"Max tokens output: {args.max_token_output}")

    return sampling_params
