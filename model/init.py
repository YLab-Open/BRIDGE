import os
import json
import torch
import random
import numpy as np
import transformers
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from util.tool import set_default_args


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


def get_token_config(args):
    # load the config for context window
    if not hasattr(args, "max_token_all"):
        if "Qwen" in args.model_name or "Athene" in args.model_name.lower():
            path_file_config = os.path.join(args.model_path, "tokenizer_config.json")
            with open(path_file_config, "r", encoding="utf-8") as f:
                dict_config = json.load(f)
            max_token_all = dict_config["model_max_length"]
        elif "BioMistral-7B" == args.model_name:
            max_token_all = 2048
        else:
            path_file_config = os.path.join(args.model_path, "config.json")
            with open(path_file_config, "r", encoding="utf-8") as f:
                dict_config = json.load(f)
            if "max_position_embeddings" in dict_config:
                max_token_all = dict_config["max_position_embeddings"]
            elif "text_config" in dict_config:
                max_token_all = dict_config["text_config"]["max_position_embeddings"]
            else:
                print(
                    f"Not assign max_token_all and can not find information in {path_file_config}"
                )
                return None
        print(
            f"Not assign max_token_all, but found the information in model config: {max_token_all}"
        )
    else:
        max_token_all = args.max_token_all
        print(f"Manually set max_token_all: {max_token_all}")

    # Several model only support the short context window
    if not hasattr(args, "max_token_output"):
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
            max_token_output = 512
            print(f"Not assign max_token_output, set: {max_token_output}")
        else:
            max_token_output = int(max_token_all * 0.125)
            print(
                f"Not assign max_token_output, set: {max_token_output} for 1/8 of max_token_all"
            )
    else:
        max_token_output = args.max_token_output
        print(f"Manually set max_token_output: {max_token_output}")

    # Set the max token input
    if not hasattr(args, "max_token_input"):
        max_token_input = int(max_token_all - max_token_output)
        print(
            f"Not assign max_token_input, set: {max_token_input} for max_token_all - max_token_output"
        )
    else:
        max_token_input = args.max_token_input
        print(f"Manually set max_token_input: {max_token_input}")

    return max_token_all, max_token_input, max_token_output


def get_chat_config(args) -> dict:
    """Build chat_kwargs based on model name and args."""

    model_name = args.model_name.lower()

    # Initialize chat_kwargs
    kwargs = {}

    # ---- Qwen3 (Earlier version) ----
    if (
        model_name.startswith("qwen3")
        and "thinking" not in model_name
        and "instruct" not in model_name
    ):
        val, is_default = set_default_args(
            args, "enable_thinking", default=True, valid_set={True, False}
        )
        kwargs["enable_thinking"] = val
        print(
            f"[Qwen3] {'default' if is_default else 'manually set'} enable_thinking={val}"
        )

    # ---- gpt-oss ----
    elif model_name.startswith("gpt-oss"):
        val, is_default = set_default_args(
            args,
            "reasoning_effort",
            default="medium",
            valid_set={"low", "medium", "high"},
            cast=lambda x: str(x).lower(),
        )
        kwargs["reasoning_effort"] = val
        print(
            f"[gpt-oss] {'default' if is_default else 'manually set'} reasoning_effort={val}"
        )

    else:
        # For other models, no special chat_kwargs are needed
        pass

    return kwargs


def load_model(args):
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
                gpu_memory_utilization=0.95,
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
                gpu_memory_utilization=0.95,
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
    # Print the token limit
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

    # Print the token limit
    logger.info(f"Max tokens all: {args.max_token_all}")
    logger.info(f"Max tokens input: {args.max_token_input}")
    logger.info(f"Max tokens output: {args.max_token_output}")

    return sampling_params
