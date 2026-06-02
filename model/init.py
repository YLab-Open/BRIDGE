import os
import json
import torch
import random
import numpy as np
from numbers import Number
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from util.local_config import resolve_model_path
from util.tool import set_default_args


SAMPLING_DEFAULTS = {
    "sample_size": 1,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}


def seed_everything(seed):
    """Seed Python, NumPy, PyTorch, CUDA, and Transformers for one experiment."""
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
    """
    Resolve model context-window, input, and output token budgets.

    Explicit `args.max_token_*` values take priority. When they are missing, the
    function reads tokenizer/model config files from `args.model_path` and keeps
    a small set of model-family overrides for known short-context checkpoints.
    """
    if not hasattr(args, "max_token_all"):
        if hasattr(args, "max_token_input") and hasattr(args, "max_token_output"):
            max_token_all = args.max_token_input + args.max_token_output
            print(f"Manually set max_token_all: {max_token_all}")
        else:
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

    # ---- gemma-4 ----
    elif model_name.startswith("gemma-4"):
        val, is_default = set_default_args(
            args, "enable_thinking", default=True, valid_set={True, False}
        )
        kwargs["enable_thinking"] = val
        print(
            f"[gemma-4] {'default' if is_default else 'manually set'} enable_thinking={val}"
        )

    else:
        # For other models, no special chat_kwargs are needed
        pass

    return kwargs


def init_model(args):
    """
    Resolve local model assets, runtime token settings, tokenizer, and model.

    The model-path mapping is loaded through `util.local_config` so notebooks,
    scripts, and tests share the same validation behavior for local-only config.
    """
    args.model_path = resolve_model_path(args.model_name, args.model_file)

    # set the config for model and decoding
    args.max_token_all, args.max_token_input, args.max_token_output = get_token_config(
        args
    )
    # Set the process kwargs for apply_chat_template()
    args.chat_kwargs = get_chat_config(args)

    # Initialize the model and tokenizer
    tokenizer, model = load_model(args)

    # Add the suffix for models with different thinking mode
    #  - For earlier Qwen3 models
    if "enable_thinking" in args.chat_kwargs:
        if args.chat_kwargs["enable_thinking"]:
            args.model_name += "-Thinking"
        else:
            args.model_name += "-Non-Thinking"
    #  - For gpt-oss models
    if "reasoning_effort" in args.chat_kwargs:
        args.model_name += f"-{args.chat_kwargs['reasoning_effort']}"

    return tokenizer, model


def load_model(args):
    """Load the tokenizer and model for the selected inference engine."""
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, padding_side="left", trust_remote_code=True
    )
    if args.engine == "vllm":
        from vllm import LLM

        if not hasattr(args, "gpu_memory_util"):
            args.gpu_memory_util = 0.95
            print("Not assign gpu_memory_util, set to 0.95")
        else:
            print(f"Manually set gpu_memory_util: {args.gpu_memory_util}")

        vllm_kwargs = {
            "model": args.model_path,
            "tensor_parallel_size": len(args.gpus),
            # "dtype": "bfloat16",
            "max_model_len": args.max_token_all,
            "gpu_memory_utilization": args.gpu_memory_util,
        }
        optional_vllm_kwargs = [
            "max_num_seqs",
            "max_num_batched_tokens",
        ]
        for key in optional_vllm_kwargs:
            if hasattr(args, key):
                vllm_kwargs[key] = getattr(args, key)
                print(f"Set vLLM {key}: {vllm_kwargs[key]}")

        if (
            "mistral" in args.model_name.lower()
            and "biomistral" not in args.model_name.lower()
        ):
            vllm_kwargs.update(
                {
                    "tokenizer_mode": "mistral",
                    "load_format": "mistral",
                    "config_format": "mistral",
                }
            )
            model = LLM(**vllm_kwargs)
        else:
            model = LLM(**vllm_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            # torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    return tokenizer, model


def load_generation_config(args, logger):
    """
    Load generation_config.json from the local model directory.

    Missing files are common for older checkpoints, so the function returns an
    empty dict instead of failing. Malformed files are logged and ignored.
    """
    dict_config_gen = {}
    model_path = getattr(args, "model_path", None)
    if not model_path:
        logger.warning("model_path is not set; skip loading generation_config.json.")
        return dict_config_gen

    path_file_config = os.path.join(model_path, "generation_config.json")
    if os.path.exists(path_file_config):
        try:
            with open(path_file_config, "r", encoding="utf-8") as f:
                dict_config_gen = json.load(f)
        except Exception as e:
            logger.warning(
                f"Failed to load generation_config.json from {path_file_config}: {e}"
            )
    else:
        logger.info(f"generation_config.json not found at: {path_file_config}")

    return dict_config_gen


def _generation_config_value(dict_config_gen, field_name):
    """Return a generation_config value and its source key for one runtime field."""
    aliases = {
        "sample_size": ("sample_size", "num_return_sequences"),
    }
    for key in aliases.get(field_name, (field_name,)):
        if key in dict_config_gen and dict_config_gen[key] is not None:
            return dict_config_gen[key], key
    return None, None


def _coerce_sampling_value(field_name, value):
    """Coerce and validate a sampling value loaded from YAML or generation_config."""
    if field_name in {"sample_size", "top_k"}:
        value = int(value)
        if field_name == "sample_size" and value < 1:
            raise ValueError("sample_size must be >= 1.")
        if field_name == "top_k" and value < -1:
            raise ValueError("top_k must be >= -1.")
        return value

    if field_name in {
        "temperature",
        "top_p",
        "min_p",
        "repetition_penalty",
        "presence_penalty",
        "frequency_penalty",
    }:
        if not isinstance(value, Number):
            value = float(value)
        value = float(value)
        if field_name == "temperature" and value < 0:
            raise ValueError("temperature must be >= 0.")
        if field_name == "top_p" and not (0 < value <= 1):
            raise ValueError("top_p must be in (0, 1].")
        if field_name == "min_p" and not (0 <= value <= 1):
            raise ValueError("min_p must be in [0, 1].")
        if field_name == "repetition_penalty" and value <= 0:
            raise ValueError("repetition_penalty must be > 0.")
        return value

    return value


def _resolve_sampling_value(args, field_name, dict_config_gen, logger):
    """
    Resolve one sampling parameter with this priority:
    YAML explicit value > generation_config.json > code default.
    """
    manual_keys = getattr(args, "_manual_config_keys", set())
    if field_name in manual_keys and getattr(args, field_name, None) is not None:
        raw_value = getattr(args, field_name)
        source = "YAML"
    else:
        raw_value, source_key = _generation_config_value(dict_config_gen, field_name)
        if source_key:
            source = f"generation_config.json:{source_key}"
        else:
            raw_value = SAMPLING_DEFAULTS[field_name]
            source = "default"

    try:
        value = _coerce_sampling_value(field_name, raw_value)
    except (TypeError, ValueError) as e:
        if source == "YAML":
            raise ValueError(f"Invalid YAML value for {field_name}: {raw_value}") from e
        original_source = source
        value = SAMPLING_DEFAULTS[field_name]
        source = f"default after invalid {original_source}"
        logger.warning(
            f"Invalid {original_source} value for {field_name}: {raw_value}. "
            f"Use default {value}."
        )

    setattr(args, field_name, value)
    logger.info(f"Set {field_name}: {value} from {source}")
    return value


def get_sampling_params(args, logger):
    """
    Prepare sampling-related parameters on args (for non-greedy decoding).

    For sampling strategy, the priority is:
    1) Use the value in args if present.
    2) Else use generation_config.json if present.
    3) Else fall back to a hard-coded default.

    Default sample_size is 1 so result files remain directly evaluable unless
    the YAML explicitly requests multiple samples per prompt.
    """
    dict_config_gen = load_generation_config(args, logger)
    for field_name in SAMPLING_DEFAULTS:
        _resolve_sampling_value(args, field_name, dict_config_gen, logger)

    return args


def log_token_limits(args, logger):
    """
    Log token limit related runtime arguments.
    """
    logger.info(f"Max tokens all: {getattr(args, 'max_token_all', None)}")
    logger.info(f"Max tokens input: {getattr(args, 'max_token_input', None)}")
    logger.info(f"Max tokens output: {getattr(args, 'max_token_output', None)}")


def configure_padding(tokenizer, model, logger):
    """
    Ensure pad_token_id is set for both tokenizer and model.generation_config.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        logger.info(
            f"Set tokenizer.pad_token_id to eos_token_id: {tokenizer.pad_token_id}"
        )
    else:
        logger.info(f"tokenizer.pad_token_id is already set: {tokenizer.pad_token_id}")

    if getattr(model.generation_config, "pad_token_id", None) is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        logger.info(
            f"Set generation_config.pad_token_id: {model.generation_config.pad_token_id}"
        )
    else:
        logger.info(
            f"generation_config.pad_token_id is already set: {model.generation_config.pad_token_id}"
        )


def model_config_hf(args, logger, model, tokenizer):
    """
    Configure a HuggingFace model and tokenizer for decoding.

    - Greedy:
        do_sample = False
        temperature = 0.0
        top_p = 1.0
        top_k = 0
        repetition_penalty = 1.0
    - Sampling:
        do_sample = True
        temperature, top_p, top_k, repetition_penalty are taken from
        args / generation_config / defaults via get_sampling_params().
    """
    logger.info(f"Decoding strategy (HF): {args.decoding_strategy}")

    if args.decoding_strategy == "greedy":
        # Greedy decoding: no sampling
        model.generation_config.do_sample = False
        model.generation_config.temperature = 0.0
        model.generation_config.top_p = 1.0
        model.generation_config.top_k = 0
        # For greedy, repetition_penalty is effectively 1.0
        rep_penalty = getattr(args, "repetition_penalty", 1.0)
        model.generation_config.repetition_penalty = rep_penalty
    else:
        # Sampling-based decoding
        get_sampling_params(args, logger)
        if args.temperature == 0:
            logger.warning(
                "HF sampling received temperature=0. Use greedy decoding for this run "
                "because transformers requires a positive sampling temperature."
            )
            model.generation_config.do_sample = False
        else:
            model.generation_config.do_sample = True
        model.generation_config.temperature = args.temperature
        model.generation_config.top_p = args.top_p
        model.generation_config.top_k = 0 if args.top_k == -1 else args.top_k
        model.generation_config.repetition_penalty = getattr(
            args, "repetition_penalty", 1.0
        )

    # Log final decoding hyperparameters
    logger.info(f"Do_sample: {model.generation_config.do_sample}")
    logger.info(f"Temperature: {model.generation_config.temperature}")
    logger.info(f"Top_p: {model.generation_config.top_p}")
    logger.info(f"Top_k: {model.generation_config.top_k}")
    logger.info(f"Repetition_penalty: {model.generation_config.repetition_penalty}")

    # Token limits
    log_token_limits(args, logger)

    # Tokenizer / model padding settings
    configure_padding(tokenizer, model, logger)

    return tokenizer, model


def model_config_vllm(args, logger):
    """
    Configure SamplingParams for vLLM.

    - Greedy:
        temperature = 0.0
        (top_p, top_k use vLLM defaults; this corresponds to greedy)
        repetition_penalty = 1.0 (or args.repetition_penalty if provided)
    - Sampling:
        n, temperature, top_p, top_k, repetition_penalty come from
            args / generation_config / defaults via get_sampling_params().
    """
    from vllm import SamplingParams

    logger.info(f"Decoding strategy (vLLM): {args.decoding_strategy}")

    if args.decoding_strategy == "greedy":
        # Greedy decoding for vLLM: temperature=0.0 implies greedy behavior.
        sampling_kwargs = {
            "seed": args.seed,
            "temperature": 0.0,
            "max_tokens": args.max_token_output,
        }
        if (
            args.model_name.lower().startswith("gemma-4")
            and getattr(args, "chat_kwargs", {}).get("enable_thinking")
        ):
            sampling_kwargs["skip_special_tokens"] = False
            logger.info(
                "Set skip_special_tokens=False for Gemma4 thinking output parsing."
            )
        sampling_params = SamplingParams(**sampling_kwargs)
        logger.info("Do_sample: False (greedy via temperature=0.0)")
        logger.info("Temperature: 0.0")
        logger.info("Top_p: default (1.0)")
        logger.info("Top_k: default (0)")
        rep_penalty = getattr(args, "repetition_penalty", 1.0)
        sampling_params.repetition_penalty = rep_penalty
        logger.info(f"Repetition_penalty: {rep_penalty}")
    else:
        # Sampling-based decoding
        get_sampling_params(args, logger)
        if args.temperature == 0 and args.sample_size > 1:
            logger.warning(
                "vLLM sampling received temperature=0 with sample_size>1. "
                "The generated samples are expected to be deterministic duplicates."
            )
        sampling_kwargs = {
            "n": args.sample_size,
            "seed": args.seed,
            "temperature": args.temperature,
            "max_tokens": args.max_token_output,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "min_p": args.min_p,
            "repetition_penalty": args.repetition_penalty,
            "presence_penalty": args.presence_penalty,
            "frequency_penalty": args.frequency_penalty,
        }
        if (
            args.model_name.lower().startswith("gemma-4")
            and getattr(args, "chat_kwargs", {}).get("enable_thinking")
        ):
            sampling_kwargs["skip_special_tokens"] = False
            logger.info(
                "Set skip_special_tokens=False for Gemma4 thinking output parsing."
            )
        sampling_params = SamplingParams(**sampling_kwargs)
        rep_penalty = args.repetition_penalty

        logger.info("Do_sample: True")
        logger.info(f"Temperature: {args.temperature}")
        logger.info(f"Top_p: {args.top_p}")
        logger.info(f"Top_k: {args.top_k}")
        logger.info(f"Min_p: {args.min_p}")
        logger.info(f"Repetition_penalty: {rep_penalty}")
        logger.info(f"Presence_penalty: {args.presence_penalty}")
        logger.info(f"Frequency_penalty: {args.frequency_penalty}")

    # Token limits
    log_token_limits(args, logger)

    return sampling_params
