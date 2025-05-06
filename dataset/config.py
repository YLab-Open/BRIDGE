import numpy as np

list_str_templates_direct = [
    "Return your answer in the following format. DO NOT GIVE ANY EXPLANATION:",
]

list_str_templates_cot = [
    "Solve it in a step-by-step fashion, return your answer in the following format, PROVIDE DETAILED ANALYSIS BEFORE THE RESULT:",
]
str_format_cot = """\nAnalysis:
...
Result:"""

list_str_templates_cot = [
    str_template + str_format_cot for str_template in list_str_templates_cot
]

assert len(list_str_templates_direct) == len(
    list_str_templates_cot
), "The number of Direct templates and CoT templates should be the same."


def get_tempalte_direct():
    """
    Get the template of Direct
    Returns:
        list: The list of Direct templates
    """
    return list_str_templates_direct


def get_tempalte_cot():
    """
    Get the template of CoT
    Returns:
        list: The list of CoT templates
    """
    return list_str_templates_cot


def transform_instruction_to_cot(instruction, flag_print=True):
    """
    Transform the instruction to CoT format
    Args:
        instruction (str): The instruction to be transformed
    Returns:
        str: The transformed instruction
    """
    for str_template_direct, str_template_cot in zip(
        list_str_templates_direct, list_str_templates_cot
    ):
        if str_template_direct in instruction:
            # Replace the Direct template with the CoT template
            return instruction.replace(
                str_template_direct,
                str_template_cot,
            )
    if flag_print:
        print(
            f"Warning: The instruction does not contain any of the Direct templates. Return None."
        )
    return None


def transform_instruction_to_direct(instruction, flag_print=True):
    """
    Transform the instruction to Direct format
    Args:
        instruction (str): The instruction to be transformed
    Returns:
        str: The transformed instruction
    """
    for str_template_direct, str_template_cot in zip(
        list_str_templates_direct, list_str_templates_cot
    ):
        if str_template_cot in instruction:
            # Replace the CoT template with the direct template
            return instruction.replace(
                str_template_cot,
                str_template_direct,
            )
    if flag_print:
        print(
            f"Warning: The instruction does not contain any of the CoT templates. Return None."
        )
    return None


def get_instruction_cot_from_direct(instruction):
    """
    Get the instruction from direct format to CoT format
    Args:
        instruction (str): The instruction to be transformed
    Returns:
        str: The matched template of CoT
        str: The matched template of Direct
    """
    for str_template_direct, str_template_cot in zip(
        list_str_templates_direct, list_str_templates_cot
    ):
        if str_template_direct in instruction:
            return str_template_direct, str_template_cot
    print(
        f"Warning: The instruction does not contain any of the Direct templates. Return None, None."
    )
    return None, None


def get_instruction_direct_from_cot(instruction):
    """
    Get the instruction from CoT format to direct format
    Args:
        instruction (str): The instruction to be transformed
    Returns:
        str: The matched template of CoT
        str: The matched template of Direct
    """
    for str_template_direct, str_template_cot in zip(
        list_str_templates_direct, list_str_templates_cot
    ):
        if str_template_cot in instruction:
            return str_template_direct, str_template_cot
    print(
        f"Warning: The instruction does not contain any of the CoT templates. Return None, None."
    )
    return None, None


def extract_cot_pred(str_response):
    list_cot_split_token = ["Result:"]
    for cot_split_token in list_cot_split_token:
        cot_split_token = cot_split_token.lower()
        if cot_split_token in str_response:
            str_response = str_response.split(cot_split_token, 1)
            return str_response[1].strip()
    return str_response


def get_metrics_clf():
    return ["accuracy", "f1_macro", "f1_micro", "num_failed_ratio"]


def get_metrics_gen():
    return ["bleu", "rouge", "bertscore", "num_failed_ratio"]


def get_metrics_ext():
    return ["f1_subject", "f1_event", "num_failed_ratio"]


def get_metrics_ext_qa():
    return ["exact_match", "overlap_match"]


def get_pred_none_clf(list_pred, list_label):
    # count the number of invalid response
    num_failed = sum([1 for pred in list_pred if pred == -1])
    # get the valid labels
    labels_valid = list(set(list_label))
    # random label
    list_pred = [
        np.random.choice(labels_valid) if pred == -1 else pred for pred in list_pred
    ]

    return list_pred, num_failed


def get_pred_none_clf_mul_label(list_list_pred, list_list_label):
    # count the number of invalid response
    num_failed = sum([1 for pred in list_list_pred if -1 in pred])
    # get the valid labels
    labels_valid = list(
        set([label for list_label in list_list_label for label in list_label])
    )
    # random one label for multi-label classification
    list_list_pred = [
        [np.random.choice(labels_valid)] if -1 in list_pred else list_pred
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed


def get_pred_none_clf_mul_question(list_list_pred, list_list_label):
    # count the number of invalid response
    num_failed = sum([1 for list_pred in list_list_pred if -1 in list_pred])
    num_question = len(list_list_label[0])
    # get the unique label for each question
    dict_idx_label = {
        idx: list(set([list_label[idx] for list_label in list_list_label]))
        for idx in range(num_question)
    }

    # random label
    list_list_pred = [
        [
            np.random.choice(dict_idx_label[idx]) if pred == -1 else pred
            for idx, pred in enumerate(list_pred)
        ]
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed


def get_pred_none_ext(list_pred):
    # return the empty list
    num_failed = sum([1 for pred in list_pred if -1 in pred])
    list_pred = [[] if -1 in pred else pred for pred in list_pred]

    return list_pred, num_failed


def get_pred_none_gen(list_pred):
    # return the empty string
    num_failed = sum([1 for pred in list_pred if pred == -1])
    list_pred = ["" if pred == -1 else pred for pred in list_pred]

    return list_pred, num_failed


def get_pred_none_gen_qa_mul(list_list_pred):
    # return the empty string
    num_failed = sum([1 for list_pred in list_list_pred if -1 in list_pred])
    list_list_pred = [
        ["" if pred == -1 else pred for pred in list_pred]
        for list_pred in list_list_pred
    ]

    return list_list_pred, num_failed


def get_models_evaluate():
    list_model = list_model = [
        "Baichuan-M1-14B-Instruct",
        "DeepSeek-R1",
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Llama-70B",
        "DeepSeek-R1-Distill-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
        "DeepSeek-R1-Distill-Qwen-32B",
        "gemma-2-9b-it",
        "gemma-2-27b-it",
        "gemma-3-1b-it",
        "gemma-3-4b-it",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "Llama-3.1-8B-Instruct",
        "Llama-3.1-70B-Instruct",
        "Llama-3.2-1B-Instruct",
        "Llama-3.2-3B-Instruct",
        "Llama-3.3-70B-Instruct",
        "Llama-4-Scout-17B-16E-Instruct",
        "Llama-3.1-Nemotron-70B-Instruct-HF",
        "meditron-7b",
        "meditron-70b",
        "MeLLaMA-13B-chat",
        "MeLLaMA-70B-chat",
        "Llama3-OpenBioLLM-8B",
        "Llama3-OpenBioLLM-70B",
        "MMed-Llama-3-8B",
        "Llama-3.1-8B-UltraMedical",
        "Llama-3-70B-UltraMedical",
        "Ministral-8B-Instruct-2410",
        "Mistral-Small-Instruct-2409",
        "Mistral-Small-24B-Instruct-2501",
        "Mistral-Small-3.1-24B-Instruct-2503",
        "Mistral-Large-Instruct-2411",
        "BioMistral-7B",
        "Phi-3.5-mini-instruct",
        "Phi-3.5-MoE-instruct",
        "Phi-4",
        "Qwen2.5-1.5B-Instruct",
        "Qwen2.5-3B-Instruct",
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen3-0.6B-Thinking",
        "Qwen3-1.7B-Thinking",
        "Qwen3-4B-Thinking",
        "Qwen3-8B-Thinking",
        "Qwen3-14B-Thinking",
        "Qwen3-32B-Thinking",
        "Qwen3-30B-A3B-Thinking",
        "Qwen3-235B-A22B-Thinking",
        "Qwen3-0.6B-Non-Thinking",
        "Qwen3-1.7B-Non-Thinking",
        "Qwen3-4B-Non-Thinking",
        "Qwen3-8B-Non-Thinking",
        "Qwen3-14B-Non-Thinking",
        "Qwen3-32B-Non-Thinking",
        "Qwen3-30B-A3B-Non-Thinking",
        "Qwen3-235B-A22B-Non-Thinking",
        "QwQ-32B-Preview",
        "QWQ-32B",
        "Athene-V2-Chat",
        "Yi-1.5-9B-Chat-16K",
        "Yi-1.5-34B-Chat-16K",
        "gpt-35-turbo",
        "gpt-4o",
        "gemini-2.0-flash",
        "gemini-1.5-pro",
    ]

    return list_model
