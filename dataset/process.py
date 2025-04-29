def process_punc_to_en(text):
    """
    Converts common punctuation into English version.
    Input:
        text: str
    Output:
        text: str
    """
    # define the mapping
    FULL_ANGLE_ALPHABET = r"""　“‘”’，。：；＂＃＄％＆＇＊＋－．／０１２３４５６７８９＜＝＞＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝（）～"""
    HALF_ANGLE_ALPHABET = r""" "'"',.:;"#$%&'*+-./0123456789<=>@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}()~"""

    translation_table = str.maketrans(FULL_ANGLE_ALPHABET, HALF_ANGLE_ALPHABET)

    return text.translate(translation_table)


def process_text_clean(text, flag_lower=True, flag_punc_to_en=True):
    """
    Clean the text.
    Input:
        text: str
        flag_lower: bool
        flag_punc_to_en: bool
    Output:
        text: str
    """
    if flag_lower:
        text = text.lower()
    if flag_punc_to_en:
        text = process_punc_to_en(text)
    return text.strip()


def get_truncate_text(text, tokenizer, max_tokens):
    """Helper function to truncate text to a specified token length."""
    tokenized = tokenizer.encode(text)
    if len(tokenized) > max_tokens:
        return tokenizer.decode(tokenized[:max_tokens], skip_special_tokens=True) + "\n"
    return text


def num_tokens_from_messages(messages, tokenizer):
    """Return the number of tokens used by a list of messages."""
    token_template = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    token_template_count = len(token_template)
    return token_template_count


def get_truncate_text_openai(text, tokenizer, max_tokens):
    """Helper function to truncate text to a specified token length."""
    tokenized = tokenizer.encode(text)
    # The OpenAI tokenizer does not have a `skip_special_tokens` argument
    # and it also will not add special tokens
    if len(tokenized) > max_tokens:
        return tokenizer.decode(tokenized[:max_tokens]) + "\n"
    return text


def get_truncate_text_gemini(text, tokenizer, max_tokens):
    """Helper function to truncate text to a specified token length."""
    total_tokens = tokenizer.count_tokens(text).total_tokens
    if total_tokens > max_tokens:
        truncated_tokens_list = tokenizer.compute_tokens(text).tokens_info[0].tokens[
            : max_tokens - 1
        ] + [b"\n"]
        truncated_tokens_list = [
            character.decode("utf-8") for character in truncated_tokens_list
        ]
        return "".join(truncated_tokens_list)
    return text


def get_truncate_text_deepseek(text, tokenizer, max_tokens):
    """Helper function to truncate text to a specified token length."""
    tokenized = tokenizer.encode(text)
    if len(tokenized) > max_tokens:
        return tokenizer.decode(tokenized[: max_tokens - 1]) + "\n"
    return text


def num_tokens_from_messages_openai(messages, tokenizer):
    """Return the number of tokens used by a list of messages."""
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(tokenizer.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    # every reply is primed with <|start|>assistant<|message|>
    num_tokens += 3
    return num_tokens


def num_tokens_from_messages_gemini(messages, tokenizer):
    """Return the number of tokens used by a list of messages."""
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            if key == "parts":
                contents = value[0]["text"]
                num_tokens += tokenizer.count_tokens(contents).total_tokens
    return num_tokens


def num_tokens_from_messages_deepseek(messages, tokenizer):
    """Return the number of tokens used by a list of messages."""
    num_tokens = 0
    for message in messages:
        for key, value in message.items():
            if key == "text":
                contents = value
                num_tokens += len(tokenizer.encode(contents))
    return num_tokens


def estimate_token_rough(
    input_system, input_user, language, times_word_token=7, examples=None
):
    """
    Calculate the rough number of tokens.

    Args:
        input_system: System instruction.
        input_user: User input.

    Returns:
        int: The rough number of tokens.
    """
    if examples:
        str_example = " ".join(
            [
                dict_example["input"] + dict_example["output"]
                for dict_example in examples
            ]
        )
        str_prompt = input_system + str_example + input_user
    else:
        str_prompt = input_system + input_user

    if language in ["zh", "ja", "cn", "jp"]:
        return len(str_prompt) * times_word_token
    else:
        return len(str_prompt.split()) * times_word_token


def format_chat(
    model_name,
    tokenizer,
    data,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
):
    """
    Format the input for different models based on the prompt mode.
    Args:
        model_name: The model's name.
        tokenizer: The tokenizer object for processing text.
        data: The input data.
        max_token_input: The maximum token limit for the input.
        times_word_token: The factor for rough token calculation.
        num_token_reserve: The number of tokens to reserve for formatting.
        examples: Optional list of examples, each a dict with "input" and "output".

    Returns:
        The formatted input text for large language models.

    """

    input_system = data["instruction"]
    input_user = data["input"]
    language = data["language"]

    if "meditron" in model_name.lower():
        example_ids, formatted_input = format_chat_meditron(
            model_name=model_name,
            tokenizer=tokenizer,
            language=language,
            input_system=input_system,
            input_user=input_user,
            max_token_input=max_token_input,
            times_word_token=times_word_token,
            num_token_reserve=num_token_reserve,
            examples=examples,
        )
    elif "mellama" in model_name.lower():
        example_ids, formatted_input = format_chat_mellama(
            model_name=model_name,
            tokenizer=tokenizer,
            language=language,
            input_system=input_system,
            input_user=input_user,
            max_token_input=max_token_input,
            times_word_token=times_word_token,
            num_token_reserve=num_token_reserve,
            examples=examples,
        )
    else:
        example_ids, formatted_input = format_chat_general(
            model_name=model_name,
            tokenizer=tokenizer,
            language=language,
            input_system=input_system,
            input_user=input_user,
            max_token_input=max_token_input,
            times_word_token=times_word_token,
            num_token_reserve=num_token_reserve,
            examples=examples,
        )

    if examples:
        data["example_ids"] = example_ids

    return formatted_input


def format_message(input_system, input_user, examples):
    formatted_chat = [{"role": "system", "content": input_system}]
    for ex in examples:
        ex_chat = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        formatted_chat.extend(ex_chat)
    formatted_chat.append({"role": "user", "content": input_user})
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_chat


def get_formatted_chat(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
    flag_openai=False,
):
    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_chat = format_message(input_system, input_user, examples)
        return example_ids, formatted_chat

    # Prepare the token counting functions based on the model type
    if flag_openai:
        # For OpenAI models, use a custom function to count tokens
        get_token_func = num_tokens_from_messages_openai
        truncate_text_func = get_truncate_text_openai
    else:
        # For general models, use the tokenizer and apply_chat_template to count tokens
        get_token_func = num_tokens_from_messages
        truncate_text_func = get_truncate_text

    # Step 2: Handle truncation when exceeding token limit
    _, base_chat = format_message(input_system, input_user, [])
    base_token_count = get_token_func(base_chat, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - len(tokenizer.encode(input_user)))
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        truncated_user = truncate_text_func(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message(input_system, truncated_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    examples_included = []
    for ex in examples:
        # Construct the example as a chat sequence (user->assistant)
        chat_example = [
            {"role": "user", "content": ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = get_token_func(messages=chat_example, tokenizer=tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    example_ids, formatted_chat = format_message(
        input_system, input_user, examples_included
    )

    return example_ids, formatted_chat


def format_chat_general(
    model_name,
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
):
    """
    Format the input for general models with optional examples and length truncation.
    The output will be a chat-like structure compatible with `tokenizer.apply_chat_template`.

    Args:
        model_name (str): The model's name, may influence formatting strategy.
        tokenizer: Tokenizer object for processing text.
        language (str): The language code for rough token calculation.
        input_system (str): The system-level instruction or prompt.
        input_user (str): The user-level input query.
        max_token_input (int): The maximum allowable token count for the model input.
        times_word_token (int): Multiplicative factor for rough token counting.
        num_token_reserve (int): Number of tokens to reserve (e.g., for generation).
        list_example (list of dict): Optional examples, each dict contains {"input":..., "output":...}.

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for general models.

    """

    if any(keyword in model_name.lower() for keyword in ["gemma-2", "biomistral"]):
        example_ids, formatted_chat = get_formatted_chat_user_assistant(
            tokenizer=tokenizer,
            language=language,
            input_system=input_system,
            input_user=input_user,
            max_token_input=max_token_input,
            times_word_token=times_word_token,
            num_token_reserve=num_token_reserve,
            examples=examples,
        )
    else:
        example_ids, formatted_chat = get_formatted_chat(
            tokenizer=tokenizer,
            language=language,
            input_system=input_system,
            input_user=input_user,
            max_token_input=max_token_input,
            times_word_token=times_word_token,
            num_token_reserve=num_token_reserve,
            examples=examples,
        )

    # Apply the chat template for final formatting
    formatted_input = tokenizer.apply_chat_template(
        formatted_chat, tokenize=False, add_generation_prompt=True
    )

    return example_ids, formatted_input


def format_message_user_assistant(input_system, input_user, examples):
    input_system = f"{input_system}\n\nInput:\n"
    if len(examples) > 0:
        formatted_input = [
            {
                "role": "user",
                "content": input_system + examples[0]["input"],
            },
            {"role": "assistant", "content": examples[0]["output"]},
        ]
        for ex in examples[1:]:
            formatted_input.extend(
                [
                    {"role": "user", "content": f"Input:\n{ex['input']}"},
                    {"role": "assistant", "content": ex["output"]},
                ]
            )
        formatted_input.append({"role": "user", "content": f"Input:\n{input_user}"})
    else:
        formatted_input = [{"role": "user", "content": input_system + input_user}]
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_input


def get_formatted_chat_user_assistant(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
):
    """
    Format input for 'user_assistant' style models, supporting examples and length truncation.
    Unlike general format, this format integrates the system prompt into the user's role content
    and prefixes user inputs with "Input:\n".

    Args:
        tokenizer: The tokenizer object for processing text.
        language (str): The language of the input, used for rough token estimation.
        input_system (str): System instruction.
        input_user (str): User input query.
        max_token_input (int): Maximum token limit for the input.
        times_word_token (int): The factor for rough token calculation.
        num_token_reserve (int): Tokens to reserve for response generation.
        examples (list): Optional list of examples, each a dict with "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for 'user_assistant' models.
    """

    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_input = format_message_user_assistant(
            input_system, input_user, examples
        )
        return example_ids, formatted_input

    # Step 2: Handle truncation when exceeding token limit
    # Prepare a base template to estimate how many tokens system and minimal structure consumes
    _, base_template = format_message_user_assistant(input_system, input_user, [])
    base_token_count = num_tokens_from_messages(base_template, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - len(tokenizer.encode(input_user)))
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        input_user = get_truncate_text(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message_user_assistant(input_system, input_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    # Step 4: Gradually add examples if there's space left
    examples_included = []
    for idx_example, ex in enumerate(examples):
        chat_example = [
            {"role": "user", "content": "Input:\n" + ex["input"]},
            {"role": "assistant", "content": ex["output"]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = num_tokens_from_messages(chat_example, tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    examples_included_ids, formatted_input = format_message_user_assistant(
        input_system, input_user, examples_included
    )

    return examples_included_ids, formatted_input


def format_chat_meditron(
    model_name,
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
):
    """
    Format input for Meditron, supporting examples and length truncation.

    Args:
        model_name: The model's name (not used in current implementation, reserved for extensibility).
        tokenizer: The tokenizer object for processing text.
        language: The language of the input (used for rough token estimation).
        input_system: System instruction.
        input_user: User input.
        max_token_input: Maximum token limit for the input.
        times_word_token: Multiplier for rough token estimation.
        num_token_reserve: Number of tokens to reserve for formatting.
        examples: List of example dictionaries (optional). Each example contains "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A Formatted input string for Meditron.
    """

    # Estimate rough token count
    token_rough = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If token count is below the limit, format without truncation
    if token_rough <= max_token_input:
        chat_example = ""
        if examples:
            for ex in examples:
                chat_example += (
                    f"<|im_start|> user\n{ex['input']}<|im_end|>\n"
                    f"<|im_start|> assistant\n{ex['output']}<|im_end|>\n"
                )
        example_ids = [ex["id"] for ex in examples] if examples else []
        formatted_input = (
            f"<|im_start|> system\n{input_system}<|im_end|>\n"
            f"{chat_example}"
            f"<|im_start|> user\n{input_user}<|im_end|>\n"
            f"<|im_start|> assistant\n"
        )
        return example_ids, formatted_input

    # Otherwise, perform truncation
    base_template = f"<|im_start|> system\n{input_system}<|im_end|>\n"
    base_token_count = len(tokenizer.encode(base_template))
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    # Truncate input_user
    # Note: the prefix like "<|im_start|> user..." required 27 tokens
    input_user = get_truncate_text(input_user, tokenizer, remaining_tokens - 30)
    user_template = (
        f"<|im_start|> user\n{input_user}<|im_end|>\n<|im_start|> assistant\n"
    )
    user_token_count = len(tokenizer.encode(user_template))
    remaining_tokens -= user_token_count

    # Handle examples
    chat_example = ""
    examples_included_ids = []
    if examples and remaining_tokens > 0:
        for ex in examples:
            current_example_text = (
                f"<|im_start|> user\n{ex['input']}<|im_end|>\n"
                f"<|im_start|> assistant\n{ex['output']}<|im_end|>\n"
            )
            current_example_token_count = len(tokenizer.encode(current_example_text))
            # Check if adding the current example would exceed the remaining token budget
            if current_example_token_count <= remaining_tokens:
                chat_example += current_example_text
                examples_included_ids.append(ex["id"])
                remaining_tokens -= current_example_token_count
            else:
                break

    # Combine all parts
    formatted_input = f"{base_template}{chat_example}{user_template}"
    return examples_included_ids, formatted_input


def format_chat_mellama(
    model_name,
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=7,
    num_token_reserve=10,
    examples=[],
):
    """
    Format input for Mellama, supporting examples and length truncation.

    Args:
        model_name: The model's name (not used in current implementation, reserved for extensibility).
        tokenizer: The tokenizer object for processing text.
        language: Language of the input, used for rough token estimation.
        input_system: System instruction.
        input_user: User
          input.
        max_token_input: Maximum token limit for the input.
        times_word_token: Multiplier for rough token estimation.
        num_token_reserve: Number of tokens to reserve for formatting.
        examples: List of example dictionaries (optional). Each example contains "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for Mellama.
        formatted_input: A formatted input string for Mellama.
    """

    token_rough = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    if token_rough <= max_token_input:
        example_texts = ""
        if examples:
            for ex in examples:
                example_texts += f"INPUT: {ex['input']}\nOUTPUT: {ex['output']}\n"
        example_ids = [ex["id"] for ex in examples] if examples else []
        formatted_input = (
            f"{input_system}\n\n{example_texts}INPUT: {input_user}\nOUTPUT: "
        )
        return example_ids, formatted_input

    # Otherwise, perform truncation
    base_template = f"{input_system}\n\n"
    token_template_count = len(tokenizer.encode(base_template))
    remaining_tokens = max_token_input - token_template_count - num_token_reserve

    # Truncate input_user
    # Note: the prefix like "INPUT: \nOUTPUT: " required 7 tokens
    input_user = get_truncate_text(input_user, tokenizer, remaining_tokens - 10)
    user_template = f"INPUT: {input_user}\nOUTPUT: "
    user_token_count = len(tokenizer.encode(user_template))
    remaining_tokens -= user_token_count

    # Truncate examples
    examples_text = ""
    examples_included_ids = []
    if examples:
        for ex in examples:
            # Build the example text for this example
            ex_text = f"INPUT: {ex['input']}\nOUTPUT: {ex['output']}\n"
            ex_token_count = len(tokenizer.encode(ex_text))
            # Check if adding this example would exceed the remaining token budget
            if ex_token_count <= remaining_tokens:
                examples_text += ex_text
                examples_included_ids.append(ex["id"])
                remaining_tokens -= ex_token_count
            else:
                # Skip example if it doesn't fit
                continue

    formatted_input = f"{base_template}{examples_text}{user_template}"

    return examples_included_ids, formatted_input


def format_message_gemini(input_system, input_user, examples):
    input_system = f"{input_system}\n\nInput:\n"
    if len(examples) > 0:
        formatted_input = [
            {
                "role": "user",
                "parts": [{"text": input_system + examples[0]["input"]}],
            },
            {"role": "model", "parts": [{"text": examples[0]["output"]}]},
        ]
        for ex in examples[1:]:
            formatted_input.extend(
                [
                    {"role": "user", "parts": [{"text": f"Input:\n{ex['input']}"}]},
                    {"role": "model", "parts": [{"text": ex["output"]}]},
                ]
            )
        formatted_input.append(
            {"role": "user", "parts": [{"text": f"Input:\n{input_user}"}]}
        )
    else:
        formatted_input = [
            {"role": "user", "parts": [{"text": input_system + input_user}]}
        ]
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_input


def get_formatted_chat_gemini(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=4,
    num_token_reserve=5,
    examples=[],
):
    """
    Format input for Gwmini models, supporting examples and length truncation.
    Unlike general format, this format integrates the system prompt into the user's role content and follows the Gemini style
    and prefixes user inputs with "Input:\n".

    Args:
        tokenizer: The tokenizer object for processing text.
        language (str): The language of the input, used for rough token estimation.
        input_system (str): System instruction.
        input_user (str): User input query.
        max_token_input (int): Maximum token limit for the input.
        times_word_token (int): The factor for rough token calculation.
        num_token_reserve (int): Tokens to reserve for response generation.
        examples (list): Optional list of examples, each a dict with "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for 'user_assistant' models.
    """

    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_input = format_message_gemini(
            input_system, input_user, examples
        )
        return example_ids, formatted_input

    # Step 2: Handle truncation when exceeding token limit
    # Prepare a base template to estimate how many tokens system and minimal structure consumes
    _, base_template = format_message_gemini(input_system, input_user, [])
    base_token_count = num_tokens_from_messages_gemini(base_template, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - tokenizer.count_tokens(input_user).total_tokens)
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        input_user = get_truncate_text_gemini(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message_gemini(input_system, input_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    examples_included = []
    for idx_example, ex in enumerate(examples):
        chat_example = [
            {"role": "user", "parts": [{"text": "Input:\n" + ex["input"]}]},
            {"role": "model", "parts": [{"text": ex["output"]}]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = num_tokens_from_messages_gemini(chat_example, tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    example_ids, formatted_input = format_message_gemini(
        input_system, input_user, examples_included
    )

    return example_ids, formatted_input


def format_message_deepseek(input_system, input_user, examples):
    input_system = f"{input_system}\n\nInput:\n"
    if len(examples) > 0:
        formatted_input = [
            {
                "role": "user",
                "text": input_system + examples[0]["input"],
            },
            {"role": "assistant", "text": examples[0]["output"]},
        ]
        for ex in examples[1:]:
            formatted_input.extend(
                [
                    {"role": "user", "text": f"Input:\n{ex['input']}"},
                    {"role": "assistant", "text": ex["output"]},
                ]
            )
        formatted_input.append({"role": "user", "text": f"Input:\n{input_user}"})
    else:
        formatted_input = [{"role": "user", "text": input_system + input_user}]
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_input


def get_formatted_chat_deepseek(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=4,
    num_token_reserve=5,
    examples=[],
):
    """
    Format input for DeepSeek models, supporting examples and length truncation.
    Unlike general format, this format integrates the system prompt into the user's role content and follows the DeepSeek style
    and prefixes user inputs with "Input:\n".

    Args:
        tokenizer: The tokenizer object for processing text.
        language (str): The language of the input, used for rough token estimation.
        input_system (str): System instruction.
        input_user (str): User input query.
        max_token_input (int): Maximum token limit for the input.
        times_word_token (int): The factor for rough token calculation.
        num_token_reserve (int): Tokens to reserve for response generation.
        examples (list): Optional list of examples, each a dict with "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for 'user_assistant' models.
    """

    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_input = format_message_deepseek(
            input_system, input_user, examples
        )
        return example_ids, formatted_input

    # Step 2: Handle truncation when exceeding token limit
    # Prepare a base template to estimate how many tokens system and minimal structure consumes
    _, base_template = format_message_deepseek(input_system, input_user, [])
    base_token_count = num_tokens_from_messages_deepseek(base_template, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - len(tokenizer.encode(input_user)))
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        input_user = get_truncate_text_deepseek(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message_deepseek(input_system, input_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    examples_included = []
    for idx_example, ex in enumerate(examples):
        chat_example = [
            {"role": "user", "text": "Input:\n" + ex["input"]},
            {"role": "assistant", "text": ex["output"]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = num_tokens_from_messages_deepseek(chat_example, tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    example_ids, formatted_input = format_message_deepseek(
        input_system, input_user, examples_included
    )

    return example_ids, formatted_input


def format_message_gemini(input_system, input_user, examples):
    input_system = f"{input_system}\n\nInput:\n"
    if len(examples) > 0:
        formatted_input = [
            {
                "role": "user",
                "parts": [{"text": input_system + examples[0]["input"]}],
            },
            {"role": "model", "parts": [{"text": examples[0]["output"]}]},
        ]
        for ex in examples[1:]:
            formatted_input.extend(
                [
                    {"role": "user", "parts": [{"text": f"Input:\n{ex['input']}"}]},
                    {"role": "model", "parts": [{"text": ex["output"]}]},
                ]
            )
        formatted_input.append(
            {"role": "user", "parts": [{"text": f"Input:\n{input_user}"}]}
        )
    else:
        formatted_input = [
            {"role": "user", "parts": [{"text": input_system + input_user}]}
        ]
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_input


def get_formatted_chat_gemini(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=4,
    num_token_reserve=5,
    examples=[],
):
    """
    Format input for Gwmini models, supporting examples and length truncation.
    Unlike general format, this format integrates the system prompt into the user's role content and follows the Gemini style
    and prefixes user inputs with "Input:\n".

    Args:
        tokenizer: The tokenizer object for processing text.
        language (str): The language of the input, used for rough token estimation.
        input_system (str): System instruction.
        input_user (str): User input query.
        max_token_input (int): Maximum token limit for the input.
        times_word_token (int): The factor for rough token calculation.
        num_token_reserve (int): Tokens to reserve for response generation.
        examples (list): Optional list of examples, each a dict with "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for 'user_assistant' models.
    """

    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_input = format_message_gemini(
            input_system, input_user, examples
        )
        return example_ids, formatted_input

    # Step 2: Handle truncation when exceeding token limit
    # Prepare a base template to estimate how many tokens system and minimal structure consumes
    _, base_template = format_message_gemini(input_system, input_user, [])
    base_token_count = num_tokens_from_messages_gemini(base_template, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - tokenizer.count_tokens(input_user).total_tokens)
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        input_user = get_truncate_text_gemini(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message_gemini(input_system, input_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    examples_included = []
    for idx_example, ex in enumerate(examples):
        chat_example = [
            {"role": "user", "parts": [{"text": "Input:\n" + ex["input"]}]},
            {"role": "model", "parts": [{"text": ex["output"]}]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = num_tokens_from_messages_gemini(chat_example, tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    example_ids, formatted_input = format_message_gemini(
        input_system, input_user, examples_included
    )

    return example_ids, formatted_input


def format_message_deepseek(input_system, input_user, examples):
    input_system = f"{input_system}\n\nInput:\n"
    if len(examples) > 0:
        formatted_input = [
            {
                "role": "user",
                "text": input_system + examples[0]["input"],
            },
            {"role": "assistant", "text": examples[0]["output"]},
        ]
        for ex in examples[1:]:
            formatted_input.extend(
                [
                    {"role": "user", "text": f"Input:\n{ex['input']}"},
                    {"role": "assistant", "text": ex["output"]},
                ]
            )
        formatted_input.append({"role": "user", "text": f"Input:\n{input_user}"})
    else:
        formatted_input = [{"role": "user", "text": input_system + input_user}]
    example_ids = [ex["id"] for ex in examples] if examples else []
    return example_ids, formatted_input


def get_formatted_chat_deepseek(
    tokenizer,
    language,
    input_system,
    input_user,
    max_token_input,
    times_word_token=4,
    num_token_reserve=5,
    examples=[],
):
    """
    Format input for DeepSeek models, supporting examples and length truncation.
    Unlike general format, this format integrates the system prompt into the user's role content and follows the DeepSeek style
    and prefixes user inputs with "Input:\n".

    Args:
        tokenizer: The tokenizer object for processing text.
        language (str): The language of the input, used for rough token estimation.
        input_system (str): System instruction.
        input_user (str): User input query.
        max_token_input (int): Maximum token limit for the input.
        times_word_token (int): The factor for rough token calculation.
        num_token_reserve (int): Tokens to reserve for response generation.
        examples (list): Optional list of examples, each a dict with "input" and "output".

    Returns:
        Example_ids: A list of example IDs used in the formatted input.
        formatted_input: A formatted input string for 'user_assistant' models.
    """

    # Step 1: Rough token estimation (not exact, but a starting point)
    initial_estimated_token_count = estimate_token_rough(
        input_system, input_user, language, times_word_token, examples
    )

    # If the rough token count is within the limit, no truncation is needed
    if initial_estimated_token_count <= max_token_input:
        example_ids, formatted_input = format_message_deepseek(
            input_system, input_user, examples
        )
        return example_ids, formatted_input

    # Step 2: Handle truncation when exceeding token limit
    # Prepare a base template to estimate how many tokens system and minimal structure consumes
    _, base_template = format_message_deepseek(input_system, input_user, [])
    base_token_count = num_tokens_from_messages_deepseek(base_template, tokenizer)

    # Step 3: Truncate user input first to fit into remaining tokens
    if base_token_count > max_token_input:
        # Calculate how many tokens we have left for user
        remain_for_user = (
            max_token_input
            - (base_token_count - len(tokenizer.encode(input_user)))
            - num_token_reserve
        )
        # Truncate user input first to fit into remaining tokens
        input_user = get_truncate_text_deepseek(input_user, tokenizer, remain_for_user)

        # Construct the chat sequence without examples
        return format_message_deepseek(input_system, input_user, [])

    # Step 4: Gradually add examples if there's space left
    remaining_tokens = max_token_input - base_token_count - num_token_reserve

    examples_included = []
    for idx_example, ex in enumerate(examples):
        chat_example = [
            {"role": "user", "text": "Input:\n" + ex["input"]},
            {"role": "assistant", "text": ex["output"]},
        ]
        # Tokenize to check if we can fit this example within remaining tokens
        ex_token_count = num_tokens_from_messages_deepseek(chat_example, tokenizer)

        # If we have enough room for this example, add it to the list
        if ex_token_count < remaining_tokens:
            examples_included.append(ex)
            remaining_tokens -= ex_token_count
        else:
            break

    # Step 5: Combine all parts: system, examples, user
    example_ids, formatted_input = format_message_deepseek(
        input_system, input_user, examples_included
    )

    return example_ids, formatted_input
