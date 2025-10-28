import torch
from model.init import model_config_hf, model_config_vllm


def run_hf(args, logger, dataloader, model, tokenizer):
    logger.info("Run via Hugging face style:")

    # set the decoding strategy and parameters
    tokenizer, model = model_config_hf(args, logger, model, tokenizer)

    list_response = []
    # model eval
    model.eval()
    with torch.no_grad():
        for idx_batch, batch_data in enumerate(dataloader):
            batch_input = tokenizer(
                batch_data,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_token_input,
                # after applying the chat template, remove one <bos>
                add_special_tokens=False,
            ).to(model.device)
            batch_output = model.generate(
                # # input setting
                inputs=batch_input.input_ids,
                attention_mask=batch_input.attention_mask,
                # # token setting
                max_new_tokens=args.max_token_output,
                # # decoding strategy
                do_sample=model.generation_config.do_sample,
                temperature=model.generation_config.temperature,
                top_k=model.generation_config.top_k,
                top_p=model.generation_config.top_p,
            )
            batch_output = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(batch_input.input_ids, batch_output)
            ]
            batch_response = tokenizer.batch_decode(
                batch_output, skip_special_tokens=True
            )
            list_response += batch_response
            if idx_batch == 0:
                logger.info("\n" + "-" * 30 + "\n")
                logger.info("Input: ")
                logger.info(batch_data[0])
                logger.info("Output: ")
                logger.info(batch_response[0])
                logger.info("\n" + "-" * 30 + "\n")
            torch.cuda.empty_cache()

    return list_response


def run_vllm(args, logger, dataloader, model):
    logger.info("Run via vLLM:")

    # set the decoding strategy and parameters
    sampling_params = model_config_vllm(args, logger)

    list_input = []
    for idx_batch, batch_data in enumerate(dataloader):
        list_input.extend(batch_data)

    list_response_generator = model.generate(
        list_input,
        sampling_params,
        use_tqdm=True,
    )

    list_response = []
    for idx, response_generator in enumerate(list_response_generator):
        list_response.append(response_generator.outputs[0].text)
        if idx == 0:
            logger.info("\n" + "-" * 30 + "\n")
            logger.info("Input: ")
            logger.info(list_input[0])
            logger.info("Output: ")
            logger.info(list_response[0])
            logger.info("\n" + "-" * 30 + "\n")

    # list_response = []
    # for idx_batch, batch_data in enumerate(dataloader):
    #     list_response_batch = model.generate(
    #         batch_data,
    #         sampling_params,
    #         use_tqdm=False,
    #     )
    #     list_response.extend(list_response_batch)
    #     if idx_batch == 0:
    #         logger.info("\nInput: ")
    #         logger.info(batch_data[0])
    #         logger.info("Output: ")
    #         logger.info(list_response_batch[0])

    return list_response
