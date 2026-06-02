import json
import torch
from tqdm import tqdm
from pathlib import Path
from model.init import model_config_hf, model_config_vllm


def _log_text_preview(logger, title, text, max_chars=2000):
    """Log a prompt or response preview without flooding the task log."""
    text = str(text)
    if len(text) > max_chars:
        text = text[:max_chars] + f"\n...[truncated log preview, total chars={len(text)}]"
    logger.info(title)
    logger.info(text)


def _collect_vllm_outputs(args, response_generators):
    """Convert vLLM RequestOutput objects to saved prediction strings/lists."""
    list_response = []
    for response_generator in response_generators:
        if args.decoding_strategy != "greedy" and getattr(args, "sample_size", 1) > 1:
            list_response.append([output.text for output in response_generator.outputs])
        else:
            list_response.append(response_generator.outputs[0].text)
    return list_response


def save_result(logger, dataset, list_response, path_file_save, save_input=True):
    """Save predictions into json file."""
    if len(dataset) != len(list_response):
        raise ValueError(
            f"Cannot save results because dataset and response lengths differ: "
            f"{len(dataset)} samples vs {len(list_response)} responses. "
            f"Target file: {path_file_save}"
        )

    list_to_save = []
    for dict_data, pred in zip(dataset, list_response):
        if save_input:
            # Save the original data and the response
            dict_result = dict_data.copy()
            dict_result["pred"] = pred
            list_to_save.append(dict_result)
        else:
            # Only save the response, not the original instruction and input
            dict_result = {
                "id": dict_data.get("id"),
                "type": dict_data.get("type"),
                "language": dict_data.get("language"),
                "output": dict_data.get("output"),
                "pred": pred,
            }
            list_to_save.append(dict_result)

    # Save the results to a json file
    path_file_save = Path(path_file_save)
    path_file_save.parent.mkdir(parents=True, exist_ok=True)
    with path_file_save.open("w", encoding="utf-8") as f:
        json.dump(list_to_save, f, indent=4, ensure_ascii=False)

    logger.info(f"Save: {path_file_save} ({len(list_to_save)} rows)")
    logger.info("\n" + "-" * 30 + "\n")
    return list_to_save


def run_hf(args, logger, dataloader, model, tokenizer):
    """Run one Hugging Face generation pass over an already formatted dataloader."""
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
            generation_kwargs = {
                "inputs": batch_input.input_ids,
                "attention_mask": batch_input.attention_mask,
                "max_new_tokens": args.max_token_output,
                "do_sample": model.generation_config.do_sample,
                "repetition_penalty": model.generation_config.repetition_penalty,
            }
            if model.generation_config.do_sample:
                generation_kwargs.update(
                    {
                        "temperature": model.generation_config.temperature,
                        "top_k": model.generation_config.top_k,
                        "top_p": model.generation_config.top_p,
                    }
                )
                sample_size = int(getattr(args, "sample_size", 1))
                if sample_size > 1:
                    generation_kwargs["num_return_sequences"] = sample_size

            batch_output = model.generate(**generation_kwargs)
            batch_output = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(
                    batch_input.input_ids.repeat_interleave(
                        generation_kwargs.get("num_return_sequences", 1), dim=0
                    ),
                    batch_output,
                )
            ]
            batch_response = tokenizer.batch_decode(
                batch_output, skip_special_tokens=True
            )
            if generation_kwargs.get("num_return_sequences", 1) > 1:
                sample_size = generation_kwargs["num_return_sequences"]
                batch_response = [
                    batch_response[i : i + sample_size]
                    for i in range(0, len(batch_response), sample_size)
                ]
            list_response += batch_response
            if idx_batch == 0:
                logger.info("\n" + "-" * 30 + "\n")
                _log_text_preview(logger, "Input: ", batch_data[0])
                _log_text_preview(logger, "Output: ", batch_response[0])
                logger.info("\n" + "-" * 30 + "\n")
            torch.cuda.empty_cache()

    return list_response


def run_vllm(args, logger, dataset, dataloader, model):
    """Run one vLLM generation pass and optionally save partial results."""
    logger.info("Run via vLLM:")

    # set the decoding strategy and parameters
    sampling_params = model_config_vllm(args, logger)

    # whether to save partial results every n batches
    save_every_n_batches = getattr(args, "save_every_n_batches", None)

    list_response = []
    for idx_batch, batch_data in enumerate(tqdm(dataloader, desc="vLLM batches")):
        logger.info(
            f"vLLM batch {idx_batch + 1}/{len(dataloader)}: "
            f"{len(batch_data)} prompts"
        )
        list_response_generator = model.generate(
            list(batch_data),
            sampling_params,
            use_tqdm=True,
        )
        list_response_batch = _collect_vllm_outputs(args, list_response_generator)
        list_response.extend(list_response_batch)

        if idx_batch == 0:
            logger.info("\n" + "-" * 30 + "\n")
            _log_text_preview(logger, "Input: ", batch_data[0])
            _log_text_preview(logger, "Output: ", list_response_batch[0])
            logger.info("\n" + "-" * 30 + "\n")

        should_save_partial = (
            save_every_n_batches is not None
            and (idx_batch + 1) % int(save_every_n_batches) == 0
        )
        if should_save_partial:
            save_result(
                logger=logger,
                dataset=dataset[: len(list_response)],
                list_response=list_response,
                path_file_save=args.path_file_result,
                save_input=args.save_input,
            )

    return list_response
