import os
import json
import hashlib
import numpy as np
import warnings
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from .config import (
    get_models_evaluate,
    transform_instruction_to_cot,
    transform_instruction_to_direct,
)

from .process import format_chat

from util.tool import get_mean_std_ci
from util.runtime import parse_path_list, parse_prompt_mode


class GeneralDataset(Dataset):
    """Torch dataset that lazily formats BRIDGE samples into model prompts."""

    def __init__(self, args, data, tokenizer, example=None):
        """
        Initialize the dataset for benchmarking.

        Input:
            list_dict_data: list of dictionary, each dictionary represent a sample
        Each sample contains:
            - task: str, name of the task
            - id: str, id of the sample
            - split: str, split of the sample
            - instruction: str, instruction for LLM conduct the task
            - input: str, input for the task
            - output: str, expected output for the task
        """
        self.args = args
        self.data = data
        self.tokenizer = tokenizer
        self.model_name = args.model_name
        self.examples = example or []
        self.chat_kwargs = args.chat_kwargs
        self.max_token_input = args.max_token_input
        self.max_token_output = args.max_token_output
        self.dict_idx_cache = {}

    def __len__(self):
        """Return the number of source samples in this split."""
        return len(self.data)

    def __getitem__(self, idx):
        """Return one formatted prompt, using a per-index cache after first access."""
        if idx in self.dict_idx_cache:
            return self.dict_idx_cache[idx]
        input_text = format_chat(
            model_name=self.model_name,
            tokenizer=self.tokenizer,
            data=self.data[idx],
            max_token_input=self.max_token_input,
            examples=self.examples,
            chat_kwargs=self.chat_kwargs,
        )
        self.dict_idx_cache[idx] = input_text
        return input_text


class GeneralTask:
    """Load one BRIDGE task file and prepare split-specific prompt datasets."""

    def __init__(self, args, task_name):
        """Load task JSON data from configured dataset roots and split it."""
        # config
        self.args = args
        self.name = task_name

        # Load the first matching task file from the configured dataset roots.
        self.path_file_data = None
        for dir_dataset_one in args.dir_dataset:
            path_file_data = f"{dir_dataset_one}/{self.name}.SFT.json"
            if os.path.exists(path_file_data):
                self.path_file_data = path_file_data
                break
        if self.path_file_data is None:
            searched = [f"{d}/{self.name}.SFT.json" for d in args.dir_dataset]
            raise FileNotFoundError(
                f"Task data file not found for {self.name}. Searched: {searched}"
            )

        with open(self.path_file_data, "r", encoding="utf-8") as file:
            list_dict_data = json.load(file)

        # Ensure data is not empty
        if not list_dict_data:
            raise ValueError(f"No data found in {self.path_file_data}")

        self.language = list_dict_data[0]["language"]

        # split data
        self.dataset_train_base = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "train" and dict_data["output"].strip() != ""
        ]
        self.dataset_dev_base = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "dev" and dict_data["output"].strip() != ""
        ]
        self.dataset_test_base = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "test" and dict_data["output"].strip() != ""
        ]

        print(
            "Load {} data: train: {}, val: {}, test: {}".format(
                self.name,
                len(self.dataset_train_base),
                len(self.dataset_dev_base),
                len(self.dataset_test_base),
            )
        )

        if hasattr(args, "max_data_size") and args.max_data_size:
            max_data_size = args.max_data_size
            if len(self.dataset_train_base) > max_data_size:
                self.dataset_train_base = self.dataset_train_base[:max_data_size]
            if len(self.dataset_dev_base) > max_data_size:
                self.dataset_dev_base = self.dataset_dev_base[:max_data_size]
            if len(self.dataset_test_base) > max_data_size:
                self.dataset_test_base = self.dataset_test_base[:max_data_size]
            print(f"Set maximum data size to {max_data_size} samples.")
            print(
                f"After sampling, data size: train: {len(self.dataset_train_base)}, "
                f"val: {len(self.dataset_dev_base)}, test: {len(self.dataset_test_base)}"
            )

        self._reset_runtime_datasets()

    def _reset_runtime_datasets(self):
        """Create fresh mutable split copies for one prompt setup."""
        self.dataset_train = [item.copy() for item in self.dataset_train_base]
        self.dataset_dev = [item.copy() for item in self.dataset_dev_base]
        self.dataset_test = [item.copy() for item in self.dataset_test_base]

    def _iter_runtime_data(self):
        """Iterate through the currently active split copies."""
        return self.dataset_train + self.dataset_dev + self.dataset_test

    def setup_direct(self):
        """
        Transform the dataset to the direct format
        """
        print(f"Prompt mode: Direct")
        num_transformed = 0
        for idx_data, dict_data in enumerate(self._iter_runtime_data()):
            instruction_direct = transform_instruction_to_direct(
                dict_data["instruction"],
                flag_print=False,
            )
            if instruction_direct:
                if idx_data == 0:
                    print(
                        f"Original instruction: {dict_data['instruction']}\n"
                        f"Transformed instruction: {instruction_direct}"
                )
                # Replace the instruction in the dataset
                dict_data["instruction"] = instruction_direct
                num_transformed += 1
        if num_transformed:
            print(f"Transformed {num_transformed} instructions from CoT to Direct.")

    def setup_cot(self):
        """
        Transform the dataset to the Chain of Thought format
        """
        print(f"Prompt mode: Chain of Thought")
        num_transformed = 0
        for idx_data, dict_data in enumerate(self._iter_runtime_data()):
            instruction_cot = transform_instruction_to_cot(
                dict_data["instruction"],
                flag_print=False,
            )
            if instruction_cot:
                if idx_data == 0:
                    print(
                        f"Original instruction: {dict_data['instruction']}\n"
                        f"Transformed instruction: {instruction_cot}"
                )
                # Replace the instruction in the dataset
                dict_data["instruction"] = instruction_cot
                num_transformed += 1
            else:
                # If no direct instruction is found for CoT conversion
                sample_id = dict_data.get("id", dict_data.get("idx", "unknown"))
                raise ValueError(
                    "No Direct instruction template was found for CoT conversion "
                    f"in task={self.name}, sample_id={sample_id}."
                )
        if num_transformed:
            print(f"Transformed {num_transformed} instructions from Direct to CoT.")
        else:
            print("Warning: no Direct prompt template was found for CoT conversion.")

    def setup_few_shot(self, num_example):
        """
        Get the example for the task
        """
        list_path_searched = []
        path_file_example = ""
        if self.prompt_base == "direct":
            print(f"Prompt mode: Direct with {num_example}-Shot")
            for dir_dataset_one in self.args.dir_dataset:
                path_file_example = (
                    f"{dir_dataset_one}/example/{self.name}.example.json"
                )
                list_path_searched.append(path_file_example)
                if os.path.exists(path_file_example):
                    break
        else:
            print(f"Prompt mode: CoT with {num_example}-Shot")
            for dir_dataset_one in self.args.dir_dataset:
                path_file_example = (
                    f"{dir_dataset_one}/example_cot/{self.name}.example.json"
                )
                list_path_searched.append(path_file_example)
                if os.path.exists(path_file_example):
                    break
        if not os.path.exists(path_file_example):
            raise FileNotFoundError(
                f"Example file not found for task={self.name}, prompt_mode={self.prompt_mode}. "
                f"Searched: {list_path_searched}"
            )
        with open(path_file_example, "r", encoding="utf-8") as f:
            list_example = json.load(f)
        self.examples = list_example[:num_example]
        # remove the example from the training data
        self.dataset_train = [
            dict_data
            for dict_data in self.dataset_train
            if dict_data not in self.examples
        ]
        print(f"Load {num_example} examples for {path_file_example}")

    def setup(self, tokenizer, prompt_mode="direct"):
        """Prepare the task data for one prompt mode and tokenizer."""
        # config of tokenizer and model
        self.tokenizer = tokenizer
        # config of inference
        self.prompt_mode = prompt_mode
        self.prompt_base, num_example = parse_prompt_mode(prompt_mode)
        self.examples = []
        self._reset_runtime_datasets()
        # Prompt mode: "direct", "cot"
        if self.prompt_base == "direct":
            self.setup_direct()
        elif self.prompt_base == "cot":
            self.setup_cot()
        # Prompt mode: "5-shot"
        if num_example is not None:
            self.setup_few_shot(num_example)

    def dataloader_all(self, batch_size, num_workers):
        """Return a dataloader over train, dev, and test splits."""
        dataset = self.dataset_train + self.dataset_dev + self.dataset_test
        return dataset, DataLoader(
            dataset=GeneralDataset(
                self.args, dataset, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    def dataloader_train(self, batch_size, num_workers):
        """Return a dataloader over the train split."""
        dataset = self.dataset_train
        return dataset, DataLoader(
            dataset=GeneralDataset(
                self.args, dataset, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    def dataloader_dev(self, batch_size, num_workers):
        """Return a dataloader over the dev split."""
        dataset = self.dataset_dev
        return dataset, DataLoader(
            dataset=GeneralDataset(
                self.args, self.dataset_dev, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    def dataloader_test(self, batch_size, num_workers):
        """Return a dataloader over the test split."""
        dataset = self.dataset_test
        return dataset, DataLoader(
            dataset=GeneralDataset(
                self.args, self.dataset_test, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )


class GeneralEvaluation(GeneralTask):
    """Base class for task-specific result parsing and metric computation."""

    def __init__(self, args, task):
        """Initialize evaluation from the same task data loading path as inference."""
        super().__init__(args, task)

    def _normalize_predictions_for_evaluation(self, list_dict_result):
        """
        Convert multi-sample predictions into one prediction per record.

        vLLM/HF sampling can save pred as a list when sample_size > 1. Current
        task metrics expect a single string prediction, so evaluation selects
        sample_eval_index from the list while keeping the saved result file
        unchanged.
        """
        sample_eval_index = int(getattr(self.args, "sample_eval_index", 0))
        normalized = []
        for dict_result in list_dict_result:
            pred = dict_result.get("pred")
            if isinstance(pred, list):
                if len(pred) == 0:
                    pred = ""
                else:
                    if sample_eval_index >= len(pred):
                        raise ValueError(
                            "sample_eval_index is out of range for a multi-sample "
                            f"prediction: index={sample_eval_index}, size={len(pred)}"
                        )
                    pred = pred[sample_eval_index]
                dict_result = dict_result.copy()
                dict_result["pred"] = pred
            normalized.append(dict_result)
        return normalized

    def _seed_evaluation_rng(self, file_result, prompt_mode):
        """
        Seed numpy before parsing invalid predictions and bootstrapping.

        Invalid classification outputs are intentionally replaced with random
        valid labels. Seeding per result file makes that behavior reproducible
        even when a notebook evaluates tasks in a different order.
        """
        base_seed = int(getattr(self.args, "seed", 42))
        key = f"{base_seed}|{self.name}|{prompt_mode}|{file_result}"
        seed_offset = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed((base_seed + seed_offset) % (2**32))

    def _get_expected_records_for_result(self, list_dict_result):
        """Infer which task split(s) a result file should align with."""
        split_to_records = {
            "train": self.dataset_train_base,
            "dev": self.dataset_dev_base,
            "test": self.dataset_test_base,
        }
        split_order = ["train", "dev", "test"]
        result_splits = {
            str(dict_result.get("split"))
            for dict_result in list_dict_result
            if dict_result.get("split") in split_to_records
        }
        if result_splits:
            splits = [split for split in split_order if split in result_splits]
        else:
            # Standard BRIDGE result files are test-split outputs. Compact
            # result files may not keep the split field, so default to test.
            splits = ["test"]
        expected = []
        for split in splits:
            expected.extend(split_to_records[split])
        return splits, expected

    def _warn_if_result_not_aligned(self, file_result, list_dict_result):
        """Warn when result row count or ids do not align with task data."""
        splits, expected = self._get_expected_records_for_result(list_dict_result)
        messages = []
        if len(list_dict_result) != len(expected):
            messages.append(
                f"row_count={len(list_dict_result)} but expected={len(expected)} "
                f"for split(s)={splits}"
            )

        result_ids = [
            str(dict_result["id"])
            for dict_result in list_dict_result
            if dict_result.get("id") is not None
        ]
        expected_ids = [
            str(dict_data["id"])
            for dict_data in expected
            if dict_data.get("id") is not None
        ]
        if result_ids and expected_ids:
            if len(result_ids) == len(expected_ids) and result_ids != expected_ids:
                missing_ids = sorted(set(expected_ids) - set(result_ids))[:5]
                extra_ids = sorted(set(result_ids) - set(expected_ids))[:5]
                messages.append(
                    "result ids do not align with task data order "
                    f"(missing sample={missing_ids}, extra sample={extra_ids})"
                )
            elif len(result_ids) != len(list_dict_result):
                messages.append("some result rows do not contain an id field")

        if messages:
            warnings.warn(
                f"Result alignment warning for {file_result}: " + "; ".join(messages),
                RuntimeWarning,
            )

    def get_label_one(self, str_output, prompt_mode):
        """
        The function to get the label from one dictionary data
        """
        # Will be defined in the child class of specific task
        pass

    def get_pred_one(self, str_pred, prompt_mode):
        """
        The function to get the prediction from one dictionary data
        """
        # Will be defined in the child class of specific task
        pass

    def get_label(self, list_dict_data, prompt_mode):
        """
        The function to get the label from the list of dictionary data
        """
        # Will be defined in the child class of specific task
        pass

    def get_pred(self, list_dict_data, prompt_mode):
        """
        The function to get the prediction from the list of dictionary data
        """
        # Will be defined in the child class of specific task
        pass

    def get_pred_none(self, list_pred, list_label):
        """
        The function to get the prediction for the invalid response
        """
        # Will be defined in the child class of specific task
        pass

    def get_performance(self, list_pred, list_label):
        """
        Calculate performance metrics based on the task type.

        Input:
            list_pred: list of pred, the pred can be a label, a list of labels, or a a list of dictionary
            list_label: list of labels, the true results for each sample

        Returns:
            dict: A dictionary containing performance metrics.
        """
        # Will be defined in the child class of specific task
        pass

    def search_result_by_model(self, model_name=None, prompt_mode="direct"):
        """
        Search and collect result files for each model.

        Returns:
            dict: A dictionary mapping model names to their corresponding result file paths.
        """

        def _get_model_result_files(model_path, prompt_mode):
            """Return result files in one model directory for a prompt mode."""
            list_file_result = []
            if not os.path.isdir(model_path):
                return list_file_result
            for file_name in os.listdir(model_path):
                # Check if the file is a result file for the current prompt mode
                # Example:
                # task: ADE identification
                # prompt mode: direct-5-shot
                # decoding strategy: greedy
                # seed: 42
                # file: ADE identification-direct-5-shot-greedy-42.result.json
                if file_name.endswith(".result.json"):
                    file_prompt_mode = file_name.replace(f"{self.name}-", "")
                    if "greedy" in file_prompt_mode:
                        file_prompt_mode = file_prompt_mode.split("-greedy")[0]
                    elif "sampling" in file_prompt_mode:
                        file_prompt_mode = file_prompt_mode.split("-sampling")[0]
                    else:
                        raise ValueError(
                            f"Invalid file name: {file_name}. Expected 'greedy' or 'sampling'."
                        )
                    if prompt_mode == file_prompt_mode:
                        list_file_result.append(os.path.join(model_path, file_name))
            list_file_result.sort()

            return list_file_result

        # search the result file for each model
        dict_model_result = {}
        dir_result_list = parse_path_list(self.args.dir_result, "dir_result")
        if model_name and not isinstance(model_name, list):
            model_name = [model_name]

        for dir_result_one in dir_result_list:
            path_dir_result = os.path.join(dir_result_one, self.name)
            if not os.path.isdir(path_dir_result):
                print(f"Result directory does not exist: {path_dir_result}")
                continue
            if model_name:
                for model_one in model_name:
                    model_path = os.path.join(path_dir_result, model_one)
                    if not os.path.exists(model_path):
                        print(
                            f"Result of Model:{model_one} does not exist in {dir_result_one}"
                        )
                    list_file_result = _get_model_result_files(model_path, prompt_mode)
                    if list_file_result:
                        dict_model_result.setdefault(model_one, []).extend(
                            list_file_result
                        )
            else:
                for model_one in os.listdir(path_dir_result):
                    model_path = os.path.join(path_dir_result, model_one)
                    list_file_result = _get_model_result_files(model_path, prompt_mode)
                    if list_file_result:
                        dict_model_result.setdefault(model_one, []).extend(
                            list_file_result
                        )

        for list_file_result in dict_model_result.values():
            list_file_result.sort()

        return dict_model_result

    def evaluate_by_model(self, prompt_mode="direct", model_name=None, bootstrap=1000):
        """
        Evaluate the performance of each model.

        Input:
            bootstrap: boolean, default=1000, number of bootstrap samples

        Returns:
            dict: A dictionary containing performance metrics for each model.
        """
        parse_prompt_mode(prompt_mode)

        # get the result of each model
        dict_model_result = self.search_result_by_model(
            model_name=model_name, prompt_mode=prompt_mode
        )

        # evaluate the performance of each model
        dict_model_performance = {}
        for model_one, list_file_result in tqdm(dict_model_result.items()):
            # Initialize performance list for the model
            list_dict_performance = []
            # Process each result file and collect performance metrics
            for file_result in list_file_result:
                list_dict_performance.extend(
                    self._process_result_file(file_result, prompt_mode, bootstrap)
                )
            # Aggregate metrics for the model
            dict_model_performance[model_one] = self._aggregate_performance_metrics(
                list_dict_performance
            )

        # Sort the result by model name
        list_model = get_models_evaluate()
        dict_model_performance_sorted = self._sort_model_performance(
            list_model=list_model, dict_model_performance=dict_model_performance
        )

        return dict_model_performance_sorted

    def _process_result_file(self, file_result, prompt_mode, bootstrap):
        """
        Process a single result file and compute performance metrics.

        Returns:
            list: A list of performance metrics dictionaries.
        """
        with open(file_result, "r", encoding="utf-8") as f:
            list_dict_result = json.load(f)
        self._warn_if_result_not_aligned(file_result, list_dict_result)
        self._seed_evaluation_rng(file_result, prompt_mode)
        list_dict_result = self._normalize_predictions_for_evaluation(list_dict_result)

        list_label = self.get_label(list_dict_result, prompt_mode)
        list_pred = self.get_pred(list_dict_result, prompt_mode)
        if not list_pred:
            raise ValueError(f"No predictions found in result file: {file_result}")
        # If the task is multi-label classification, check for invalid responses
        # Note: the "gen" task may have multiple references for one prediction
        if isinstance(list_pred[0], list):
            list_failed = [1 if -1 in pred else 0 for pred in list_pred]
        else:
            list_failed = [1 if -1 == pred else 0 for pred in list_pred]
        list_pred, _ = self.get_pred_none(list_pred, list_label)

        # Check that list_pred and list_label have the same length
        if len(list_pred) != len(list_label):
            raise ValueError("The length of list_pred and list_label must match.")

        if bootstrap:
            return self.get_performance_bootstrap(
                list_pred, list_label, list_failed, bootstrap
            )
        else:
            dict_performance = self.get_performance(
                list_pred=list_pred,
                list_label=list_label,
            )
            if isinstance(dict_performance, tuple):
                dict_performance = dict_performance[0]
            num_failed = sum(list_failed)
            dict_performance["num_failed"] = num_failed
            dict_performance["num_failed_ratio"] = round(
                num_failed / len(list_label) * 100, 2
            )
            return [dict_performance]

    def get_performance_bootstrap(self, list_pred, list_label, list_failed, bootstrap):
        """
        Perform bootstrapping to compute performance metrics.

        Returns:
            list: A list of performance metrics dictionaries.
        """
        # Initialize arguments for each iteration
        num_data = len(list_label)
        num_sample = len(list_label)

        # Initialize list of arguments for each bootstrap iteration
        list_args = []
        for _ in range(bootstrap):
            # Randomly sample with replacement
            indices = np.random.choice(num_data, num_sample, replace=True)
            list_pred_sample = [list_pred[i] for i in indices]
            list_label_sample = [list_label[i] for i in indices]
            num_failed_sample = sum(list_failed[i] for i in indices)
            # Append the arguments for the current iteration
            list_args.append((list_pred_sample, list_label_sample, num_failed_sample))

        # Single thread, suitable for small-scale experiments
        list_dict_performance = []
        for args in list_args:
            dict_performance = self._performance_one_iteration(args)
            list_dict_performance.append(dict_performance)

        return list_dict_performance

    def _performance_one_iteration(self, args):
        """
        Helper function for one bootstrap iteration.
        Args is a tuple of (list_label, list_pred, list_failed).
        """
        list_pred_sample, list_label_sample, num_failed_sample = args

        dict_performance = self.get_performance(
            list_pred=list_pred_sample, list_label=list_label_sample
        )
        dict_performance["num_failed"] = num_failed_sample
        dict_performance["num_failed_ratio"] = round(
            num_failed_sample / len(list_label_sample) * 100, 2
        )
        return dict_performance

    def _aggregate_performance_metrics(self, list_dict_performance):
        """
        Aggregate performance metrics over multiple experiments.

        Returns:
            dict: A dictionary of aggregated performance metrics.
        """
        dict_metrics_statistic = {}
        if not list_dict_performance:
            return dict_metrics_statistic
        list_metrics = list(list_dict_performance[0].keys())

        for metric in list_metrics:
            list_score = [
                dict_exp_performance[metric]
                for dict_exp_performance in list_dict_performance
            ]
            if len(list_score) > 1:
                mean, std, ci = get_mean_std_ci(list_score)
                dict_metrics_statistic[metric] = {
                    "mean": mean,
                    "std": std,
                    "ci": ci,
                }
            else:
                mean = list_score[0]
                dict_metrics_statistic[metric] = {
                    "mean": mean,
                    "std": 0,
                    "ci": [mean, mean],
                }
        return dict_metrics_statistic

    def _sort_model_performance(self, list_model, dict_model_performance):
        """
        Sort the performance metrics dictionary based on predefined model order.

        Returns:
            dict: A sorted dictionary of performance metrics.
        """
        # Sort by name, naturally
        dict_model_performance = dict(
            sorted(dict_model_performance.items(), key=lambda x: x[0])
        )

        # Sort by name by the order provided in the predefined list
        dict_model_performance_sorted = {
            model_name: dict_model_performance[model_name]
            for model_name in list_model
            if model_name in dict_model_performance
        }
        # Add any additional models not in the predefined list
        for model_name in dict_model_performance:
            if model_name not in list_model:
                dict_model_performance_sorted[model_name] = dict_model_performance[
                    model_name
                ]
        return dict_model_performance_sorted
