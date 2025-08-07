import os
import regex
import json
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset, DataLoader
import numpy as np

from .config import (
    get_models_evaluate,
    transform_instruction_to_cot,
    transform_instruction_to_direct,
)

from .process import format_chat

from util.tool import get_mean_std_ci


class GeneralDataset(Dataset):
    def __init__(self, args, data, tokenizer, example=None):
        """
        Initialize the dataset for benchmarking.

        Input:
            list_dict_data: list of dictionary, each dictinary represent a sample
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
        self.examples = example
        self.chat_kwargs = args.chat_kwargs
        self.max_token_input = args.max_token_input
        self.max_token_output = args.max_token_output
        self.dict_idx_cache = {}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
    def __init__(self, args, task_name):
        # config
        self.args = args
        self.name = task_name

        # load data
        self.path_file_data = f"dataset_raw/{self.name}.SFT.json"
        with open(self.path_file_data, "r", encoding="utf-8") as file:
            list_dict_data = json.load(file)

        self.language = list_dict_data[0]["language"]

        # Ensure data is not empty
        if not list_dict_data:
            raise ValueError(f"No data found in {self.path_file_data}")

        # split data
        self.dataset_train = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "train" and dict_data["output"].strip() != ""
        ]
        self.dataset_dev = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "dev" and dict_data["output"].strip() != ""
        ]
        self.dataset_test = [
            dict_data
            for dict_data in list_dict_data
            if dict_data["split"] == "test" and dict_data["output"].strip() != ""
        ]

        print(
            "Load {} data: train: {}, val: {}, test: {}".format(
                self.name,
                len(self.dataset_train),
                len(self.dataset_dev),
                len(self.dataset_test),
            )
        )

    def setup_direct(self):
        """
        Transform the dataset to the direct format
        """
        print(f"Prompt mode: Direct")
        for idx_data, dict_data in enumerate(
            self.dataset_train + self.dataset_dev + self.dataset_test
        ):
            instruction_direct = transform_instruction_to_direct(
                dict_data["instruction"]
            )
            if instruction_direct:
                if idx_data == 0:
                    print(
                        f"Original instruction: {dict_data['instruction']}\n"
                        f"Transformed instruction: {instruction_direct}"
                    )
                # Replace the instruction in the dataset
                dict_data["instruction"] = instruction_direct
            else:
                # If the instruction was Direct format or not contain the CoT template, skip
                print(
                    f"Warning: The instruction does not contain any of the CoT templates. Skip."
                )
                break

    def setup_cot(self):
        """
        Transform the dataset to the Chain of Thought format
        """
        print(f"Prompt mode: Chain of Thought")
        for idx_data, dict_data in enumerate(
            self.dataset_train + self.dataset_dev + self.dataset_test
        ):
            instruction_cot = transform_instruction_to_cot(dict_data["instruction"])
            if instruction_cot:
                if idx_data == 0:
                    print(
                        f"Original instruction: {dict_data['instruction']}\n"
                        f"Transformed instruction: {instruction_cot}"
                    )
                # Replace the instruction in the dataset
                dict_data["instruction"] = instruction_cot
            else:
                # If the instruction was CoT format or not contain the Direct template, skip
                print(
                    f"Warning: The instruction does not contain any of the Direct templates. Skip."
                )
                break

    def setup_few_shot(self, num_example):
        """
        Get the example for the task
        """
        print(f"Prompt mode: Direct with {num_example}-Shot")
        path_file_example = f"dataset_raw/example/{self.name}.example.json"
        with open(path_file_example, "r") as f:
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
        # config of tokenizer and model
        self.tokenizer = tokenizer
        # config of inference
        self.prompt_mode = prompt_mode
        self.examples = []
        # Prompt mode: "direct", "cot"
        if "direct" in self.prompt_mode:
            self.setup_direct()
        elif "cot" in self.prompt_mode:
            self.setup_cot()
        # Prompt mode: "5-shot"
        if self.prompt_mode.endswith("shot"):
            num_example = int(regex.findall(r"\d+", self.prompt_mode)[0])
            self.setup_few_shot(num_example)

    def dataloader_train(self, batch_size, num_workers):
        return DataLoader(
            dataset=GeneralDataset(
                self.args, self.dataset_train, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=num_workers,
        )

    def dataloader_dev(self, batch_size, num_workers):
        return DataLoader(
            dataset=GeneralDataset(
                self.args, self.dataset_dev, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )

    def dataloader_test(self, batch_size, num_workers):
        return DataLoader(
            dataset=GeneralDataset(
                self.args, self.dataset_test, self.tokenizer, self.examples
            ),
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        )


class GeneralEvaluation(GeneralTask):
    def __init__(self, args, task):
        super().__init__(args, task)

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
            list_file_result = []
            for file_name in os.listdir(model_path):
                # Check if the file is a result file for the current prompt mode
                # Example:
                # task: ADE identification
                # prompt mode: direct-5-shot
                # decoding strategy: greedy
                # seed: 42
                # file: ADE identification-direct-5-shot-greedy-42.result.json
                if file_name.endswith(".json"):
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

        path_dir_result = os.path.join("result", self.name)
        # search the result file for each model
        dict_model_result = {}
        if model_name:
            if not isinstance(model_name, list):
                model_name = [model_name]
            for model_one in model_name:
                model_path = os.path.join(path_dir_result, model_one)
                if not os.path.exists(model_path):
                    print(f"Result of Model:{model_one} does not exist")
                list_file_result = _get_model_result_files(model_path, prompt_mode)
                if list_file_result:
                    dict_model_result[model_one] = list_file_result
        else:
            for model_one in os.listdir(path_dir_result):
                # print(f"Now processing: {model_one}")
                model_path = os.path.join(path_dir_result, model_one)
                list_file_result = _get_model_result_files(model_path, prompt_mode)
                if list_file_result:
                    dict_model_result[model_one] = list_file_result

        return dict_model_result

    def evaluate_by_model(self, prompt_mode="direct", model_name=None, bootstrap=1000):
        """
        Evaluate the performance of each model.

        Input:
            bootstrap: boolean, default=1000, number of bootstrap samples

        Returns:
            dict: A dictionary containing performance metrics for each model.
        """
        if prompt_mode not in ["direct", "cot"] and not prompt_mode.endswith("shot"):
            raise ValueError(f"Invalid prompt mode: {prompt_mode}")

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
            # print(f"Load {file_result} with {len(list_dict_result)} samples.")

        list_label = self.get_label(list_dict_result, prompt_mode)
        list_pred = self.get_pred(list_dict_result, prompt_mode)
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
        # list_dict_performance = []
        # for args in list_args:
        #     dict_performance = self._performance_one_iteration(args)
        #     list_dict_performance.append(dict_performance)

        # Use a pool of workers to parallelize the bootstrapping process
        n_thread = 4
        with Pool(processes=n_thread) as pool:
            list_dict_performance = pool.map(self._performance_one_iteration, list_args)

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
