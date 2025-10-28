import regex
import numpy as np

# import self-defined modules
from .dataset import GeneralEvaluation
from .config import get_pred_none_gen, get_pred_none_gen_qa_mul
from .process import process_pred, extract_pred
from metric.generation import calc_metrics_gen


class Task_gen(GeneralEvaluation):
    """
    The class for generation tasks
    """

    def __init__(self, args, task):
        super().__init__(args, task)
        # check if the task has multiple ground truth labels
        if "output_list" in self.dataset_test[0]:
            print("Notice: this task has multiple reference labels")

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_label = []
        for dict_data in list_dict_data:
            if "output_list" not in dict_data:
                label = dict_data["output"]
                label = process_pred(label, flag_lower=True, flag_punc_to_en=True)
            else:
                label = dict_data["output_list"]
                label = [
                    process_pred(l, flag_lower=True, flag_punc_to_en=True)
                    for l in label
                ]
            list_label.append(label)

        return list_label

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_gen(list_pred)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_pred = []
        for dict_data in list_dict_data:
            response = process_pred(
                dict_data["pred"], flag_lower=True, flag_punc_to_en=True
            )
            response = extract_pred(response, prompt_mode=prompt_mode)
            if response == "":
                list_pred.append(-1)
            else:
                list_pred.append(response)

        return list_pred

    def get_performance(self, list_pred, list_label):
        # generation
        dict_performance, dict_performance_sample = calc_metrics_gen(
            list_pred=list_pred,
            list_label=list_label,
            lang=self.language,
        )

        return dict_performance, dict_performance_sample

    def get_performance_bootstrap(self, list_pred, list_label, list_failed, bootstrap):
        """
        Bootstrapping for generation tasks, only compute metrics once.

        Returns:
            list: A list of performance metrics dictionaries.
        """

        dict_metrics_avg, dict_metrics_sample = self.get_performance(
            list_pred=list_pred, list_label=list_label
        )

        list_dict_performance = []
        for _ in range(bootstrap):
            indices = np.random.choice(len(list_label), len(list_label), replace=True)
            dict_performance = {}
            for key, list_metric_sample in dict_metrics_sample.items():
                list_metric_sample = [list_metric_sample[i] for i in indices]
                mean_value = np.mean(list_metric_sample)
                dict_performance[key] = mean_value
            # Add the number of failed prediction, which the pred==''
            num_failed_sample = sum([list_failed[i] for i in indices])
            dict_performance["num_failed"] = num_failed_sample
            dict_performance["num_failed_ratio"] = round(
                num_failed_sample / len(indices) * 100, 2
            )
            list_dict_performance.append(dict_performance)
        return list_dict_performance


class Task_gen_extract(Task_gen):
    """
    The class for extraction-QA tasks, which extract the short text from the input text.
    For example, "Answer: the patient has a fever, the patient has a cough"
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the entity
        self.list_pattern = [
            # Answer: the patient has a fever, the patient has a cough
            r"Answer: (.+)",
            r".+:\s*(.+)",
            r"(.+)",
        ]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_answer = []
        for dict_data in list_dict_data:
            line = process_pred(dict_data["output"]).strip()
            result = regex.search(self.list_pattern[0], line, regex.IGNORECASE)
            list_answer.append(result.group(1).strip())

        return list_answer

    def get_pred_none(self, list_pred, list_label):
        list_pred = ["" if pred == -1 else pred for pred in list_pred]
        return list_pred

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_pred = []
        for idx_data, dict_data in enumerate(list_dict_data):
            if dict_data["pred"].strip() != "":
                # Split the response by event
                response = process_pred(
                    dict_data["pred"], flag_lower=True, flag_punc_to_en=True
                )
                response = extract_pred(response, prompt_mode=prompt_mode)
                flag_match = False
                for pattern in self.list_pattern:
                    result = regex.search(pattern, response, regex.IGNORECASE)
                    if result:
                        list_pred.append(result.group(1).strip())
                        flag_match = True
                        break
                if not flag_match:
                    list_pred.append(-1)
            else:
                list_pred.append(-1)

        return list_pred


class Task_gen_extract_mul(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the regex pattern for the entity
        self.list_question_pattern = [
            {
                "idx": 0,
                "target": "q1",
                "list_pattern": [
                    # evidence of q1: ...;
                    r"evidence of q1: ([^;]+)",
                    r"q1:\s*([^;]+)",
                ],
            },
            {
                "idx": 1,
                "target": "q2",
                "list_pattern": [
                    # evidence of q2: ...;
                    r"evidence of q2: ([^;]+)",
                    r"q2:\s*([^;]+)",
                ],
            },
        ]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_list_answer = []
        num_question = len(self.list_question_pattern)
        for dict_data in list_dict_data:
            list_answer = [-1] * num_question
            list_line = process_pred(dict_data["output"]).split(self.sep_event)
            list_line = [line.strip() for line in list_line if line.strip() != ""]
            for line_one in list_line:
                for idx, dict_pattern in enumerate(self.list_question_pattern):
                    result = regex.search(
                        dict_pattern["list_pattern"][0], line_one, regex.IGNORECASE
                    )
                    if result:
                        list_answer[idx] = result.group(1).strip()
                        break
            list_list_answer.append(list_answer)

        return list_list_answer

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_gen_qa_mul(list_pred)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_list_pred = []
        num_question = len(self.list_question_pattern)
        for idx_data, dict_data in enumerate(list_dict_data):
            list_answer = [-1] * num_question
            if dict_data["pred"].strip() != "":
                response = process_pred(
                    dict_data["pred"], flag_lower=True, flag_punc_to_en=True
                )
                response = extract_pred(response, prompt_mode=prompt_mode)
                # Split the response by event
                list_line = response.split(self.sep_event)
                # Filter the invalid entity
                list_line = [line.strip() for line in list_line if line.strip() != ""]
                list_question_found = []
                for line_one in list_line:
                    for idx, dict_pattern in enumerate(self.list_question_pattern):
                        if dict_pattern["target"] in list_question_found:
                            continue
                        flag_match = False
                        for pattern in dict_pattern["list_pattern"]:
                            result = regex.search(pattern, line_one, regex.IGNORECASE)
                            if result:
                                str_answer = result.group(1).strip()
                                list_answer[idx] = str_answer
                                flag_match = True
                                list_question_found.append(dict_pattern["target"])
                                break
                        if flag_match:
                            break
            list_list_pred.append(list_answer)

        return list_list_pred

    def get_performance(self, list_pred, list_label):
        # average the performance for all questions
        dict_question_performance = {}
        dict_question_metric_sample = {}
        for idx_question, dict_question_pattern in enumerate(
            self.list_question_pattern
        ):
            list_pred_q = [pred[idx_question] for pred in list_pred]
            list_label_q = [label[idx_question] for label in list_label]
            # dict_performance: {"bleu": 0.5, "rouge": 0.6, "meteor": 0.7}
            # dict_performance_sample: {"bleu": [0.5, 0.6, 0.7], "rouge": [0.6, 0.7, 0.8], "meteor": [0.7, 0.8, 0.9]}
            dict_performance, dict_metric_sample = calc_metrics_gen(
                list_pred=list_pred_q,
                list_label=list_label_q,
                lang=self.language,
            )
            # Save the performance for each question
            name_q = dict_question_pattern["target"]
            dict_question_performance[name_q] = dict_performance
            dict_question_metric_sample[name_q] = dict_metric_sample

        # Get the average performance for all questions
        dict_performance_avg = {}
        for key in dict_performance.keys():
            list_value = [
                dict_question_performance[name_q][key]
                for name_q in dict_question_performance
            ]
            dict_performance_avg[key] = sum(list_value) / len(list_value)

        return dict_performance_avg, dict_question_metric_sample

    def get_performance_bootstrap(self, list_pred, list_label, list_failed, bootstrap):
        """
        Bootstrapping for generation tasks, only compute metrics once.

        Returns:
            list: A list of performance metrics dictionaries.
        """

        dict_metrics_avg, dict_question_metric_sample = self.get_performance(
            list_pred=list_pred, list_label=list_label
        )

        num_question = len(self.list_question_pattern)
        list_dict_performance = []
        for _ in range(bootstrap):
            indices = np.random.choice(len(list_label), len(list_label), replace=True)
            # average the performance for all questions
            dict_performance = {}
            for name_q, dict_metric_sample in dict_question_metric_sample.items():
                for metric, list_metric_sample in dict_metric_sample.items():
                    list_metric_q = [list_metric_sample[i] for i in indices]
                    # Calculate the mean value of the metric for the question
                    mean_value_q = np.mean(list_metric_q)
                    if metric not in dict_performance:
                        dict_performance[metric] = mean_value_q
                    else:
                        dict_performance[metric] += mean_value_q

            for metric in dict_performance:
                dict_performance[metric] = dict_performance[metric] / num_question

            # Add the number of failed prediction, which the pred==''
            num_failed_sample = sum([list_failed[i] for i in indices])
            dict_performance["num_failed"] = num_failed_sample
            dict_performance["num_failed_ratio"] = round(
                num_failed_sample / len(indices) * 100, 2
            )

            list_dict_performance.append(dict_performance)

        return list_dict_performance


class Task_gen_xxx(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_MTS_Dialog_MEDIQA_2023_chat_task_A(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_MTS_Dialog_MEDIQA_2023_sum_task_B(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_MedDG(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_IMCS_V2_MRG(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_cMedQA(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_EHRQA_qa(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_icliniq(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_HealthCareMagic(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_gen_mimic_iv_BHC(Task_gen):
    def __init__(self, args, task):
        super().__init__(args, task)


# extraction-QA -> Task_ext_qa
class Task_gen_CAS_evidence(Task_gen_extract_mul):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the regex pattern for the entity
        self.list_question_pattern = [
            {
                "idx": 0,
                "target": "genre",
                "list_pattern": [
                    # evidence of genre: ...;
                    r"evidence of genre: ([^;]+)",
                    r"genre:\s*([^;]+)",
                ],
            },
            {
                "idx": 1,
                "target": "origine",
                "list_pattern": [
                    # evidence of origine: ...;
                    r"evidence of origine: ([^;]+)",
                    r"origine:\s*([^;]+)",
                ],
            },
            {
                "idx": 2,
                "target": "issue",
                "list_pattern": [
                    # evidence of issue: ...;
                    r"evidence of issue: ([^;]+)",
                    r"issue:\s*([^;]+)",
                ],
            },
        ]
