import os
import json
import regex
import numpy as np
from multiprocessing import Pool, cpu_count

# import self-defined modules
from .dataset import GeneralDataset, GeneralTask, GeneralEvaluation
from .config import (
    get_pred_none_clf,
    get_pred_none_clf_mul_label,
    get_pred_none_clf_mul_question,
)
from .process import process_text_clean
from .config import extract_cot_pred
from metric.classification import (
    calc_metrics_clf,
    get_arr_multi_hot,
    calc_metrics_clf_mul_label,
    compute_metrics_clf_mul_label,
)


class Task_clf_binary(GeneralEvaluation):
    """
    The class for classification tasks with binary classes
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.dict_label_map = {
            "no": 0,
            "yes": 1,
        }

        # Define the regex pattern for the binary label
        # Importanly, the order of the pattern is essential, the first pattern has higher priority which means it totally match the expected output
        self.list_pattern = [
            r"answer: (yes|no)",
            r"answer:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_label = []
        for dict_data in list_dict_data:
            string_label = process_text_clean(
                dict_data["output"], flag_lower=True, flag_punc_to_en=True
            )
            label = regex.search(
                self.list_pattern[0], string_label, regex.IGNORECASE
            ).group(1)
            label = self.dict_label_map[label]
            list_label.append(label)
        return list_label

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_clf(list_pred, list_label)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_pred = []
        for dict_data in list_dict_data:
            response = process_text_clean(
                dict_data["pred"], flag_lower=True, flag_punc_to_en=True
            )
            if "</think>" in response:
                response = response.split("</think>", 1)[1]
            if "cot" in prompt_mode:
                response = extract_cot_pred(response)
            # Initialize the prediction as -1
            pred = -1
            # Find the label by searching the list of patterns
            for pattern in self.list_pattern:
                result_found = regex.search(pattern, response, regex.IGNORECASE)
                if result_found:
                    # Transform the texual response to binary label
                    pred = self.dict_label_map[result_found.group(1)]
                    break
            # Append the prediction to the list of prediction
            # If the prediction is not found, append -1
            # The invalid prediction will be handled later
            # If the prediction is found, append the extracted label
            list_pred.append(pred)

        return list_pred

    def get_performance(self, list_pred, list_label):
        # single-label classification
        dict_performance = calc_metrics_clf(
            list_pred=list_pred,
            list_label=list_label,
        )

        return dict_performance


class Task_clf_mul_class(GeneralEvaluation):
    """
    The class for classification tasks with multiple classes
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the binary label
        self.list_pattern = [
            r"answer: ([a-e])",
            r"answer:?\s*([a-e])",
            r"\b([a-e])\b",
        ]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_label = []
        for dict_data in list_dict_data:
            string_label = process_text_clean(
                dict_data["output"], flag_lower=True, flag_punc_to_en=True
            )
            result_match = regex.search(
                self.list_pattern[0], string_label, regex.IGNORECASE
            )
            string_label = result_match.group(1)
            list_label.append(string_label)
        return list_label

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_clf(list_pred, list_label)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_pred = []
        for dict_data in list_dict_data:
            response = process_text_clean(
                dict_data["pred"], flag_lower=True, flag_punc_to_en=True
            )
            if "</think>" in response:
                response = response.split("</think>", 1)[1]
            if "cot" in prompt_mode:
                response = extract_cot_pred(response)
            pred = -1
            for pattern in self.list_pattern:
                result_found = regex.search(pattern, response, regex.IGNORECASE)
                if result_found:
                    pred = result_found.group(1)
                    break
            list_pred.append(pred)

        return list_pred

    def get_performance(self, list_pred, list_label):
        # single-label classification
        dict_performance = calc_metrics_clf(
            list_pred=list_pred,
            list_label=list_label,
        )

        return dict_performance


class Task_clf_mul_label(GeneralEvaluation):
    """
    The class for classification tasks with multiple labels in one response
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the multi-label
        # For example: answer: a, b, c, ...
        self.list_pattern = [
            r"answer: (.+)",
            r"answer:?\s*(.+)",
            r"\b(.+)\b",
        ]

        # Define the separator for the multi-label
        self.sep_label = ","

        # Define the list of labels
        # self.list_label = ["a", "b", ...]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_list_label = []
        for dict_data in list_dict_data:
            list_label = []
            string_result = process_text_clean(
                dict_data["output"], flag_lower=True, flag_punc_to_en=True
            )
            # Find the string of result by seaching the prefix pattern
            for pattern in self.list_pattern:
                result_match = regex.search(pattern, string_result, regex.IGNORECASE)
                if result_match:
                    string_result = result_match.group(1)
                    break
            # Split the string of result by the separator, get the list of labels
            for string_label in string_result.split(self.sep_label):
                string_label = string_label.strip()
                # Find the label by searching the list of labels
                if self.list_label:
                    for label in self.list_label:
                        if label == string_label:
                            list_label.append(label)
                            break
                # If the list of labels is not defined, directly use the string as label, such as for the normalizaiton task
                else:
                    list_label.append(string_label)
            list_list_label.append(list_label)
        return list_list_label

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_clf_mul_label(list_pred, list_label)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_list_pred = []
        for dict_data in list_dict_data:
            list_pred = []
            response = process_text_clean(
                dict_data["pred"], flag_lower=True, flag_punc_to_en=True
            )
            if "</think>" in response:
                response = response.split("</think>", 1)[1]
            if "cot" in prompt_mode:
                response = extract_cot_pred(response)
            for pattern in self.list_pattern:
                result_match = regex.search(pattern, response, regex.IGNORECASE)
                if result_match:
                    response = result_match.group(1)
                    break
            for string_label in response.split(self.sep_label):
                string_label = string_label.strip()
                string_label = string_label.replace('"', "").replace("'", "")
                if self.list_label:
                    for label in self.list_label:
                        if label in string_label:
                            list_pred.append(label)
                            break
                else:
                    list_pred.append(string_label)
            if len(list_pred) == 0:
                list_list_pred.append([-1])
            else:
                list_list_pred.append(list_pred)
        return list_list_pred

    def get_performance(self, list_pred, list_label):
        # single question - multiple-label classification
        dict_performance, array_pred, array_label, label_to_index = (
            calc_metrics_clf_mul_label(
                list_list_pred=list_pred,
                list_list_label=list_label,
            )
        )

        return dict_performance

    def get_performance_bootstrap(self, list_pred, list_label, list_failed, bootstrap):
        """
        Bootstrapping for multi-label classification tasks, only compute the array_pred and array_label once.

        Returns:
            list: A list of performance metrics dictionaries
        """

        array_pred, array_label, label_to_index = get_arr_multi_hot(
            list_list_pred=list_pred, list_list_label=list_label
        )

        # Initialize arguments for each iteration
        num_data = len(list_label)
        num_sample = len(list_label)

        # Initialize the list of arguments for each iteration
        list_args = []
        for _ in range(bootstrap):
            indices = np.random.choice(num_data, num_sample, replace=True)
            array_label_sample = array_label[indices]
            array_pred_sample = array_pred[indices]
            num_failed_sample = sum(list_failed[i] for i in indices)
            list_args.append(
                (
                    array_label_sample,
                    array_pred_sample,
                    label_to_index,
                    num_failed_sample,
                )
            )

        # Use a pool of workers to parallelize the bootstrapping process
        n_thread = 32
        with Pool(processes=n_thread) as pool:
            list_dict_performance = pool.map(
                _performance_one_iteration_clf_mul_label, list_args
            )

        return list_dict_performance

    def _performance_one_iteration(self, args):
        """
        Helper function for one bootstrap iteration for multi-label classification tasks.
        Args is a tuple of (array_label, array_pred, list_failed).
        """
        pass


def _performance_one_iteration_clf_mul_label(args):
    """
    Helper function for one bootstrap iteration for multi-label classification tasks.
    Args is a tuple of (array_label, array_pred, list_failed).
    """
    array_label_sample, array_pred_sample, label_to_index, num_failed_sample = args
    dict_performance = compute_metrics_clf_mul_label(
        array_pred_sample, array_label_sample, label_to_index
    )
    dict_performance["num_failed"] = num_failed_sample
    dict_performance["num_failed_ratio"] = round(
        num_failed_sample / len(array_label_sample) * 100, 2
    )
    return dict_performance


class Task_clf_mul_question(GeneralEvaluation):
    """
    The class for classification tasks with multiple questions in one response, each question has one label
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_label = "\n"
        # Define the regex pattern for the binary label
        self.dict_list_pattern = {
            "q1": [
                r"answer of question 1: ([a-e])",
                r"answer of question 1:?\s*([a-e])",
                r"\b(.+)\b",
            ],
            "q2": [
                r"answer of question 2: ([a-e])",
                r"answer of question 2:?\s*(.+)",
                r"\b(.+)\b",
            ],
        }

    def get_label(self, list_dict_data, prompt_mode="direct"):
        # expected output: "label_of_question_1, label_of_question_2, ..."
        list_list_label = [[] for _ in range(len(self.dict_list_pattern))]
        for dict_data in list_dict_data:
            string_label_all = process_text_clean(
                dict_data["output"], flag_lower=True, flag_punc_to_en=True
            )
            for string_label in string_label_all.split(self.sep_label):
                for idx, (key, list_pattern) in enumerate(
                    self.dict_list_pattern.items()
                ):
                    result_match = regex.search(
                        list_pattern[0], string_label, regex.IGNORECASE
                    )
                    label = result_match.group(1).strip()
                    list_list_label[idx].append(label)
        return list_list_label

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_clf_mul_question(list_pred, list_label)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        num_questions = len(self.dict_list_pattern)
        list_list_pred = []
        for dict_data in list_dict_data:
            list_label = [-1 for _ in range(num_questions)]
            response = process_text_clean(
                dict_data["pred"], flag_lower=True, flag_punc_to_en=True
            )
            if "</think>" in response:
                response = response.split("</think>", 1)[1]
            if "cot" in prompt_mode:
                response = extract_cot_pred(response)
            # The flag indicates whether the labels for this sample are found
            list_flag_found_label = [False for _ in range(num_questions)]
            for line_one in response.split(self.sep_label):
                # The flag indicates whether this response found the label
                flag_found_line = False
                for idx, (key, list_pattern) in enumerate(
                    self.dict_list_pattern.items()
                ):
                    # If this label is found, break the loop
                    if list_flag_found_label[idx]:
                        continue
                    for pattern in list_pattern:
                        result_match = regex.search(pattern, line_one, regex.IGNORECASE)
                        # If the label is found, append it to the list of prediction
                        if result_match:
                            label_one = result_match.group(1)
                            list_label[idx] = label_one
                            list_flag_found_label[idx] = True
                            flag_found_line = True
                            break
                    # If this line found a label, break the loop
                    if flag_found_line:
                        break
            list_list_pred.append(list_label)
        return list_list_pred

    def get_performance_questions(self, list_pred, list_label):
        # multiple-question - single label classification
        num_question = len(list_label[0])
        name_question = list(self.dict_list_pattern.keys())
        dict_performance_question = {}, {}
        for idx_q, name_q in enumerate(name_question):
            list_pred_q = [pred[idx_q] for pred in list_pred]
            list_label_q = [label[idx_q] for label in list_label]
            dict_performance_one = calc_metrics_clf(
                list_pred=list_pred_q,
                list_label=list_label_q,
            )
            # Save the performance for each question
            dict_performance_question[name_q] = dict_performance_one

        return dict_performance_question

    def get_performance(self, list_pred, list_label):
        # multiple-question - single label classification
        num_question = len(list_label[0])
        name_question = list(self.dict_list_pattern.keys())
        dict_performance_question = self.get_performance_questions(
            list_pred=list_pred, list_label=list_label
        )
        # Get the average performance for all questions
        dict_performance_avg = {}
        for name_q in name_question:
            for key, value in dict_performance_question[name_q].items():
                dict_performance_avg[key] = dict_performance_avg.get(key, 0) + value
        for key in dict_performance_avg:
            dict_performance_avg[key] /= num_question

        return dict_performance_avg


class Task_clf_xxx(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)


# Detailed task definition


# 20
class Task_clf_clinical_records_from_the_Mexican_Social_Security_Institute(
    Task_clf_mul_class
):
    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the binary label
        self.list_pattern = [
            r"answer:\s*(pneumonia|thromboembolism|control)",
            r"\b(pneumonia|thromboembolism|control)\b",
        ]


# 46
class Task_clf_Japanese_Case_Reports(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the binary label
        self.list_pattern = [r"similarity score:\s*([0-5])", r"\b([0-5])\b"]


# 51
class Task_clf_MEDIQA_2019_Task2_RQE(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.dict_label_map = {
            "false": 0,
            "true": 1,
        }

        # Define the regex pattern for the binary label
        self.list_pattern = [r"answer:\s*(true|false)", r"\b(true|false)\b"]


# 55
class Task_clf_MedNLI(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"relation:\s*(entailment|contradiction|neutral)",
            r"\b(entailment|contradiction|neutral)\b",
        ]


# 57
class Task_clf_MedSTS(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [r"similarity score:\s*([0-5])", r"\b([0-5])\b"]


# 62
class Task_clf_mtsamples(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [r"answer: (.+)", r"\b(.+)\b"]

        self.sep_label = ","

        list_label = [
            "Allergy / Immunology",
            "Autopsy",
            "Bariatrics",
            "Cardiovascular / Pulmonary",
            "Chiropractic",
            "Consult - History and Phy.",
            "Cosmetic / Plastic Surgery",
            "Dentistry",
            "Dermatology",
            "Diets and Nutritions",
            "Discharge Summary",
            "ENT - Otolaryngology",
            "Emergency Room Reports",
            "Endocrinology",
            "Gastroenterology",
            "General Medicine",
            "Hematology - Oncology",
            "Hospice - Palliative Care",
            "IME-QME-Work Comp etc.",
            "Lab Medicine - Pathology",
            "Letters",
            "Nephrology",
            "Neurology",
            "Neurosurgery",
            "Obstetrics / Gynecology",
            "Office Notes",
            "Ophthalmology",
            "Orthopedic",
            "Pain Management",
            "Pediatrics - Neonatal",
            "Physical Medicine - Rehab",
            "Podiatry",
            "Psychiatry / Psychology",
            "Radiology",
            "Rheumatology",
            "SOAP / Chart / Progress Notes",
            "Sleep Medicine",
            "Speech - Language",
            "Surgery",
            "Urology",
        ]

        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)

        self.list_label = [x.lower() for x in list_label]


# 76-2
class Task_clf_MTS_Dialog_MEDIQA_2023_sum_task_A(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        # Define the regex pattern for the binary label
        self.list_pattern = [
            r"topic:\s*(family history/social history|history of patient illness|past medical history|chief complaint|past surgical history|allergy|review of systems|medications|assessment|exam|diagnosis|disposition|plan|emergency department course|immunizations|imaging|gynecologic history|procedures|other history|labs)",
            r"\b(family history/social history|history of patient illness|past medical history|chief complaint|past surgical history|allergy|review of systems|medications|assessment|exam|diagnosis|disposition|plan|emergency department course|immunizations|imaging|gynecologic history|procedures|other history|labs)\b",
        ]


# 78
class Task_nor_RuMedTop3(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [r"ICD-10: (.+)", r".+:\s*(.+)"]


# 79
class Task_nor_RuMedSymptomRec(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [r"symptom: (.+)", r".+:\s*(.+)"]


# 80
class Task_clf_RuMedDaNet(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.dict_label_map = {
            "no": 0,
            "yes": 1,
        }

        # Define the regex pattern for the binary label
        self.list_pattern = [r"answer:\s*(yes|no)", r"\b(yes|no)\b"]


# 90-6
class Task_clf_n2c2_2014_Heart_Disease_Challenge_Family_History(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.dict_label_map = {
            "not present": 0,
            "present": 1,
        }

        # Define the regex pattern for the binary label
        self.list_pattern = [
            r"answer:\s*(present|not present)",
            r"\b(present|not present)\b",
        ]


# 90-7
class Task_clf_n2c2_2014_Heart_Disease_Challenge_Smoker(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"answer:\s*(current|past|ever|never|unknown)",
            r"\b(current|past|ever|never|unknown)\b",
        ]


# 93
class Task_clf_RuMedNLI(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"relation:\s*(entailment|contradiction|neutral)",
            r"\b(entailment|contradiction|neutral)\b",
        ]


class Task_clf_ADE_ADE_identification(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        # adverse drug effect: Yes
        # adverse drug effect: No
        self.list_pattern = [
            r"adverse drug effect: (yes|no)",
            r"adverse drug effect:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_CHIP_CTC(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        list_label = [
            "疾病",
            "症状-患者感受",
            "体征-医生检测",
            "怀孕相关",
            "肿瘤进展",
            "疾病分期",
            "过敏耐受",
            "器官组织状态",
            "预期寿命",
            "口腔相关",
            "药物",
            "治疗或手术",
            "设备",
            "护理",
            "诊断",
            "实验室检查",
            "风险评估",
            "受体状态",
            "年龄",
            "特殊病人特征",
            "读写能力",
            "性别",
            "教育情况",
            "居住情况",
            "种族",
            "知情同意",
            "参与其它试验",
            "研究者决定",
            "能力",
            "伦理审查",
            "依存性",
            "成瘾行为",
            "睡眠",
            "锻炼",
            "饮食",
            "酒精使用",
            "性取向",
            "吸烟状况",
            "献血",
            "病例来源",
            "残疾群体",
            "健康群体",
            "数据可及性",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"clinical trial criterion: ({str_label})",
            rf"clinical trial criterion:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_IMCS_V2_DAC(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        list_label = [
            "提问-症状",
            "提问-病因",
            "提问-基本信息",
            "提问-已有检查和治疗",
            "告知-用药建议",
            "告知-就医建议",
            "告知-注意事项",
            "诊断",
            "告知-症状",
            "告知-病因",
            "告知-基本信息",
            "告知-已有检查和治疗",
            "提问-用药建议",
            "提问-就医建议",
            "提问-注意事项",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        str_label = "|".join(list_label)

        self.list_pattern = [
            # dialogue
            rf"dialogue act: ({str_label})",
            rf"dialogue act:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_Brain_MRI_AIS(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"AIS: (yes|no)",
            r"AIS:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_C_EMRS(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "胃息肉",
            "泌尿道感染",
            "慢性阻塞性肺病",
            "痛风",
            "胃溃疡",
            "高血压",
            "哮喘",
            "胃炎",
            "心律失常",
            "糖尿病",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        str_label = "|".join(list_label)

        # self.list_pattern = [rf"\b({str_label})\b"]
        self.list_pattern = [
            rf"diagnosis: ({str_label})",
            rf"diagnosis:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_CLISTER(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        # similarity score: [0-5]
        self.list_pattern = [
            r"similarity score: ([0-5])",
            r"similarity score:?\s*([0-5])",
            r"\b([0-5])\b",
        ]


class Task_clf_Brateca_mortality(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"survival: (yes|no)",
            r"survival:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_Brateca_hospitalization(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"Hospitalization > 7 days: (yes|no)",
            r"Hospitalization > 7 days:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_CARES_area(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)

        list_label = ["Columna", "Neuro", "Musculoskeletal", "Body"]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]
        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"anatomical area: ({str_label})",
            rf"anatomical area:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_CARES_icd10_chapter(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        # expected output: "ICD-10 chapter: I, II, III"
        self.list_pattern = [
            r"ICD-10 chapter: (.+)",
            r"ICD-10 chapter:?\s*(.+)",
            r"\b(.+)\b",
        ]

        self.sep_label = ","

        list_label = [
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
            "XVII",
            "XVIII",
            "XIX",
            "XX",
            "XXI",
            "XXII",
        ]

        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)

        self.list_label = [x.lower() for x in list_label]


class Task_clf_Clinical_Notes_UPMC(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"affirmed: (yes|no)",
            r"affirmed:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_CLIP(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"action items: (.+)",
            r"action items:?\s*(.+)",
            r"\b(.+)\b",
        ]

        self.sep_label = ","

        list_label = [
            "Patient Instructions",
            "Appointment",
            "Medications",
            "Lab",
            "Procedure",
            "Imaging",
            "Other",
        ]

        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)

        self.list_label = [x.lower() for x in list_label]


class Task_clf_Dial_Med(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            r"medication: (.+)",
            r"medication:?\s*(.+)",
            r"\b(.+)\b",
        ]

        self.sep_label = ","

        list_label = [
            "酮康唑",
            "板蓝根",
            "右美沙芬",
            "莫沙必利",
            "风寒感冒颗粒",
            "双黄连口服液",
            "蒲地蓝消炎口服液",
            "水飞蓟素",
            "米诺环素",
            "氯雷他定",
            "布地奈德",
            "苏黄止咳胶囊",
            "胶体果胶铋",
            "哈西奈德",
            "谷胱甘肽",
            "二硫化硒",
            "泰诺",
            "硫磺皂",
            "对乙酰氨基酚",
            "奥司他韦",
            "甘草酸苷",
            "红霉素",
            "西替利嗪",
            "克拉霉素",
            "氢化可的松",
            "复方甲氧那明胶囊",
            "三九胃泰",
            "替诺福韦",
            "健胃消食片",
            "炉甘石洗剂",
            "蒙脱石",
            "曲美布汀",
            "阿奇霉素",
            "扶正化瘀胶囊",
            "依巴斯汀",
            "感冒灵",
            "他克莫司",
            "氨溴索",
            "康复新液",
            "多烯磷脂酰胆碱",
            "恩替卡韦",
            "桉柠蒎肠溶软胶囊",
            "曲安奈德",
            "甘草片",
            "左氧氟沙星",
            "奥美拉唑",
            "铝镁化合物",
            "复方消化酶",
            "头孢类",
            "甲氧氯普胺",
            "地塞米松",
            "美沙拉秦",
            "双环醇",
            "肠炎宁",
            "抗病毒颗粒",
            "阿莫西林",
            "川贝枇杷露",
            "谷氨酰胺",
            "山莨菪碱",
            "阿达帕林",
            "孟鲁司特",
            "糠酸莫米松",
            "快克",
            "布洛芬",
            "益生菌",
            "通窍鼻炎颗粒",
            "阿昔洛韦",
            "生理氯化钠溶液",
            "连花清瘟胶囊",
            "黄连素",
        ]

        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)

        self.list_label = [x.lower() for x in list_label]


class Task_clf_EHRQA_primary_department(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = """[\"儿科\", \"妇产科\", \"传染病科\", \"皮肤性病科\", \"外科\", \"内科\", \"五官科\"]"""
        list_label = json.loads(list_label)
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]

        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"department: ({str_label})",
            rf"department:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_EHRQA_sub_department(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = """[\"泌尿外科\", \"性病科\", \"胃肠外科\", \"肝胆外科\", \"骨科\", \"小儿内科\", \"妇科\", \"小儿精神科\", \"普外科\", \"其他传染病\", \"皮肤病\", \"消化内科\", \"风湿免疫科\", \"口腔科\", \"产科\", \"肝病科\", \"肛肠外科\", \"肾内科\", \"眼科\", \"血管外科\", \"小儿外科\", \"乳腺外科\", \"心胸外科\", \"烧伤科\", \"血液科\", \"内分泌科\", \"新生儿科\", \"神经外科\", \"呼吸内科\"]"""
        list_label = json.loads(list_label)
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]

        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"department: ({str_label})",
            rf"department:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_GOUT_CC_consensus(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = ["Yes", "No"]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]

        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"Gout flare: ({str_label})",
            rf"Gout flare:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_GOUT_CC_predict(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = ["Yes", "No"]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]

        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"Gout flare: ({str_label})",
            rf"Gout flare:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_IFMIR_IncidentType(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "Wrong Drug",
            "Wrong Form",
            "Wrong Mode",
            "Wrong Strength amount",
            "Wrong Strength rate",
            "Wrong Strength concentration",
            "Wrong Timing",
            "Wrong Date",
            "Wrong Duration",
            "Wrong Frequency",
            "Wrong Dosage",
            "Wrong Route",
            "Others",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]

        str_label = "|".join(list_label)

        self.list_pattern = [
            rf"incident type: ({str_label})",
            rf"incident type:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_mimic_iv_CDM(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "Appendicitis",
            "Cholecystitis",
            "Diverticulitis",
            "Pancreatitis",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]
        str_label = "|".join(list_label)
        self.list_pattern = [
            rf"diagnosis: ({str_label})",
            rf"diagnosis:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_mimic_iii_outcome_LoS(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "A",
            "B",
            "C",
            "D",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]
        str_label = "|".join(list_label)
        self.list_pattern = [
            rf"length of stay: ({str_label})",
            rf"length of stay:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_mimic_iii_outcome_Mortality(Task_clf_binary):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.list_pattern = [
            r"in-hospital mortality: (yes|no)",
            r"in-hospital mortality:?\s*(yes|no)",
            r"\b(yes|no)\b",
        ]


class Task_clf_mimic_iv_DiReCT_Dis(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "Acute Coronary Syndrome",
            "Heart Failure",
            "Gastro-oesophageal Reflux Disease",
            "Pulmonary Embolism",
            "Hypertension",
            "Peptic Ulcer Disease",
            "Stroke",
            "Gastritis",
            "Multiple Sclerosis",
            "Adrenal Insufficiency",
            "Pneumonia",
            "Chronic Obstructive Pulmonary Disease",
            "Aortic Dissection",
            "Asthma",
            "Diabetes",
            "Pituitary Disease",
            "Alzheimer",
            "Atrial Fibrillation",
            "Thyroid Disease",
            "Cardiomyopathy",
            "Epilepsy",
            "Upper Gastrointestinal Bleeding",
            "Tuberculosis",
            "Migraine",
            "Hyperlipidemia",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]
        str_label = "|".join(list_label)
        self.list_pattern = [
            rf"diagnosis: ({str_label})",
            rf"diagnosis:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


class Task_clf_mimic_iv_DiReCT_PDD(Task_clf_mul_class):
    def __init__(self, args, task):
        super().__init__(args, task)
        list_label = [
            "Heart Failure",
            "Gastro-oesophageal Reflux Disease",
            "Hypertension",
            "Non-ST-Elevation Myocardial Infarction",
            "Relapsing-Remitting Multiple Sclerosis",
            "Unstable Angin",
            "Low-risk Pulmonary Embolism",
            "Gastric Ulcers",
            "Chronic Obstructive Pulmonary Disease",
            "Bacterial Pneumonia",
            "ST-Elevation Myocardial Infarction",
            "Hemorrhagic Stroke",
            "Acute Gastritis",
            "Ischemic Stroke",
            "Submassive Pulmonary Embolism",
            "Pituitary Macroadenoma",
            "Secondary Adrenal Insufficiency",
            "Alzheimer",
            "Type B Aortic Dissection",
            "Duodenal Ulcers",
            "Chronic Atrophic Gastritis",
            "Paroxysmal Atrial Fibrillation",
            "Primary Adrenal Insufficiency",
            "Upper Gastrointestinal Bleeding",
            "Type I Diabetes",
            "Type II Diabetes",
            "Chronic Non-atrophic Gastritis",
            "Type A Aortic Dissection",
            "Non-epileptic Seizure",
            "Tuberculosis",
            "Viral Pneumonia",
            "Severe Asthma Exacerbation",
            "Hyperthyroidism",
            "Dilated Cardiomyopathy",
            "Congenital Adrenal Hyperplasia",
            "Epilepsy",
            "Non-Allergic Asthma",
            "Migraine With Aura",
            "Secondary Progressive Multiple Sclerosis",
            "Hypothyroidism",
            "Massive Pulmonary Embolism",
            "Hyperlipidemia",
            "Restrictive Cardiomyopathy",
            "Asthma",
            "COPD Asthma",
            "Hypertrophic Cardiomyopathy",
            "Persistent Atrial Fibrillation",
            "Thyroid Nodules",
            "Primary Progressive Multiple Sclerosis",
            "Allergic Asthma",
            "Cough-Variant Asthma",
            "Arrhythmogenic Right Ventricular Cardiomyopathy",
            "Migraine Without Aura",
            "Thyroiditis",
            "Pituitary Microadenoma",
        ]
        list_label = sorted(list_label, key=lambda x: len(x), reverse=True)
        list_label = [x.lower() for x in list_label]
        str_label = "|".join(list_label)
        self.list_pattern = [
            rf"diagnosis: ({str_label})",
            rf"diagnosis:?\s*({str_label})",
            rf"\b({str_label})\b",
        ]


# normalization-code -> Task_ext_nor
class Task_nor_Cantemist_CODING(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            # ICD-10 block: N50, N70
            r"Morphology code: (.+)",
            r"Morphology code:?\s*(.+)",
            r"(.+)",
        ]

        self.sep_label = ","

        self.list_label = None


class Task_nor_CARES_icd10_block(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            # ICD-10 block: N50, N70
            r"ICD-10 block: (.+)",
            r"ICD-10 block:?\s*(.+)",
            r"(.+)",
        ]

        self.sep_label = ","

        self.list_label = None


class Task_nor_CARES_icd10_sub_block(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            # ICD-10 sub block: N50.5, N70.7
            r"ICD-10 sub block: (.+)",
            r"ICD-10 sub block:?\s*(.+)",
            r"\b(.+)\b",
        ]

        self.sep_label = ","

        self.list_label = None


class Task_nor_CHIP_CDN(Task_clf_mul_label):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_pattern = [
            # Normalized terms: 前列腺未特指的疾患, 高甘油三酯血症
            r"Normalized terms: (.+)",
            r"Normalized terms:?\s*(.+)",
            r"\b(.+)\b",
        ]

        self.sep_label = ","

        self.list_label = None
