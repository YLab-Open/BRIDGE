import os
import json
import regex
import numpy as np

# import self-defined modules
from .dataset import GeneralDataset, GeneralTask, GeneralEvaluation
from .config import get_pred_none_ext, extract_cot_pred
from .process import process_text_clean
from metric.extraction import (
    calc_metrics_ext,
    ext_compute_overall_metrics,
    calc_metrics_ext_qa,
)


class Task_ext(GeneralEvaluation):
    """
    The class for extraction tasks
    """

    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.num_subject = len(self.list_field_subject)
        # Define the regex pattern for the entity
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r"entity:(.+), type:(.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_list_dict_entity = []
        for dict_data in list_dict_data:
            list_dict_entity = []
            list_line = process_text_clean(dict_data["output"]).split(self.sep_event)
            list_line = [line.strip() for line in list_line if line.strip() != ""]
            for line_one in list_line:
                result = regex.search(self.list_pattern[0], line_one, regex.IGNORECASE)
                # If the entity is found, extract the entity and the features
                # The group(1)-group(num_subject) is the subject, the group(num_subject+1)-group(num_subject+num_feature) is the features
                dict_entity = {}
                for idx, field_subject in enumerate(self.list_field_subject):
                    try:
                        dict_entity[field_subject] = result.group(idx + 1).strip()
                    except:
                        print(line_one)
                        raise ValueError("Error in get_label")
                # Extract the features
                for idx, field_feature in enumerate(self.list_field_feature):
                    dict_entity[field_feature] = result.group(
                        idx + self.num_subject + 1
                    ).strip()
                list_dict_entity.append(dict_entity)
            list_list_dict_entity.append(list_dict_entity)

        return list_list_dict_entity

    def get_pred_none(self, list_pred, list_label):
        return get_pred_none_ext(list_pred)

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_list_dict_entity = []
        for idx_data, dict_data in enumerate(list_dict_data):
            list_dict_entity = []
            if dict_data["pred"].strip() != "":
                response = process_text_clean(
                    dict_data["pred"], flag_lower=True, flag_punc_to_en=True
                )
                if "</think>" in response:
                    response = response.split("</think>", 1)[1]
                if "cot" in prompt_mode:
                    response = extract_cot_pred(response)
                # Split the response by event
                list_line = []
                for line in response.split(self.sep_event):
                    line = line.strip()
                    if not line:
                        continue
                    if line.count(";") > 1:
                        parts = line.split(";")
                        list_line.extend(
                            f"{part.strip()};" for part in parts if part.strip()
                        )
                    else:
                        list_line.append(line)
                # Search the entity by regex for each line
                for line_one in list_line:
                    # Search the entity by regex for each pattern
                    dict_entity = {}
                    for pattern in self.list_pattern:
                        result = regex.search(pattern, line_one, regex.IGNORECASE)
                        if result:
                            # Extract the subject
                            for idx, field_subject in enumerate(
                                self.list_field_subject
                            ):
                                dict_entity[field_subject] = result.group(
                                    idx + 1
                                ).strip()
                            # Extract the features
                            for idx, field_feature in enumerate(
                                self.list_field_feature
                            ):
                                dict_entity[field_feature] = result.group(
                                    idx + self.num_subject + 1
                                ).strip()
                            list_dict_entity.append(dict_entity)
                            # one entity is found, skip the other patterns
                            break
            # If no entity is found, set the entity to -1
            if len(list_dict_entity) > 0:
                list_list_dict_entity.append(list_dict_entity)
            else:
                list_list_dict_entity.append([-1])

        return list_list_dict_entity

    def get_performance(self, list_pred, list_label):
        # extraction task
        # list_pred, list_label: list of list of dict, each dict contains the entity field and one or more attribute fields
        dict_performance, dict_performance_sample = calc_metrics_ext(
            list_pred=list_pred,
            list_label=list_label,
            list_field_subject=self.list_field_subject,
        )

        return dict_performance, dict_performance_sample

    def get_performance_bootstrap(self, list_pred, list_label, list_failed, bootstrap):
        """
        Bootstrapping for extraction tasks, only compute metrics once.

        Returns:
            list: A list of performance metrics dictionaries.
        """

        dict_metrics_avg, dict_metrics_sample = self.get_performance(
            list_pred=list_pred, list_label=list_label
        )

        list_dict_performance = []
        for _ in range(bootstrap):
            # sample the data
            indices = np.random.choice(len(list_label), len(list_label), replace=True)
            # count the metrics
            dict_count = {}
            for key, list_metric_sample in dict_metrics_sample.items():
                dict_count[key] = [list_metric_sample[i] for i in indices]

            # Compute metrics
            dict_performance = ext_compute_overall_metrics(dict_count)

            # Add the number of failed prediction
            num_failed_sample = sum([list_failed[i] for i in indices])
            dict_performance["num_failed"] = num_failed_sample
            dict_performance["num_failed_ratio"] = round(
                num_failed_sample / len(indices) * 100, 2
            )
            list_dict_performance.append(dict_performance)

        return list_dict_performance


# Simple task definition
class Task_ext_xxx(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


# Detailed task definition
# 3
class Task_ext_BARR2_short_long(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["abbreviation"]
        self.list_field_feature = ["definition"]
        self.list_pattern = [
            r"abbreviation: (.+), definition: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_BARR2_resolution(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["abbreviation"]
        self.list_field_feature = ["definition"]
        self.list_pattern = [
            r"abbreviation: (.+), definition: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 17
class Task_ext_CLEF_ICD_10_CM(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["diagnosis"]
        self.list_field_feature = ["ICD-10-CM"]
        self.list_pattern = [
            r"diagnosis: (.+), ICD-10-CM: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_CLEF_ICD_10_PCS(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["procedure"]
        self.list_field_feature = ["ICD-10-PCS"]
        self.list_pattern = [
            r"procedure: (.+), ICD-10-PCS: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 21
class Task_ext_CLINpt(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 27
class Task_ext_DiSMed(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 35
class Task_ext_n2c2_2006_De_Identification(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["phi context"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"phi context: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 37
class Task_ext_i2b2_2009_Medication_Extraction_Challenge(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["medication"]
        self.list_field_feature = [
            "dosage",
            "mode",
            "frequency",
            "duration",
            "reason",
            "list/narrative",
        ]
        self.list_pattern = [
            r"medication: (.+), dosage: (.+), mode: (.+), frequency: (.+), duration: (.+), reason: (.+), list/narrative: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 38
class Task_ext_i2b2_2010_Relations_Challenge_concept(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_i2b2_2010_Relations_Challenge_assertion(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["problem"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"problem: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_i2b2_2010_Relations_Challenge_relation(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity_1", "entity_2"]
        self.list_field_feature = ["relation"]
        self.list_pattern = [
            r"entity_1: (.+), entity_2: (.+), relation: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 40
class Task_ext_i2b2_2012_Temporal_Relations_Challenge_event(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["clinical event"]
        self.list_field_feature = ["type", "modality", "polarity"]
        self.list_pattern = [
            r"clinical event: (.+), type: (.+), modality: (.+), polarity: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_i2b2_2012_Temporal_Relations_Challenge_time(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["temporal expression"]
        self.list_field_feature = ["type", "value", "modifier"]
        self.list_pattern = [
            r"temporal expression: (.+), type: (.+), value: (.+), modifier: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_i2b2_2012_Temporal_Relations_Challenge_relation(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = [
            "clinical event a",
            "clinical event/temporal expression b",
        ]
        self.list_field_feature = ["temporal relation"]
        self.list_pattern = [
            r"clinical event a: (.+), clinical event/temporal expression b: (.+), temporal relation: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 41
class Task_ext_n2c2_2014_De_identification(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["phi context"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"phi context: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 48
class Task_ext_meddocan(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 63
class Task_ext_MTSamples_temporal_annotation(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["time expression"]
        self.list_field_feature = ["category"]
        self.list_pattern = [
            r"time expression: (.+), category: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 65
class Task_ext_n2c2_2018_Track2_Adverse_Drug_Events_and_Medication_Extraction(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["drug"]
        self.list_field_feature = [
            "adverse drug event",
            "dosage",
            "duration",
            "form",
            "frequency",
            "reason",
            "route",
            "strength",
        ]
        self.list_pattern = [
            r"drug: (.+), adverse drug event: (.+), dosage: (.+), duration: (.+), form: (.+), frequency: (.+), reason: (.+), route: (.+), strength: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]

    #   Convert original label/pred to its complete format
    def convert(self, line):
        drug_pattern = r"drug:\s*([^;,]+)[;,]"
        adverse_drug_event_pattern = r"adverse drug event:\s*([^;,]+)[;,]"
        dosage_pattern = r"dosage:\s*([^;,]+)[;,]"
        duration_pattern = r"duration:\s*([^;,]+)[;,]"
        form_pattern = r"form:\s*([^;,]+)[;,]"
        frequency_pattern = r"frequency:\s*([^;,]+)[;,]"
        reason_pattern = r"reason:\s*([^;,]+)[;,]"
        route_pattern = r"route:\s*([^;,]+)[;,]"
        strength_pattern = r"strength:\s*([^;,]+)[;,]"

        drug = regex.search(drug_pattern, line)
        adverse_drug_event = regex.search(adverse_drug_event_pattern, line)
        dosage = regex.search(dosage_pattern, line)
        duration = regex.search(duration_pattern, line)
        form = regex.search(form_pattern, line)
        frequency = regex.search(frequency_pattern, line)
        reason = regex.search(reason_pattern, line)
        route = regex.search(route_pattern, line)
        strength = regex.search(strength_pattern, line)

        if drug is None:
            return ""

        result = f"drug: {drug.group(1)}, adverse drug event: {adverse_drug_event.group(1) if adverse_drug_event is not None else 'not mentioned'}, dosage: {dosage.group(1) if dosage is not None else 'not mentioned'}, duration: {duration.group(1) if duration is not None else 'not mentioned'}, form: {form.group(1) if form is not None else 'not mentioned'}, frequency: {frequency.group(1) if frequency is not None else 'not mentioned'}, reason: {reason.group(1) if reason is not None else 'not mentioned'}, route: {route.group(1) if route is not None else 'not mentioned'}, strength: {strength.group(1) if strength is not None else 'not mentioned'};"
        return result

    def get_label(self, list_dict_data, prompt_mode="direct"):
        list_list_dict_entity = []
        for dict_data in list_dict_data:
            list_dict_entity = []
            list_line = process_text_clean(dict_data["output"]).split(self.sep_event)
            list_line = [line.strip() for line in list_line if line.strip() != ""]
            for line_one in list_line:
                line_one = self.convert(line_one)
                if line_one == "":
                    continue
                result = regex.search(self.list_pattern[0], line_one, regex.IGNORECASE)
                # If the entity is found, extract the entity and the features
                # The group(1)-group(num_subject) is the subject, the group(num_subject+1)-group(num_subject+num_feature) is the features
                dict_entity = {}
                for idx, field_subject in enumerate(self.list_field_subject):
                    dict_entity[field_subject] = result.group(idx + 1).strip()
                # Extract the features
                for idx, field_feature in enumerate(self.list_field_feature):
                    dict_entity[field_feature] = result.group(
                        idx + self.num_subject + 1
                    ).strip()
                list_dict_entity.append(dict_entity)
            list_list_dict_entity.append(list_dict_entity)

        return list_list_dict_entity

    def get_pred(self, list_dict_data, prompt_mode="direct"):
        list_list_dict_entity = []
        for idx_data, dict_data in enumerate(list_dict_data):
            list_dict_entity = []
            if dict_data["pred"].strip() != "":
                # Split the response by event
                str_line = process_text_clean(
                    dict_data["pred"], flag_lower=True, flag_punc_to_en=True
                )
                if "cot" in prompt_mode:
                    str_line = extract_cot_pred(str_line)
                list_line = str_line.split(self.sep_event)
                # Filter the invalid entity
                list_line = [line.strip() for line in list_line if line.strip() != ""]
                # Search the entity by regex for each line
                for line_one in list_line:
                    line_one = self.convert(line_one)
                    if line_one == "":
                        continue
                    # Search the entity by regex for each pattern
                    dict_entity = {}
                    for pattern in self.list_pattern:
                        result = regex.search(pattern, line_one, regex.IGNORECASE)
                        if result:
                            # Extract the subject
                            for idx, field_subject in enumerate(
                                self.list_field_subject
                            ):
                                dict_entity[field_subject] = result.group(
                                    idx + 1
                                ).strip()
                            # Extract the features
                            for idx, field_feature in enumerate(
                                self.list_field_feature
                            ):
                                dict_entity[field_feature] = result.group(
                                    idx + self.num_subject + 1
                                ).strip()
                            list_dict_entity.append(dict_entity)
                            # one entity is found, skip the other patterns
                            break
            # If no entity is found, set the entity to -1
            if len(list_dict_entity) > 0:
                list_list_dict_entity.append(list_dict_entity)
            else:
                list_list_dict_entity.append([-1])

        return list_list_dict_entity


# 66
class Task_ext_NorSynthClinical_entity(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_NorSynthClinical_relation(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity_1", "entity_2"]
        self.list_field_feature = ["relation"]
        self.list_pattern = [
            r"entity_1: (.+), entity_2: (.+), relation: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 68
class Task_ext_NUBES(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "scope"]
        self.list_pattern = [
            r"entity: (.+), type: (.+), scope: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 90
class Task_ext_n2c2_2014_Heart_Disease_Challenge_Diabete(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category_1", "category_2"]
        self.list_pattern = [
            r"indicator: (.+), category_1: (.+), category_2: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_n2c2_2014_Heart_Disease_Challenge_CAD(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category_1", "category_2"]
        self.list_pattern = [
            r"indicator: (.+), category_1: (.+), category_2: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_n2c2_2014_Heart_Disease_Challenge_Hyperlipidemia(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category_1", "category_2"]
        self.list_pattern = [
            r"indicator: (.+), category_1: (.+), category_2: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_n2c2_2014_Heart_Disease_Challenge_Hypertension(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category_1", "category_2"]
        self.list_pattern = [
            r"indicator: (.+), category_1: (.+), category_2: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_n2c2_2014_Heart_Disease_Challenge_Obesity(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category_1", "category_2"]
        self.list_pattern = [
            r"indicator: (.+), category_1: (.+), category_2: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_n2c2_2014_Heart_Disease_Challenge_Medication(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["indicator"]
        self.list_field_feature = ["category"]
        self.list_pattern = [
            r"indicator: (.+), category: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 94
class Task_ext_RuDReC(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


# 95
class Task_ext_NorSynthClinical_PHI(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type"]
        self.list_pattern = [
            r"entity: (.+), type: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
        ]


class Task_ext_ADE_ADE_relation(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the main regex pattern for the entity
        self.list_field_subject = ["drug"]
        self.list_field_feature = ["adverse effect"]

        self.list_pattern = [
            # drug: cimetidine, adverse effect: decreased oxygen;
            r"drug: (.+), adverse effect: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]


class Task_ext_ADE_Drug_dosage(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the main regex pattern for the entity
        self.list_field_subject = ["drug"]
        self.list_field_feature = ["dosage"]

        self.list_pattern = [
            # Drug: methotrexate, Dosage: high-dose;
            r"drug: (.+), dosage: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]


class Task_ext_Cantemist_NER(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_MIE(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "status"]

        self.list_pattern = [
            # entity: 胸痛, type: 症状, status: 病人-阳性;
            r"entity: (.+), type: (.+), status: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):(.+),([^;]+)",
        ]


class Task_ext_Ex4CDS(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_IMCS_V2_NER(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_CHIP_MDCFNPC(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["findings"]
        self.list_field_feature = ["status"]
        self.list_pattern = [
            # findings: 吐, status: 阳性;
            r"findings: (.+), status: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]


class Task_ext_IMCS_V2_SR(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["symptom"]
        self.list_field_feature = ["status"]
        self.list_pattern = [
            # symptom: 发热, status: negative;
            r"symptom: (.+), status: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]


class Task_ext_CAS_label(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["issue"]
        self.list_field_feature = ["age", "genre"]
        self.list_pattern = [
            # age: 53 ans, genre: féminin, issue: None
            r"age: (.+), genre: (.+), issue: (.+)",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*(.+)",
            r"(.+),(.+),(.+)",
        ]


class Task_ext_RuCCoN_NER(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_RuCCoN_NER_Nor(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "normalized terms"]
        self.list_pattern = [
            r"entity: (.+), type: (.+), normalized terms: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):(.+),([^;]+)",
        ]


class Task_ext_BRONCO150_NER_Nor(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "normalized terms"]
        self.list_pattern = [
            r"entity: (.+), type: (.+), normalized terms: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):(.+),([^;]+)",
        ]


class Task_ext_BRONCO150_NER_status(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "status"]
        self.list_pattern = [
            r"entity: (.+), type: (.+), status: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):(.+),([^;]+)",
        ]


class Task_ext_CARDIO_DE(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_GraSSCo_PHI(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_IFMIR_NER(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


class Task_ext_IFMIR_NER_factuality(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["type", "intention"]
        self.list_pattern = [
            r"entity: (.+), type: (.+), intention: (.+);",
            r".*:\s*(.+),\s*.*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):(.+),([^;]+)",
        ]


class Task_ext_iCorpus(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)


# normalization-entity, and code -> Task_ext
class Task_ext_Cantemist_Norm(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["entity"]
        self.list_field_feature = ["code"]
        self.list_pattern = [
            # entity: Carcinoma microcítico, code: 8041/3;
            r"entity: (.+), code: (.+);",
            r".*:\s*(.+),\s*.*:\s*([^;]+)",
            r"(.+):([^;]+)",
        ]


class Task_ext_CHIP_CDEE(Task_ext):
    def __init__(self, args, task):
        super().__init__(args, task)

        self.sep_event = "\n"
        # Define the subject and the features of the entity
        self.list_field_subject = ["subject"]
        self.list_field_feature = ["description", "location", "status"]
        self.list_pattern = [
            # Subject: 结节, Description: [], Location: [右肺门], Status: 肯定;
            r"subject: (.+), description: \[(.*)\], location: \[(.*)\], status: (.*);",
            r".*:(.+),\s*.*:\s*\[(.*)\],\s*.*:\s*\[(.*)\],\s*.*:\s*([^;]+)",
            r"(.+):(.+),\[(.*)\],\[(.*)\],([^;]+)",
        ]
