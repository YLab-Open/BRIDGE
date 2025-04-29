from transformers import BasicTokenizer
from dataset.config import (
    get_models_evaluate,
    get_metrics_ext,
    get_metrics_ext_qa,
)


def ext_prepare_subject_event_sets(list_dict, subject_fields):
    """
    Extract both subject sets and event sets from a given list of dicts.
    Subject set is represented by sorted subject fields tuple.
    Event set is represented by sorted dictionary items including sorted subject fields.

    Parameters:
    list_dict (list of dict): Input dict list for one sample
    subject_fields (list): List of fields considered as subject identifiers

    Returns:
    subject_set (set of tuples): Each tuple is a sorted subject representation
    event_set (set of tuples): Each tuple is the sorted representation of all fields including subject
    """
    subject_set = set()
    event_set = set()

    for item in list_dict:
        # Check if all subject fields are present in the item
        if all(field in item for field in subject_fields):
            # Extract subject values and sort them
            subject_values = [item[field] for field in subject_fields]
            subject_values_sorted = tuple(sorted(subject_values))

            # Add the sorted subject tuple to subject_set
            subject_set.add(subject_values_sorted)

            # Create an item copy with sorted subject fields
            item_copy = item.copy()
            for idx, field in enumerate(subject_fields):
                item_copy[field] = subject_values_sorted[idx]

            # Convert the full item dict to a tuple of sorted items for event comparison
            event_tuple = tuple(sorted(item_copy.items()))
            event_set.add(event_tuple)

    return subject_set, event_set


def ext_compute_sample_metrics(list_pred, list_label, list_field_subject=["entity"]):
    """
    Compute per-sample TP, FP, FN for both subject recognition and event extraction.
    Also record whether the subject and event are fully correctly recognized at the sample level.

    Parameters:
    list_pred (list of list of dict): Predicted results for each sample.
    list_label (list of list of dict): True results for each sample.
    list_field_subject (list of str): Subject field names, e.g., ['entity'] or ['subject_1','subject_2'].

    Returns:
    dict_metrics_sample (dict): A dictionary with lists of per-sample TP, FP, FN, and correctness flags.
    """

    # Initialize counters for subject recognition and event extraction
    dict_metrics_sample = {
        "subject_tp": [],
        "subject_fp": [],
        "subject_fn": [],
        "event_tp": [],
        "event_fp": [],
        "event_fn": [],
        "subject_correct_samples": [],
        "event_correct_samples": [],
    }

    for list_dict_pred, list_dict_label in zip(list_pred, list_label):
        # Prepare sets for subject recognition
        pred_subject_set, pred_event_set = ext_prepare_subject_event_sets(
            list_dict_pred, list_field_subject
        )
        label_subject_set, label_event_set = ext_prepare_subject_event_sets(
            list_dict_label, list_field_subject
        )

        # Calculate Subject-Level metrics
        subject_tp = len(pred_subject_set & label_subject_set)
        subject_fp = len(pred_subject_set - label_subject_set)
        subject_fn = len(label_subject_set - pred_subject_set)

        # Check subject correctness for this sample
        subject_correct = 1 if pred_subject_set == label_subject_set else 0

        # Calculate Event-Level metrics
        event_tp = len(pred_event_set & label_event_set)
        event_fp = len(pred_event_set - label_event_set)
        event_fn = len(label_event_set - pred_event_set)

        # Check event correctness for this sample
        event_correct = 1 if pred_event_set == label_event_set else 0

        # Record metrics per sample
        dict_metrics_sample["subject_tp"].append(subject_tp)
        dict_metrics_sample["subject_fp"].append(subject_fp)
        dict_metrics_sample["subject_fn"].append(subject_fn)
        dict_metrics_sample["event_tp"].append(event_tp)
        dict_metrics_sample["event_fp"].append(event_fp)
        dict_metrics_sample["event_fn"].append(event_fn)
        dict_metrics_sample["subject_correct_samples"].append(subject_correct)
        dict_metrics_sample["event_correct_samples"].append(event_correct)

    return dict_metrics_sample


def ext_compute_overall_metrics(dict_metrics_sample):
    """
    Compute overall accuracy, precision, recall, and F1 for subjects and events.

    Parameters:
    dict_sums (dict): A dictionary containing sums of TP, FP, FN, and correct samples:
        {
            "subject_tp": list of int,
            "subject_fp": list of int,
            ...
        }

    Returns:
    dict_performance (dict):
        {
            "accuracy_subject": float,
            "precision_subject": float,
            ...
        }
    All values are in percentage form.
    """
    # Get the number of samples
    num_samples = len(dict_metrics_sample["subject_tp"])

    # Convert lists to sums for final metric calculation
    # For subject and event
    dict_sums = {}
    for col in ["subject", "event"]:
        dict_sums[f"{col}_tp"] = sum(dict_metrics_sample[f"{col}_tp"])
        dict_sums[f"{col}_fp"] = sum(dict_metrics_sample[f"{col}_fp"])
        dict_sums[f"{col}_fn"] = sum(dict_metrics_sample[f"{col}_fn"])
        dict_sums[f"{col}_correct_samples"] = sum(
            dict_metrics_sample[f"{col}_correct_samples"]
        )

    dict_performance = {}
    # Iterate over both "subject" and "event" to compute metrics
    for col in ["subject", "event"]:
        # Accuracy: perfect samples / total samples
        accuracy = (
            dict_sums[f"{col}_correct_samples"] / num_samples if num_samples > 0 else 0
        )

        # Get TP, FP, FN counts
        tp = dict_sums[f"{col}_tp"]
        fp = dict_sums[f"{col}_fp"]
        fn = dict_sums[f"{col}_fn"]

        # Precision: TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0

        # Recall: TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

        # F1: 2 * (Precision * Recall) / (Precision + Recall)
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision + recall) > 0
            else 0
        )

        dict_performance[f"accuracy_{col}"] = accuracy
        dict_performance[f"precision_{col}"] = precision
        dict_performance[f"recall_{col}"] = recall
        dict_performance[f"f1_{col}"] = f1

    # Convert all metrics to percentage
    for metric, score in dict_performance.items():
        dict_performance[metric] = score * 100

    return dict_performance


def calc_metrics_ext(list_pred, list_label, list_field_subject=["entity"]):
    """
    Evaluate the performance of subject recognition when all entities in list_field_subject are correctly recognized,
    and the performance of recognizing the event and all attributes correctly.
    Example:
        entity: covid, type: symptom, status: positive

    Parameters:
    list_pred (list of list of dict): Predicted results for each sample.
    list_label (list of list of dict): True results for each sample.
    list_field_subject (list of str): List of subject field names, e.g., ['subject_1', 'subject_2', 'subject_3'].

    Returns:
    dict: A dictionary containing accuracy, precision, recall, and F1 score for both subject recognition and event extraction.
    """
    # Ensure list_field_subject is a list
    if not isinstance(list_field_subject, list):
        list_field_subject = [list_field_subject]

    # Get per-sample metrics
    dict_metrics_sample = ext_compute_sample_metrics(
        list_pred, list_label, list_field_subject
    )

    # Compute metrics
    dict_performance = ext_compute_overall_metrics(dict_metrics_sample)

    return dict_performance, dict_metrics_sample


def calc_metrics_ext_qa(list_pred, list_label, language="en"):
    """
    Evaluate the performance of QA tasks, which extract the relevant information from the orginal text.

    Parameters:
    list_pred (list of list of string prediction): Predicted results for each sample.
    list_label (list of list of string answer): True results for each sample.

    Returns:
    dict: A dictionary containing:
        - overlap_match: the percentage of the overlap between the prediction and the answer
        - exact_match: the percentage of the answer is exatly contained in the prediction

    """
    # Initialize counters for metrics
    overlap_match = 0
    exact_match = 0
    num_sample = len(list_label)

    # Initialize tokenizer
    if language == "jp":
        tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=False)
    else:
        tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=True)

    # Loop over each sample
    for preds, labels in zip(list_pred, list_label):
        for pred, label in zip(preds, labels):
            # Update counts for overlap
            if pred and label:
                pred_set = set(tokenizer.tokenize(pred))
                label_set = set(tokenizer.tokenize(label))
                overlap = pred_set & label_set
                overlap_match += len(overlap) / len(label_set)

                # Update counts for exact match
                if label == pred or label in pred:
                    exact_match += 1

    # Compute metrics
    overlap_accuracy = overlap_match / num_sample if num_sample > 0 else 0
    exact_accuracy = exact_match / num_sample if num_sample > 0 else 0

    # Return results
    dict_metrics = {
        "overlap_match": overlap_accuracy,
        "exact_match": exact_accuracy,
    }

    # round the metrics
    for key in dict_metrics.keys():
        dict_metrics[key] = dict_metrics[key] * 100

    return dict_metrics


def print_metrics_ext(
    dict_model_metrics, list_model=None, list_metrics=None, flag_print_missing=False
):
    """
    Print metrics for extraction tasks

    Input:
        dict_model_metrics: dictionary of metrics, including accuracy, precision, recall, and F1 score
    Output:
        string of metrics
    """
    if list_model is None:
        list_model = get_models_evaluate()
    if list_metrics is None:
        list_metrics = get_metrics_ext()

    str_metric = ""
    str_missing_output = "Mising model:\n\t"
    for model in list_model:
        if model in dict_model_metrics:
            for metric in list_metrics:
                # mean±std
                # str_metric += f"{dict_model_metrics[model][metric]['mean']:.2f}±{dict_model_metrics[model][metric]['std']:.2f} "

                # mean [ci_lower, ci_upper]
                mean_value = dict_model_metrics[model][metric]["mean"]
                ci_lower = dict_model_metrics[model][metric]["ci"][0]
                ci_upper = dict_model_metrics[model][metric]["ci"][1]
                str_metric += f"{mean_value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]; "
        else:
            str_metric += "N/A; " * len(list_metrics)
            str_missing_output += f"{model} "

    if flag_print_missing:
        print("=========================================")
        print(str_missing_output)
        print("=========================================")

    return str_metric.strip()


def print_metrics_ext_qa(
    dict_model_metrics, list_model=None, list_metrics=None, flag_print_missing=False
):
    """
    Print metrics for QA tasks

    Input:
        dict_model_metrics: dictionary of metrics, including overlap_match and exact_match
    Output:
        string of metrics
    """
    if list_model is None:
        list_model = get_models_evaluate()
    if list_metrics is None:
        list_metrics = get_metrics_ext_qa()

    str_metric = ""
    str_missing_output = "Mising model:\n\t"
    for model in list_model:
        if model in dict_model_metrics:
            for metric in list_metrics:
                # mean±std
                # str_metric += f"{dict_model_metrics[model][metric]['mean']:.2f}±{dict_model_metrics[model][metric]['std']:.2f} "

                # mean [ci_lower, ci_upper]
                mean_value = dict_model_metrics[model][metric]["mean"]
                ci_lower = dict_model_metrics[model][metric]["ci"][0]
                ci_upper = dict_model_metrics[model][metric]["ci"][1]
                str_metric += f"{mean_value:.2f} [{ci_lower:.2f}, {ci_upper:.2f}]; "
        else:
            str_metric += "N/A; " * len(list_metrics)
            str_missing_output += f"{model} "

    if flag_print_missing:
        print("=========================================")
        print(str_missing_output)
        print("=========================================")

    return str_metric.strip()
