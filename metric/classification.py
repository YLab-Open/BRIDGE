import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
)
from scipy.stats import pearsonr, spearmanr
from dataset.config import get_models_evaluate, get_metrics_clf


def calc_metrics_clf(list_pred, list_label, average=None):
    """
    Calculate metrics for classification tasks, including accuracy, f1, precision, recall

    Input:
        list_label: list of integers for ground-truth
        list_pred: list of integers for prediction

    Output:
        dict_metrics: dictionary of metrics, including accuracy, f1, precision, recall
    """
    # get all valid labels
    valid_labels = sorted(set(list_label))

    if average is None:
        list_average = ["macro", "micro"]
    else:
        if isinstance(average, str):
            list_average = [average]

    # calculate metrics
    accuracy = accuracy_score(y_pred=list_pred, y_true=list_label)
    dict_metrics = {"accuracy": accuracy * 100}

    for average in list_average:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_pred=list_pred,
            y_true=list_label,
            average=average,
            labels=valid_labels,
            zero_division=0,
        )
        dict_metrics[f"precision_{average}"] = precision * 100
        dict_metrics[f"recall_{average}"] = recall * 100
        dict_metrics[f"f1_{average}"] = f1 * 100

    return dict_metrics


def convert_to_binary_matrix(list_list, label_to_index):
    binary_matrix = np.zeros((len(list_list), len(label_to_index)), dtype=int)
    for i, labels in enumerate(list_list):
        for label in labels:
            if label in label_to_index:
                binary_matrix[i, label_to_index[label]] = 1
    return binary_matrix


def get_arr_multi_hot(list_list_pred, list_list_label):
    """
    Get multi-hot array from list of labels

    Input:
        list_list_pred: list of list of integers for prediction
        list_list_label: list of list of integers for ground-truth

    Output:
        array_pred: multi-hot array for prediction
        array_label: multi-hot array for ground-truth
    """
    # get all valid labels
    all_labels = sorted(set(label for labels in list_list_label for label in labels))

    # create a mapping from label to index
    label_to_index = {label: i for i, label in enumerate(all_labels)}

    # convert list of list of labels to binary matrix
    array_pred = convert_to_binary_matrix(list_list_pred, label_to_index)
    array_label = convert_to_binary_matrix(list_list_label, label_to_index)

    return array_pred, array_label, label_to_index


def calc_metrics_clf_mul_label(list_list_pred, list_list_label):
    """
    Calculate metrics for multi-label classification tasks, including accuracy, f1, precision, recall

    Input:
        list_list_label: list of list of integers for ground-truth
        list_list_pred: list of list of integers for prediction

    Output:
        dict_metrics: dictionary of metrics, including accuracy, f1, precision, recall
    """
    # get multi-hot array
    array_pred, array_label, label_to_index = get_arr_multi_hot(
        list_list_pred, list_list_label
    )

    # calculate metrics
    dict_metrics = compute_metrics_clf_mul_label(
        array_pred, array_label, label_to_index
    )

    return dict_metrics, array_pred, array_label, label_to_index


def compute_metrics_clf_mul_label(array_pred, array_label, label_to_index):
    """
    Calculate metrics for multi-label classification tasks, including accuracy, f1, precision, recall

    Input:
        array_pred: multi-hot array for prediction
        array_label: multi-hot array for ground-truth

    Output:
        dict_metrics: dictionary of metrics, including accuracy, f1, precision, recall
    """
    # get all valid labels
    valid_labels = list(label_to_index.values())

    # calculate metrics
    accuracy = accuracy_score(y_pred=array_pred, y_true=array_label)
    dict_metrics = {"accuracy": accuracy * 100}
    for average in ["macro", "micro"]:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_pred=array_pred,
            y_true=array_label,
            average=average,
            labels=valid_labels,
            zero_division=0,
        )
        dict_metrics[f"precision_{average}"] = precision * 100
        dict_metrics[f"recall_{average}"] = recall * 100
        dict_metrics[f"f1_{average}"] = f1 * 100

    return dict_metrics


def calc_metrics_clf_ordered(list_pred, list_label):
    """
    Calculate metrics for ordered classification tasks, including accuracy, f1, precision, recall

    Input:
        list_label: list of integers for ground-truth
        list_pred: list of integers for prediction

    Output:
        dict_metrics: dictionary of metrics, including accuracy, f1, precision, recall

    Note:
        The target labels are ordered, e.g., 0 < 1 < 2 < 3
    """

    # Initialize the dictionary to store the performance metrics
    valid_labels = sorted(set(list_label))

    # Calculate the overall performance
    accuracy = accuracy_score(list_label, list_pred)
    dict_metrics = {"accuracy": accuracy * 100}

    # Calculate the correlation between the predicted and true scores
    # if the list_pred is a constant array, then the correlation is not calculated
    if len(set(list_pred)) > 1:
        # Calculate the Pearson correlation
        corr_pearson, _ = pearsonr(list_label, list_pred)
        dict_metrics["pearson"] = corr_pearson
        # Calculate the Spearman correlation
        corr_spearman, p_value = spearmanr(list_label, list_pred)
        dict_metrics["spearman"] = corr_spearman
    else:
        dict_metrics["pearson"] = 0
        dict_metrics["spearman"] = 0

    # Calculat the mae and rmse
    list_diff = [abs(pred - label) for pred, label in zip(list_pred, list_label)]
    dict_metrics["mae"] = np.mean(list_diff)
    dict_metrics["rmse"] = np.sqrt(np.mean([diff**2 for diff in list_diff]))

    # Caluculate the Quadratic Weighted Kappa
    qwk = cohen_kappa_score(list_label, list_pred, weights="quadratic")
    dict_metrics["qwk"] = qwk

    # Calculate the overall precision, recall, and F1 score
    for average in ["macro", "micro"]:
        precision, recall, f1, support = precision_recall_fscore_support(
            list_label,
            list_pred,
            average=average,
            labels=valid_labels,
            zero_division=0,
        )
        dict_metrics[f"precision_{average}"] = precision * 100
        dict_metrics[f"recall_{average}"] = recall * 100
        dict_metrics[f"f1_{average}"] = f1 * 100

    return dict_metrics


def print_metrics_clf(
    dict_model_metrics, list_model=None, list_metrics=None, flag_print_missing=False
):
    """
    Print metrics for classification tasks

    Input:
        dict_model_metrics: dictionary of metrics, including accuracy, f1, precision, recall
    Output:
        string of metrics
    """
    if list_model is None:
        list_model = get_models_evaluate()
    if list_metrics is None:
        list_metrics = get_metrics_clf()

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
        print("---------------------------------------------")
        print(str_missing_output)
        print("---------------------------------------------")

    return str_metric.strip()
