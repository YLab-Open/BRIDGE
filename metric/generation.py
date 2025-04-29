import sys
import evaluate
import bert_score
import numpy as np
from rouge_score import rouge_scorer
from transformers import BasicTokenizer
from nltk.data import load as nltk_load
from nltk.tokenize import word_tokenize
from nltk.tokenize.destructive import NLTKWordTokenizer
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction
from nltk.translate.meteor_score import meteor_score

from dataset.config import get_models_evaluate, get_metrics_gen

dict_lang_language = {
    "en": "english",
    "zh": "chinese",
    "es": "spanish",
    "jp": "japanese",
    "fr": "french",
    "de": "german",
    "ru": "russian",
    "pt": "portuguese",
    "no": "norwegian",
}


class Tokenizer_gen:
    def __init__(self, lang="en"):
        self.language = dict_lang_language[lang]
        self.tokenizer_lang = nltk_load(f"tokenizers/punkt/{self.language}.pickle")
        self.tokenizer_nltk = NLTKWordTokenizer()

    def tokenize(self, text):
        token_temp = self.tokenizer_lang.tokenize(text)
        return [
            token for sent in token_temp for token in self.tokenizer_nltk.tokenize(sent)
        ]


class Tokenizer_gen_zh_jp:
    def __init__(self, lang="zh"):
        if lang != "zh" and lang != "jp":
            raise ValueError("Only support zh and jp")

    def tokenize(self, text):
        # Option for jp: janome
        # from janome.tokenizer import Tokenizer
        # return [token.surface for token in t.tokenize(text)]

        # Option for zh: jieba
        # import jieba
        # return jieba.lcut(text)

        # Simple whitespace tokenizer
        text = text.replace(" ", "")
        return list(text)


def calc_metrics_gen(list_pred, list_label, lang="en"):
    """
    Compute BLEU, ROUGE, METEOR, and BERTScore for NLG tasks.
    Supports single or multiple references for each prediction.

    Parameters:
        list_pred (list of str): System predictions
        list_label (list): Ground-truth references. Each element can be:
                           - A single string (one reference)
                           - A list of strings ([ref1, ref2, ...]) (multiple references)
        lang (str): Language code for BERTScore, default='en'.

    Returns:
        dict_metrics_avg (dict): Average metrics over all samples.
        dict_metrics_sample (dict): Per-sample metrics.
    """

    assert len(list_label) == len(
        list_pred
    ), "The length of list_label and list_pred should be the same."

    dict_metrics_sample = {"bleu": [], "rouge": [], "meteor": [], "bertscore": []}

    # Tokenizer
    if lang == "zh" or lang == "jp":
        tokenizer = Tokenizer_gen_zh_jp(lang)
    else:
        tokenizer = Tokenizer_gen(lang)

    # For rogue
    if lang == "en":
        rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"]
            # , use_stemmer=True
        )
    else:
        rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            tokenizer=tokenizer,
            # use_stemmer=True,
        )

    # Calculate BLEU, ROUGE, METEOR per sample
    non_empty_indices = []
    for pred_idx, (pred, label) in enumerate(zip(list_pred, list_label)):
        # Ensure label is a list of references
        if isinstance(label, str):
            references = [label]
        else:
            # label is already a list of references
            references = label

        pred_token = tokenizer.tokenize(pred)
        references_tokenized = [tokenizer.tokenize(ref) for ref in references]

        if pred.strip() == "":
            # Handle empty predictions
            dict_metrics_sample["bleu"].append(0.0)
            dict_metrics_sample["meteor"].append(0.0)
            dict_metrics_sample["rouge"].append(0.0)
        else:
            # Only calculate metrics for non-empty predictions
            non_empty_indices.append(pred_idx)

            # sentence_bleu can take multiple references
            dict_metrics_sample["bleu"].append(
                sentence_bleu(
                    references=references_tokenized,
                    hypothesis=pred_token,
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=SmoothingFunction().method7,
                )
            )

            # meteor_score also supports multiple references by providing them as a list
            dict_metrics_sample["meteor"].append(
                0.0
                # meteor_score(references=references_tokenized, hypothesis=pred_token)
            )

            # For ROUGE-L, calculate ROUGE with each reference and take the average
            rouge_score = rouge.score_multi(targets=references, prediction=pred)
            rouge_score_avg = (
                rouge_score["rouge1"].fmeasure
                + rouge_score["rouge2"].fmeasure
                + rouge_score["rougeL"].fmeasure
            ) / 3
            dict_metrics_sample["rouge"].append(rouge_score_avg)

    if len(non_empty_indices) == 0:
        dict_metrics_sample["bertscore"] = [0.0] * len(list_pred)
    else:
        # Calculate BERTScore for non-empty predictions
        list_pred_non_empty = [list_pred[i] for i in non_empty_indices]
        list_label_non_empty = [list_label[i] for i in non_empty_indices]

        # Calculate BERTScore
        bert_score_p, bert_score_r, bert_score_f1 = bert_score.score(
            cands=list_pred_non_empty,
            refs=list_label_non_empty,
            lang=lang,
            device="cuda",
            batch_size=1024,
            nthreads=32,
            use_fast_tokenizer=True,
            verbose=False,
        )

        # Fill in the BERTScore for all samples
        list_bertscore = [0.0] * len(list_pred)
        for idx, val in zip(non_empty_indices, bert_score_f1.tolist()):
            list_bertscore[idx] = val
        dict_metrics_sample["bertscore"] = list_bertscore

    # Convert all metrics to percentage
    for key, value in dict_metrics_sample.items():
        dict_metrics_sample[key] = [v * 100 for v in value]

    # Get the average of metrics
    dict_performance_avg = {}
    for key in dict_metrics_sample.keys():
        dict_performance_avg[key] = np.mean(dict_metrics_sample[key])

    return dict_performance_avg, dict_metrics_sample


def calc_metrics_gen_hf(list_pred, list_label, lang="en"):
    """
    Input:
        list_pred: list of string for prediction
        list_label: list of string for ground-truth
        lang: str, default='en', options=['en', 'zh', 'tr', 'sp' ...]
        (actually, only 'en', 'zh', 'tr' will be different, others: "bert-base-multilingual-cased")
    Output:
        dict_metrics: dictionary of metrics, including:
            - BLEU: BLEU-4
            - ROUGE: ROUGE-L
            - METEOR
            - BERTScore
    """

    assert len(list_label) == len(
        list_pred
    ), "The length of list_label and list_pred should be the same."

    if lang == "jp":
        tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=False)
    else:
        tokenizer = BasicTokenizer(do_lower_case=True, tokenize_chinese_chars=True)

    dict_metrics = {"bleu": 0, "rouge": 0, "meteor": 0, "bertscore": 0}

    metric_bleu = evaluate.load("bleu")
    metric_rouge = evaluate.load("rouge")
    metric_meteor = evaluate.load("meteor")
    metric_bertscore = evaluate.load("bertscore")

    # Calculate BLEU
    dict_metrics["bleu"] = metric_bleu.compute(
        predictions=list_pred,
        references=list_label,
        tokenizer=tokenizer.tokenize,
        smooth=SmoothingFunction().method7,
    )["bleu"]

    # Calculate ROUGE
    dict_metrics["rouge"] = metric_rouge.compute(
        predictions=list_pred,
        references=list_label,
        use_aggregator=True,
        tokenizer=tokenizer.tokenize,
    )["rougeL"]

    # Calculate METEOR
    dict_metrics["meteor"] = metric_meteor.compute(
        predictions=list_pred, references=list_label
    )["meteor"]

    # Calculate BERTScore
    dict_metrics["bertscore"] = np.mean(
        metric_bertscore.compute(
            predictions=list_pred, references=list_label, lang=lang
        )["f1"]
    )

    # round the metrics
    for key in dict_metrics.keys():
        dict_metrics[key] = dict_metrics[key] * 100

    return dict_metrics


def print_metrics_gen(
    dict_model_metrics, list_model=None, list_metrics=None, flag_print_missing=False
):
    """
    Print metrics for NLG tasks

    Input:
        dict_model_metrics: dictionary of metrics, including BLEU, ROUGE, METEOR, and BERTScore
    Output:
        string of metrics
    """
    if list_model is None:
        list_model = get_models_evaluate()
    if list_metrics is None:
        list_metrics = get_metrics_gen()

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
