import os
import sys
import numpy as np
import json
import time
import logging
import random
from scipy import stats


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def get_size_mem(data):
    return sizeof_fmt(sys.getsizeof(data))


def get_keywords(path_keyword):
    """
    get_keywords(path_keyword)
    note:
        prefix_suffix = '( |^|$)'
        line startswith '#': line.strip().split('#')[-1] + prefix_suffix
    """
    list_keywords = []
    prefix_suffix = "( |^|$)"
    with open(path_keyword) as f:
        for line in f.readlines():
            if len(line.strip()) == 0:
                continue
            if line.startswith("#"):
                line = line.strip().split("#")[-1] + prefix_suffix
                print(line)
                list_keywords.append(line)
            else:
                list_keywords.append(line.strip())
    print(f"Got {len(list_keywords)} keywords")
    return list_keywords


def list_to_txt(list_saved, path_saved):
    """
    list_to_txt(list_saved, path_saved)
    """
    with open(path_saved, "w") as f:
        for idx, item in enumerate(list_saved):
            if idx != (len(list_saved) - 1):
                f.write(str(item) + "\n")
            else:
                f.write(str(item))
    print(f"Save to {path_saved}")


def list_to_txt_nested(list_nested, path_saved, flag_print=True):
    """
    list_to_txt(list_saved, path_saved)
    """
    with open(path_saved, "w") as f:
        for idx, list_flat in enumerate(list_nested):
            for item in list_flat:
                f.write(str(item) + "\t")
            if idx != (len(list_nested) - 1):
                f.write("\n")
    if flag_print:
        print(f"Save to {path_saved}")


def txt_to_list(path_saved, flag_print=True):
    """
    txt_to_list(path_saved)
    """
    list_loaded = []
    with open(path_saved, "r") as f:
        for line in f.readlines():
            if len(line.strip()):
                list_loaded.append(line.strip())
    if flag_print:
        print(f"Loaded {path_saved}")
    return list_loaded


def txt_to_list_nested(path_saved, token_split=" "):
    """
    txt_to_list(path_saved)
    """
    with open(path_saved, "r") as f:
        list_loaded = []
        for line in f.readlines():
            line = line.strip()
            if len(line):
                list_line = []
                for item in line.split(token_split):
                    list_line.append(item)
            list_loaded.append(list_line)
    print(f"Loaded {path_saved}")
    return list_loaded


def list_drop_duplicate(list_processed):
    list_new = list(set(list_processed))
    list_new.sort(key=list_processed.index)
    return list_new


def many_list_count_sum(list_of_list):
    """
    many_list_count_sum(list_of_list)
    """
    sum_length = 0
    for list_one in list_of_list:
        print(f"List one: {len(list_one)}")
        sum_length += len(list_one)
    print(f"All: {sum_length}")
    return sum_length


def dict_to_json(dict_saved, path_saved):
    """
    dict_to_json(dict_saved, path_saved)
    note:
        indent=2, ensure_ascii=False
    """
    with open(path_saved, "w", encoding="utf-8") as f:
        f.write(json.dumps(dict_saved, indent=2, ensure_ascii=False))
    print(f"Save to {path_saved}")


def json_to_dict(path_saved):
    """
    json_to_dict(path_saved)
    note:
        indent=2, ensure_ascii=False
    """
    dict_saved = json.loads(open(path_saved, "r").read())
    print(f"Loaded {path_saved}")
    return dict_saved


def get_dict_part(dict_original, size, shuffle=True):
    """
    get_dict_part(dict_original, size, shuffle=True)
    """
    size = int(size)
    if size > len(dict_original):
        raise ValueError(
            f"The size {size} bigger than dict length {len(dict_original)}"
        )
    if shuffle:
        list_part = random.sample(list(dict_original.keys()), size)
    else:
        list_part = list(dict_original.keys())[:size]
    dict_temp = {}
    for key in list_part:
        dict_temp[key] = dict_original[key]
    return dict_temp


def list_flatten(nested_list):
    flat_list = []
    # Iterate through the outer list
    for element in nested_list:
        if type(element) is list:
            # If the element is of type list, iterate through the sublist
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def list_clean_blank_short(list_processed, len_min=5):
    """
    remove blank str in list
    """
    return list(
        filter(
            lambda x: isinstance(x, str) and len(x.split()) >= len_min, list_processed
        )
    )


def list_clean_blank(list_processed):
    """
    remove blank str in list
    """
    return list(
        filter(lambda x: isinstance(x, str) and len(x.split()) > 0, list_processed)
    )


def load_json(input_file):
    with open(input_file, "r") as f:
        samples = json.load(f)
    return samples


def load_dict(dict_path):
    """load_dict"""
    vocab = {}
    for line in open(dict_path, "r", encoding="utf-8"):
        key, value = line.strip("\n").split("\t")
        vocab[int(key)] = value
    return vocab


def write_dict(dict_path, dict_data):
    with open(dict_path, "w", encoding="utf-8") as f:
        for key, value in dict_data.items():
            f.writelines("{}\t{}\n".format(key, value))


def str_q2b(text):
    ustring = text
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    if os.path.exists(log_file):
        os.remove(log_file)
    log_format = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]
    if log_file and log_file != "":
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(log_file_level)
        logger.addHandler(file_handler)
    return logger


class ProgressBar(object):
    """
    custom progress bar
    Example:
        >>> pbar = ProgressBar(n_total=30,desc='Training')
        >>> step = 2
        >>> pbar(step=step)
    """

    def __init__(self, n_total, width=30, desc="Training"):
        self.width = width
        self.n_total = n_total
        self.start_time = time.time()
        self.desc = desc

    def __call__(self, step, info={}):
        now = time.time()
        current = step + 1
        recv_per = current / self.n_total
        bar = f"[{self.desc}] {current}/{self.n_total} ["
        if recv_per >= 1:
            recv_per = 1
        prog_width = int(self.width * recv_per)
        if prog_width > 0:
            bar += "=" * (prog_width - 1)
            if current < self.n_total:
                bar += ">"
            else:
                bar += "="
        bar += "." * (self.width - prog_width)
        bar += "]"
        show_bar = f"\r{bar}"
        time_per_unit = (now - self.start_time) / current
        if current < self.n_total:
            eta = time_per_unit * (self.n_total - current)
            if eta > 3600:
                eta_format = "%d:%02d:%02d" % (
                    eta // 3600,
                    (eta % 3600) // 60,
                    eta % 60,
                )
            elif eta > 60:
                eta_format = "%d:%02d" % (eta // 60, eta % 60)
            else:
                eta_format = "%ds" % eta
            time_info = f" - ETA: {eta_format}"
        else:
            if time_per_unit >= 1:
                time_info = f" {time_per_unit:.1f}s/step"
            elif time_per_unit >= 1e-3:
                time_info = f" {time_per_unit * 1e3:.1f}ms/step"
            else:
                time_info = f" {time_per_unit * 1e6:.1f}us/step"

        show_bar += time_info
        if len(info) != 0:
            show_info = f"{show_bar} " + "-".join(
                [f" {key}: {value:.4f} " for key, value in info.items()]
            )
            print(show_info, end="")
        else:
            print(show_bar, end="")


def get_mean_std_ci(data, confidence=0.95):
    """
    Calculate the mean and confidence interval of the given data.
    """
    data = np.array(data)
    n = len(data)
    # Calculate mean
    mean = np.mean(data)
    # Calculate standard deviation
    std = np.std(data, ddof=1)
    # Calculate standard error of the mean
    sem = stats.sem(data)
    # Get the t-score for the confidence level
    h = sem * stats.t.ppf((1 + confidence) / 2.0, n - 1)

    # Manual calculation
    # sem = std / np.sqrt(n)
    # h = 1.96 * sem

    return mean, std, [mean - h, mean + h]
