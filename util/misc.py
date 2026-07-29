from __future__ import annotations

import random
import sys


def sizeof_fmt(num, suffix="B"):
    """Convert a byte count into a compact human-readable string."""
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def get_size_mem(data):
    """Return the approximate memory size of a Python object."""
    return sizeof_fmt(sys.getsizeof(data))


def many_list_count_sum(list_of_list):
    """Print and return the total number of items across nested lists."""
    sum_length = 0
    for list_one in list_of_list:
        print(f"List one: {len(list_one)}")
        sum_length += len(list_one)
    print(f"All: {sum_length}")
    return sum_length


def get_dict_part(dict_original, size, shuffle=True):
    """Return a size-limited subset of a dictionary."""
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


def set_default_args(
    args,
    attr,
    default,
    valid_set=None,
    cast=lambda x: x,
):
    """Return an argparse attribute value or a validated default."""
    val = getattr(args, attr, None)
    if val is None:
        return default, True
    val = cast(val)
    if valid_set and val not in valid_set:
        raise ValueError(
            f"{attr}={val!r} is invalid; must be one of {sorted(valid_set)}"
        )
    return val, False
