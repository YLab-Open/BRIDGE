from __future__ import annotations


def list_drop_duplicate(list_processed):
    """Remove duplicate list items while preserving first-seen order."""
    list_new = list(set(list_processed))
    list_new.sort(key=list_processed.index)
    return list_new


def list_flatten(nested_list):
    """Flatten one level of list nesting."""
    flat_list = []
    for element in nested_list:
        if type(element) is list:
            for item in element:
                flat_list.append(item)
        else:
            flat_list.append(element)
    return flat_list


def list_clean_blank_short(list_processed, len_min=5):
    """Keep strings with at least len_min whitespace-delimited tokens."""
    return list(
        filter(
            lambda x: isinstance(x, str) and len(x.split()) >= len_min,
            list_processed,
        )
    )


def list_clean_blank(list_processed):
    """Keep non-empty strings."""
    return list(
        filter(lambda x: isinstance(x, str) and len(x.split()) > 0, list_processed)
    )


def str_q2b(text):
    """Convert full-width characters to half-width equivalents."""
    rstring = ""
    for uchar in text:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
