import regex


def process_punc_to_en(text):
    """
    Converts common punctuation into English version.
    Input:
        text: str
    Output:
        text: str
    """
    # define the mapping
    FULL_ANGLE_ALPHABET = r"""　“‘”’，。：；＂＃＄％＆＇＊＋－．／０１２３４５６７８９＜＝＞＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝（）～"""
    HALF_ANGLE_ALPHABET = r""" "'"',.:;"#$%&'*+-./0123456789<=>@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\]^_`abcdefghijklmnopqrstuvwxyz{|}()~"""

    translation_table = str.maketrans(FULL_ANGLE_ALPHABET, HALF_ANGLE_ALPHABET)

    return text.translate(translation_table)


def preprocess_ehr_note(text):
    """
    Preprocess electronic healthcare record (EHR) text by removing unnecessary spaces and artifacts,
    while preserving the structure and formatting to make it more readable and suitable for input to large language models.

    Args:
        text (str): The raw EHR text to be processed.

    Returns:
        str: The cleaned and formatted EHR text.
    """
    # # Remove content between ***
    text = regex.sub(r"\*\*\*.*?\*\*\*", "", text)
    # # Remove specific artifacts like '_x000C_'
    text = text.replace("_x000C_", " ")
    # # Replace multiple newlines with two newlines to separate paragraphs
    text = regex.sub(r"\n{3,}", "\n\n", text)
    # # Remove leading and trailing spaces on each line
    text = "\n".join(line.strip() for line in text.split("\n"))
    # # Remove any leftover multiple spaces again
    text = regex.sub(r"[ ]{2,}", " ", text)
    # # Remove duplicate lines
    list_line = text.split("\n")
    seen = set()
    cleaned_lines = []
    for idx_line, line in enumerate(list_line):
        # if line not in seen:
        if line != list_line[idx_line - 1] and line != list_line[idx_line - 2]:
            cleaned_lines.append(line)
            seen.add(line)
    text = "\n".join(cleaned_lines)

    # Remove empty lines
    # text = '\n'.join(line for line in text.split('\n') if line.strip() != '')

    return text.strip()
