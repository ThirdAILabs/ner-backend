import re

punct_chars = [
    ",",
    ".",
    "!",
    "?",
    ";",
    ":",
    "-",
    "_",
    '"',
    "'",
    "`",
    ")",
    "]",
    "}",
    "(",
    "[",
    "{",
]
escaped = "".join(re.escape(c) for c in punct_chars)

pattern_before_space = rf"(?<=\S)[{escaped}](?=\s+)"
pattern_after_space = rf"(\s+)[{escaped}](?=\S)"

_TOKEN_RE = re.compile(rf"\S+(?:\s+|$)")


def replace_punct_followed_by_space(text: str) -> str:
    original_len = len(text)
    text = re.sub(pattern_before_space, " ", text)
    new_len = len(text)
    assert (
        original_len == new_len
    ), f"Original length: {original_len}, New length: {new_len}"
    return text


def replace_punct_after_space(text: str) -> str:
    original_len = len(text)

    def repl(m: re.Match) -> str:
        return m.group(1) + " "

    text = re.sub(pattern_after_space, repl, text)
    new_len = len(text)
    assert (
        original_len == new_len
    ), f"Original length: {original_len}, New length: {new_len}"
    return text


def clean_text_with_spans(text: str):
    # we do length‐preserving punctuation fixes
    t = replace_punct_followed_by_space(text)
    t = replace_punct_after_space(t)

    #  token+whitespace matches so we capture every run of spaces
    spans = []
    tokens = []
    for m in _TOKEN_RE.finditer(t):
        chunk = m.group(0)
        tok = chunk.strip()
        if not tok:
            continue
        start = m.start()
        end = start + len(tok)
        spans.append((start, end))
        tokens.append(tok)

    cleaned = " ".join(tokens)
    return cleaned, spans


def build_tag_vocab():
    tag_set = set()
    tags = [
        "ADDRESS",
        "CARD_NUMBER",
        "COMPANY",
        "CREDIT_SCORE",
        "DATE",
        "EMAIL",
        "ETHNICITY",
        "GENDER",
        "ID_NUMBER",
        "LICENSE_PLATE",
        "LOCATION",
        "NAME",
        "PHONENUMBER",
        "SERVICE_CODE",
        "SEXUAL_ORIENTATION",
        "SSN",
        "URL",
        "VIN",
        "O",
    ]

    for tag in tags:
        tag_set.add(tag)

    tag2idx = {tag: idx for idx, tag in enumerate(sorted(tag_set))}
    return tag2idx
