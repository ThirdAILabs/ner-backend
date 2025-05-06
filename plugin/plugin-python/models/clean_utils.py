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


def clean_text(text: str) -> str:
    text = replace_punct_followed_by_space(text)
    text = replace_punct_after_space(text)
    text = text.split()
    text = " ".join(text)
    return text
