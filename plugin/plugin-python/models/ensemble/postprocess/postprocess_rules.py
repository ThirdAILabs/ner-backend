import re
from typing import List, Tuple

PHONE_REGEX = re.compile(
    r"(?:\+?\d{1,3}[\s.\-]?)?(?:\d{1,3}[\s.\-]?){2,5}\d{1,4}(?:\s*(?:x|ext|extension)\s*\d{1,6})?"
)
CARD_REGEX = re.compile(r"[\d\s\-]{12,19}")
CREDIT_SCORE_REGEX = re.compile(r"\b\d{2,3}\b")


def is_valid_phone(number: str) -> bool:
    """
    Returns True if `number` has at least 7 digits and fully matches PHONE_REGEX.
    """
    digits = re.sub(r"\D", "", number)
    if len(digits) < 7 or len(digits) > 15:
        return False
    return bool(PHONE_REGEX.fullmatch(number))


def is_luhn_valid(number: str) -> bool:
    """
    Standard Luhn check for a string of digits.
    """
    digits = [int(d) for d in number]
    checksum = 0
    parity = len(digits) % 2
    for i, d in enumerate(digits):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


def is_valid_card(number: str) -> bool:
    """
    Returns True if `number` has 12–19 digits and passes Luhn.
    """
    digits = re.sub(r"\D", "", number)
    if not (12 <= len(digits) <= 19):
        return False
    return is_luhn_valid(digits)


def is_valid_credit_score(
    score: str, full_text: str, char_start: int, char_end: int
) -> bool:
    """
    Validates a 2–3 digit credit score if 'credit' and 'score' appear
    within 20 characters before or after the span.
    """
    if not CREDIT_SCORE_REGEX.fullmatch(score):
        return False
    # extract context window
    before = full_text[max(0, char_start - 20) : char_start].lower()
    after = full_text[char_end : char_end + 20].lower()
    context = before + after
    return "credit" in context and "score" in context


def group_consecutive_indices(
    tags: List[str], spans: List[Tuple[int, int]], label: str
) -> List[Tuple[int, int]]:
    """
    Merge consecutive token indices where `tags[i] == label` and spans are contiguous.
    Returns list of (start_idx, end_idx) for each group.
    """
    groups: List[Tuple[int, int]] = []
    i, n = 0, len(tags)
    while i < n:
        if tags[i] == label:
            start = i
            while (
                i + 1 < n
                and tags[i + 1] == label
                and (
                    spans[i + 1][0] == spans[i][1] or spans[i + 1][0] == spans[i][1] + 1
                )
            ):
                i += 1
            groups.append((start, i))
        i += 1
    return groups
