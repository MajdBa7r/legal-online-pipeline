"""Arabic text helpers shared by every stage.

`normalize()` must stay identical to the Data team's normalization of
`articles.normalized_text`, otherwise full-text search and glossary matching
drift apart. If they change their rules, change them here too.
The original text is never modified; normalization is only for matching.
"""
from __future__ import annotations

import re

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
_DIACRITICS = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = "ـ"
_WS = re.compile(r"\s+")
_ARABIC_WORD = re.compile(r"[ء-ي]+")

# Very common function words; dropped when building a full-text query.
STOPWORDS = {
    "في", "من", "على", "عن", "الى", "الي", "ان", "اذا", "او", "ثم", "مع", "هل", "ما", "ماذا",
    "كيف", "لماذا", "متى", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "كان", "كانت",
    "يكون", "قد", "لا", "لم", "لن", "انا", "انت", "هو", "هي", "نحن", "هم", "كل", "بعد", "قبل",
    "عند", "لدي", "عندي", "بس", "شو", "اشي", "هاد", "هاي", "يعني", "اللي", "ليش", "وين", "كمان",
    "بدي", "بده", "حدا", "مش", "عشان", "لانه", "لان", "حتى", "اي", "اية", "ايه", "او", "و",
}


def to_latin_digits(text: str) -> str:
    """Convert Arabic-Indic digits to 0-9. Keeps string length (1 char -> 1 char)."""
    return text.translate(_DIGITS)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Matching form: digits unified, diacritics and tatweel removed, letter forms unified."""
    t = to_latin_digits(text)
    t = _DIACRITICS.sub("", t).replace(_TATWEEL, "")
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    return collapse_ws(t)


def arabic_tokens(text: str) -> list[str]:
    return _ARABIC_WORD.findall(normalize(text))


def build_fts_query(text: str, max_terms: int = 12) -> str:
    """OR-joined query for websearch_to_tsquery.

    websearch_to_tsquery ANDs plain words, which kills recall on long dialect
    questions. Joining the content words with ' or ' asks for any of them and
    lets ts_rank do the ordering.
    """
    seen: list[str] = []
    for tok in arabic_tokens(text):
        if len(tok) < 3 or tok in STOPWORDS or tok in seen:
            continue
        seen.append(tok)
        if len(seen) >= max_terms:
            break
    return " or ".join(seen)
