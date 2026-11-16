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
# zero-width spaces/joiners, direction marks and BOM found in scraped text
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿­]")
_WS = re.compile(r"\s+")
_ARABIC_WORD = re.compile(r"[ء-ي]+")
_SEARCH_TOKEN = re.compile(r"[ء-ي]+|\d+")

# Very common function words; dropped when building a full-text query.
STOPWORDS = {
    "في", "من", "على", "عن", "الى", "الي", "ان", "اذا", "او", "ثم", "مع", "هل", "ما", "ماذا",
    "كيف", "لماذا", "متى", "هذا", "هذه", "ذلك", "تلك", "التي", "الذي", "الذين", "كان", "كانت",
    "يكون", "قد", "لا", "لم", "لن", "انا", "انت", "هو", "هي", "نحن", "هم", "كل", "بعد", "قبل",
    "عند", "لدي", "عندي", "بس", "شو", "اشي", "هاد", "هاي", "يعني", "اللي", "ليش", "وين", "كمان",
    "بدي", "بده", "حدا", "مش", "عشان", "لانه", "لان", "حتى", "اي", "اية", "ايه", "و",
    # the same words after normalize() (ة->ه, ى->ي, أ->ا)
    "هذه", "عليه", "عليها", "فيه", "فيها", "منه", "منها", "به", "بها", "له", "لها", "الا", "او",
    "غير", "اذ", "ايضا", "بين", "حيث", "وفق", "وفقا", "اما", "لكن", "ولا", "وان", "واذا",
}

# Light stemmer affixes (in normalize() form). Order matters: longest first.
_PREFIXES = ("وبال", "وكال", "وال", "بال", "كال", "فال", "لل", "ال", "و")
# single-letter prepositions, stripped only from longer words ("باجازه" yes, "باطل" no)
_SHORT_PREFIXES = ("ب", "ل")
_SHORT_PREFIX_MIN_REST = 4
# "مدتها", "اجازته": the ta marbuta becomes ت before a pronoun; map it back to ه
_TA_PRONOUN_SUFFIXES = ("تهما", "تهم", "تها", "ته")
_PRONOUN_SUFFIXES = ("هما", "هم", "ها", "كم", "نا")
# "ا" covers the accusative tanween left after diacritics are removed ("يوماً" -> "يوما")
_SUFFIXES = ("ات", "ون", "ين", "ان", "يه", "ه", "ي", "ا")
_MIN_STEM = 3


def to_latin_digits(text: str) -> str:
    """Convert Arabic-Indic digits to 0-9. Keeps string length (1 char -> 1 char)."""
    return text.translate(_DIGITS)


def collapse_ws(text: str) -> str:
    return _WS.sub(" ", text).strip()


def normalize(text: str) -> str:
    """Matching form: digits unified, diacritics/tatweel/invisible marks removed, letter forms unified."""
    t = to_latin_digits(text)
    t = _INVISIBLE.sub("", t)
    t = _DIACRITICS.sub("", t).replace(_TATWEEL, "")
    t = re.sub("[إأآٱ]", "ا", t)
    t = t.replace("ى", "ي").replace("ة", "ه").replace("ؤ", "و").replace("ئ", "ي")
    t = t.replace("ی", "ي").replace("ک", "ك")  # Persian forms that sometimes appear in pasted text
    return collapse_ws(t)


def clean_display(text: str) -> str:
    """For text shown to users (law titles): remove tatweel and invisible marks, keep everything else."""
    return collapse_ws(_INVISIBLE.sub("", text).replace(_TATWEEL, ""))


STOPWORDS = {normalize(w) for w in STOPWORDS}


def arabic_tokens(text: str) -> list[str]:
    return _ARABIC_WORD.findall(normalize(text))


def light_stem(token: str) -> str:
    """Light Arabic stemmer for retrieval (in the spirit of Larkey's light10).

    Removes one leading clitic/article and up to one pronoun suffix plus one
    plural/feminine suffix, never leaving fewer than 3 letters. It is
    deliberately conservative: broken plurals and roots are not handled.
    Expects a normalize()d token.
    """
    t = token
    for p in _PREFIXES:
        if t.startswith(p) and len(t) - len(p) >= _MIN_STEM:
            t = t[len(p):]
            break
    else:
        for p in _SHORT_PREFIXES:
            if t.startswith(p) and len(t) - len(p) >= _SHORT_PREFIX_MIN_REST:
                t = t[len(p):]
                break
    for s in _TA_PRONOUN_SUFFIXES:
        if t.endswith(s) and len(t) - len(s) + 1 >= _MIN_STEM:
            t = t[: -len(s)] + "ه"
            break
    else:
        for s in _PRONOUN_SUFFIXES:
            if t.endswith(s) and len(t) - len(s) >= _MIN_STEM:
                t = t[: -len(s)]
                break
    for s in _SUFFIXES:
        if t.endswith(s) and len(t) - len(s) >= _MIN_STEM:
            t = t[: -len(s)]
            break
    return t


def search_tokens(text: str, *, stem: bool = True, keep_numbers: bool = True) -> list[str]:
    """Tokens used by the local keyword index: normalized, stopwords removed, light-stemmed.

    Numbers are kept (article numbers, amounts, durations matter in legal questions).
    """
    out: list[str] = []
    for tok in _SEARCH_TOKEN.findall(normalize(text)):
        if tok.isdigit():
            if keep_numbers:
                out.append(tok)
            continue
        if tok in STOPWORDS or len(tok) < 2:
            continue
        out.append(light_stem(tok) if stem else tok)
    return out


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
