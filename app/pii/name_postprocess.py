"""Clean up raw name spans from a model before the masker uses them (Phase 3B).

Steps, in order, for every raw (start, end) span:
1. Whole-word expansion: a span that starts or ends inside a word grows to the whole word
   ("عما" inside "عمار" -> "عمار"). Masking more of a word is safe; masking half a word leaks the rest.
2. Split and filter: the span is cut at punctuation and at tokens that cannot be part of a name; a token is
   kept only if it is made of letters and has at least 2 of them (digits-only, one-letter or symbol spans go).
3. Exclusion trimming: words from name_exclusions.txt are removed from the EDGES of each piece only
   ("المالك سند" -> "سند", "المالك" alone -> nothing). A name next to an excluded word is never dropped.

Only model spans go through this module. The rule-based masker never reads it.
"""
from __future__ import annotations

import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from app.text.arabic import normalize

EXCLUSIONS_FILE = Path(__file__).resolve().parent / "name_exclusions.txt"
EXCLUSION_CATEGORIES = ("legal_party", "work_role", "family")
MAX_EXCLUSIONS = 40
_CLITICS = ("و", "ف")          # conjunctions that attach to any word
_PREPOSITIONS = ("ب", "ك", "ل")  # prepositions that attach to a word (ل + ال -> لل)


# ------------------------------------------------------------------ exclusion list
def read_exclusion_entries(path: Path | str = EXCLUSIONS_FILE) -> list[tuple[str, str]]:
    """[(word, category)] in file order. Raises ValueError on a malformed line."""
    entries = []
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != 2 or not parts[0] or " " in parts[0]:
            raise ValueError(f"{path}:{n}: expected 'word | category', got {line!r}")
        if parts[1] not in EXCLUSION_CATEGORIES:
            raise ValueError(f"{path}:{n}: unknown category {parts[1]!r}")
        entries.append((parts[0], parts[1]))
    return entries


@lru_cache(maxsize=4)
def load_exclusions(path: str = str(EXCLUSIONS_FILE)) -> frozenset[str]:
    """Normalized forms of the excluded words."""
    return frozenset(normalize(w) for w, _ in read_exclusion_entries(path))


def _forms(word: str) -> set[str]:
    """The word and its forms without attached و/ف and ب/ك/ل (normalized)."""
    w = normalize(word)
    forms = {w}
    for c in _CLITICS:
        if w.startswith(c) and len(w) > 3:
            forms.add(w[1:])
    for f in list(forms):
        if f.startswith("لل") and len(f) > 3:
            forms.add("ال" + f[2:])
        for p in _PREPOSITIONS:
            if f.startswith(p) and len(f) > 3:
                forms.add(f[1:])
    return forms


def is_excluded(word: str, exclusions: frozenset[str] | None = None) -> bool:
    ex = load_exclusions() if exclusions is None else exclusions
    return bool(_forms(word) & ex)


# ------------------------------------------------------------------ words
def is_word_char(ch: str) -> bool:
    """Letters (any script, incl. tatweel) and combining marks (Arabic diacritics). Digits are NOT word chars."""
    return ch.isalpha() or unicodedata.category(ch) == "Mn"


def _letters(s: str) -> int:
    return sum(1 for ch in s if ch.isalpha() and ch != "ـ")


def expand_to_words(text: str, start: int, end: int) -> tuple[int, int]:
    while start > 0 and start < len(text) and is_word_char(text[start - 1]) and is_word_char(text[start]):
        start -= 1
    while 0 < end < len(text) and is_word_char(text[end - 1]) and is_word_char(text[end]):
        end += 1
    return start, end


def word_spans(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """(start, end) of every run of word chars inside text[start:end]."""
    end = len(text) if end is None else end
    out, i = [], start
    while i < end:
        if is_word_char(text[i]):
            j = i
            while j < end and is_word_char(text[j]):
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def is_subword_span(text: str, start: int, end: int) -> bool:
    """True if the span cuts a word: a word char on both sides of its start or of its end."""
    cut_start = 0 < start < len(text) and is_word_char(text[start - 1]) and is_word_char(text[start])
    cut_end = 0 < end < len(text) and is_word_char(text[end - 1]) and is_word_char(text[end])
    return cut_start or cut_end


# ------------------------------------------------------------------ the pipeline
def clean_model_spans(text: str, spans: Iterable[tuple[int, int]],
                      exclusions: frozenset[str] | None = None) -> list[tuple[int, int]]:
    """Raw model spans -> whole-word, valid, edge-trimmed name spans (sorted, merged)."""
    ex = load_exclusions() if exclusions is None else exclusions
    n = len(text)
    pieces: list[tuple[int, int]] = []
    for s, e in spans:
        s, e = max(0, int(s)), min(n, int(e))
        if e <= s:
            continue
        s, e = expand_to_words(text, s, e)
        group: list[tuple[int, int]] = []
        for ws, we in word_spans(text, s, e) + [(-1, -1)]:   # sentinel flushes the last group
            breaks = ws < 0 or (group and text[group[-1][1]:ws].strip() != "") \
                or (group and "\n" in text[group[-1][1]:ws])
            if breaks and group:
                pieces.extend(_trim(text, group, ex))
                group = []
            if ws < 0:
                break
            if _letters(text[ws:we]) >= 2:
                group.append((ws, we))
            elif group:                                        # an invalid token also ends the group
                pieces.extend(_trim(text, group, ex))
                group = []
    merged: list[list[int]] = []
    for s, e in sorted(pieces):
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def _trim(text: str, group: list[tuple[int, int]], ex: frozenset[str]) -> list[tuple[int, int]]:
    i, j = 0, len(group)
    while i < j and is_excluded(text[group[i][0]:group[i][1]], ex):
        i += 1
    while j > i and is_excluded(text[group[j - 1][0]:group[j - 1][1]], ex):
        j -= 1
    return [(group[i][0], group[j - 1][1])] if i < j else []
