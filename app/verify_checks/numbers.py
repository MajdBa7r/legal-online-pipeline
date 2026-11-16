"""Extract numbers from Arabic legal text, written as digits or as words.

Law texts often write numbers in words ("ثلاثين يوماً", "خمسة وعشرين بالمئة") while
the generated claim may use digits, or the reverse. Both sides go through this
parser so the check compares values, not spellings.
"""
from __future__ import annotations

import re

from app.text.arabic import normalize

# all keys are in normalize() form (ة->ه, أ/إ/آ->ا, ى->ي)
UNITS = {
    "صفر": 0, "واحد": 1, "واحده": 1, "احد": 1, "اثنان": 2, "اثنين": 2, "اثنتان": 2, "اثنتين": 2, "اثنا": 2, "اثني": 2,
    "ثلاث": 3, "ثلاثه": 3, "اربع": 4, "اربعه": 4, "خمس": 5, "خمسه": 5, "ست": 6, "سته": 6,
    "سبع": 7, "سبعه": 7, "ثمان": 8, "ثماني": 8, "ثمانيه": 8, "تسع": 9, "تسعه": 9,
}
TEN = {"عشر": 10, "عشره": 10}
TENS = {
    "عشرون": 20, "عشرين": 20, "ثلاثون": 30, "ثلاثين": 30, "اربعون": 40, "اربعين": 40,
    "خمسون": 50, "خمسين": 50, "ستون": 60, "ستين": 60, "سبعون": 70, "سبعين": 70,
    "ثمانون": 80, "ثمانين": 80, "تسعون": 90, "تسعين": 90,
}
HUNDREDS = {
    "مائه": 100, "مئه": 100, "مائتان": 200, "مائتين": 200, "مئتان": 200, "مئتين": 200, "مائتي": 200, "مئتي": 200,
    "ثلاثمائه": 300, "ثلاثمئه": 300, "اربعمائه": 400, "اربعمئه": 400, "خمسمائه": 500, "خمسمئه": 500,
    "ستمائه": 600, "ستمئه": 600, "سبعمائه": 700, "سبعمئه": 700, "ثمانمائه": 800, "ثمانمئه": 800,
    "تسعمائه": 900, "تسعمئه": 900,
}
THOUSANDS = {"الف": 1000, "الفا": 1000, "الفان": 2000, "الفين": 2000, "الفي": 2000, "الاف": 1000}
# duals of time units imply 2
DUALS = {"يومين": 2, "يومان": 2, "شهرين": 2, "شهران": 2, "سنتين": 2, "سنتان": 2, "عامين": 2, "عامان": 2,
         "اسبوعين": 2, "اسبوعان": 2, "ساعتين": 2, "ساعتان": 2}
FRACTIONS = {"نصف": 0.5, "ربع": 0.25, "ثلث": 1 / 3, "ثلثي": 2 / 3, "ثلثا": 2 / 3}

# normalize() also maps ئ -> ي, so run every key through it once
UNITS, TEN, TENS, HUNDREDS, THOUSANDS, DUALS, FRACTIONS = (
    {normalize(k): v for k, v in t.items()} for t in (UNITS, TEN, TENS, HUNDREDS, THOUSANDS, DUALS, FRACTIONS)
)

_PREFIXES = ("وبال", "وال", "بال", "لل", "ال", "و", "ب", "ل")
_PLACEHOLDER = re.compile(r"\[[^\[\]\s]+_\d+\]")
_DIGIT_NUM = re.compile(r"\d+(?:[.,٫]\d+)?")
_WORD = re.compile(r"[ء-ي]+")


def _word_value(tok: str) -> tuple[str, float, str] | None:
    """(kind, value, prefix) for a number word, trying common prefixes ("وعشرين" -> tens, 20, "و")."""
    candidates = [("", tok)] + [(p, tok[len(p):]) for p in _PREFIXES if tok.startswith(p) and len(tok) > len(p) + 1]
    for prefix, t in candidates:
        for kind, table in (("unit", UNITS), ("ten", TEN), ("tens", TENS), ("hundred", HUNDREDS),
                            ("thousand", THOUSANDS), ("dual", DUALS), ("fraction", FRACTIONS)):
            if t in table:
                return kind, table[t], prefix
    return None


def _words_to_numbers(text: str) -> set[float]:
    found: set[float] = set()
    total, current, active = 0.0, 0.0, False

    def flush():
        nonlocal total, current, active
        if active:
            found.add(total + current)
        total, current, active = 0.0, 0.0, False

    for tok in _WORD.findall(text):
        wv = _word_value(tok)
        if wv is None:
            if tok != "و":          # a lone "و" between number words keeps the group going
                flush()
            continue
        kind, val, prefix = wv
        if kind == "hundred" and val == 100 and "ال" in prefix:
            flush()                             # "بالمئة" / "في المائة" = percent, not 100
            continue
        cont = active and prefix.startswith("و")
        if kind in ("dual", "fraction"):
            flush()
            found.add(val)
            continue
        if kind == "ten" and active and not cont and 0 < current < 10:
            current += 10                       # خمسة عشر = 15
        elif kind == "hundred" and val == 100 and active and not cont and 0 < current < 10:
            current *= 100                      # ثلاث مائة = 300
        elif kind == "thousand" and val == 1000 and active and not cont and current > 0:
            total += current * 1000             # خمسة آلاف = 5000
            current = 0.0
        else:
            if active and not cont:
                flush()
            if kind == "thousand":
                total += val
            else:
                current += val
        active = True
    flush()
    return found


def extract_numbers(text: str) -> set[float]:
    clean = _PLACEHOLDER.sub(" ", text)
    norm = normalize(clean)
    out: set[float] = set()
    for m in _DIGIT_NUM.finditer(norm):
        raw = m.group(0)
        # 1,000 -> 1000 ; 2.5 / 2٫5 -> 2.5
        if re.fullmatch(r"\d{1,3}(,\d{3})+", raw):
            out.add(float(raw.replace(",", "")))
        else:
            out.add(float(raw.replace("٫", ".").replace(",", ".")))
    out |= _words_to_numbers(norm)
    return out


def numbers_supported(claim_text: str, article_text: str, allowed: set[float] | None = None) -> tuple[bool, list[float]]:
    """Every number in the claim must appear in the article (or in its metadata: article no., law no., year)."""
    claim_nums = extract_numbers(claim_text)
    article_nums = extract_numbers(article_text) | (allowed or set())
    missing = sorted(n for n in claim_nums if not any(abs(n - a) < 1e-9 for a in article_nums))
    return (not missing), missing
