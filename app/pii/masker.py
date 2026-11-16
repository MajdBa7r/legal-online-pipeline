"""Stage 1: mask personal data before anything leaves the server.

Placeholders keep a type and a number ([شخص_1], [هاتف_1], ...). The same value
always gets the same placeholder within a request. The map lives only in the
request object; it is never logged or sent to a model.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from app.text.arabic import to_latin_digits

_HERE = Path(__file__).resolve().parent

LABELS = {
    "email": "بريد",
    "iban": "آيبان",
    "phone": "هاتف",
    "national_id": "رقم_وطني",
    "name": "شخص",
}

# Order matters: earlier patterns win on overlap.
PATTERNS: list[tuple[str, re.Pattern]] = [
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("iban", re.compile(r"\bJO\d{2}(?:[ ]?[A-Z0-9]){26}\b", re.IGNORECASE)),
    ("phone", re.compile(r"(?<!\d)(?:\+962|00962|0)[\s-]?7[789](?:[\s-]?\d){7}(?!\d)")),
    # 10-digit Jordanian national number. Confirm prefix rules with the legal members.
    ("national_id", re.compile(r"(?<![\d.,])\d{10}(?![\d.,])")),
]

# A name almost always follows these words.
STRONG_NAME_TRIGGERS = {"اسمي", "إسمي", "المدعو", "المدعوة", "السيد", "السيدة", "الأستاذ", "الأستاذة",
                        "الدكتور", "الدكتورة", "الحاج", "الحاجة", "اسمه", "اسمها"}
# These are followed by a name only sometimes: mask the next token only if it is a known name.
WEAK_NAME_TRIGGERS = {"أنا", "انا", "أبو", "ابو", "أم", "ام", "ابني", "بنتي", "زوجي", "زوجتي",
                      "مديري", "صاحب", "المالك", "المستأجر", "أخوي", "أختي"}
# Words that end a name after a strong trigger.
NAME_STOP = {"في", "من", "على", "عن", "إلى", "الى", "و", "وأنا", "وانا", "أنا", "انا", "هل", "بدي",
             "عمري", "اشتغل", "أشتغل", "اعمل", "أعمل", "موظف", "موظفة", "عامل", "عاملة", "عندي",
             "صاحب", "مستأجر", "مالك", "بشركة", "بالشركة", "لدى", "عند", "منذ", "من", "قال", "كان"}

_TOKEN = re.compile(r"[ء-ي]+")
_PLACEHOLDER = re.compile(r"\[[^\[\]\s]+_\d+\]")


def _load_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    }


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original

    @property
    def originals(self) -> list[str]:
        return list(self.mapping.values())


class PIIMasker:
    def __init__(self, common_names: set[str] | None = None):
        self.names = common_names if common_names is not None else _load_names(_HERE / "common_names.txt")

    # -- public ---------------------------------------------------------------
    def mask(self, text: str) -> MaskResult:
        probe = to_latin_digits(text)  # same length as text, so spans line up
        spans: list[tuple[int, int, str]] = []
        for kind, pattern in PATTERNS:
            for m in pattern.finditer(probe):
                spans.append((m.start(), m.end(), kind))
        spans.extend(self._name_spans(text))
        spans = self._drop_overlaps(spans)

        mapping: dict[str, str] = {}
        by_value: dict[tuple[str, str], str] = {}
        counters: dict[str, int] = {}
        out, last = [], 0
        for start, end, kind in spans:
            value = text[start:end]
            key = (kind, value.strip())
            if key not in by_value:
                counters[kind] = counters.get(kind, 0) + 1
                placeholder = f"[{LABELS[kind]}_{counters[kind]}]"
                by_value[key] = placeholder
                mapping[placeholder] = value
            out.append(text[last:start])
            out.append(by_value[key])
            last = end
        out.append(text[last:])
        return MaskResult(text="".join(out), mapping=mapping)

    @staticmethod
    def restore(text: str | None, mapping: dict[str, str]) -> str | None:
        if not text or not mapping:
            return text
        return _PLACEHOLDER.sub(lambda m: mapping.get(m.group(0), m.group(0)), text)

    @staticmethod
    def leaked(texts: list[str], originals: list[str]) -> list[str]:
        """Original values (3+ chars) that still appear in outgoing text."""
        hits = []
        for value in originals:
            v = value.strip()
            if len(v) >= 3 and any(v in t for t in texts):
                hits.append(v)
        return hits

    # -- names ----------------------------------------------------------------
    def _name_spans(self, text: str) -> list[tuple[int, int, str]]:
        tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]
        spans: list[tuple[int, int, str]] = []
        i = 0
        while i < len(tokens):
            _, _, tok = tokens[i]
            if tok in STRONG_NAME_TRIGGERS:
                j = i + 1
                taken = []
                while j < len(tokens) and len(taken) < 3:
                    nxt = tokens[j][2]
                    if nxt in NAME_STOP or self._gap_has_punct(text, tokens[j - 1][1], tokens[j][0]):
                        break
                    taken.append(tokens[j])
                    j += 1
                if taken:
                    spans.append((taken[0][0], taken[-1][1], "name"))
                    i = j
                    continue
            elif tok in WEAK_NAME_TRIGGERS and i + 1 < len(tokens) and tokens[i + 1][2] in self.names:
                j = i + 1
                end = tokens[j][1]
                # take a following family/second name if it is also in the list
                if j + 1 < len(tokens) and tokens[j + 1][2] in self.names:
                    end = tokens[j + 1][1]
                    j += 1
                start = tokens[i][0] if tok in {"أبو", "ابو", "أم", "ام"} else tokens[i + 1][0]
                spans.append((start, end, "name"))
                i = j + 1
                continue
            elif tok in self.names:
                spans.append((tokens[i][0], tokens[i][1], "name"))
            i += 1
        return self._merge_adjacent_names(spans, text)

    @staticmethod
    def _gap_has_punct(text: str, a: int, b: int) -> bool:
        return any(ch in "،,.؟?!:؛;()\n" for ch in text[a:b])

    @staticmethod
    def _merge_adjacent_names(spans, text):
        merged: list[tuple[int, int, str]] = []
        for s in sorted(spans):
            if merged and merged[-1][2] == "name" and text[merged[-1][1]:s[0]].strip() == "":
                merged[-1] = (merged[-1][0], s[1], "name")
            else:
                merged.append(s)
        return merged

    @staticmethod
    def _drop_overlaps(spans):
        kept: list[tuple[int, int, str]] = []
        for s in sorted(spans, key=lambda x: (x[0], -(x[1] - x[0]))):
            if kept and s[0] < kept[-1][1]:
                continue
            kept.append(s)
        return kept
