"""Stage 1: mask personal data before anything leaves the server.

Placeholders keep a type and a number ([شخص_1], [هاتف_1], ...). The same value
always gets the same placeholder within a request. The map lives only in the
request object; it is never logged or sent to a model.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from app.text.arabic import to_latin_digits

if TYPE_CHECKING:
    from app.pii.name_model import NameModel

log = logging.getLogger(__name__)
ModelErrorPolicy = Literal["fallback", "fail_closed"]

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


class PIIModelError(RuntimeError):
    """The optional name model failed on a request and the policy is fail_closed."""


@dataclass
class MaskResult:
    text: str
    mapping: dict[str, str] = field(default_factory=dict)  # placeholder -> original
    model_error: str | None = None  # set when the name model failed and rules-only masking was used

    @property
    def originals(self) -> list[str]:
        return list(self.mapping.values())


class PIIMasker:
    """Rule-based masker, with an optional extra name detector.

    With name_model=None and family_name_expansion=False (the defaults) the behaviour is exactly the rule-based
    masker: tests/test_masker_golden.py compares every output with outputs recorded before these options existed.
    family_name_expansion (experimental, off by default, evaluated in Phase 3D) extends a detected name by one
    following family name: an "ال" word, or بني/بنو/أبو/آل + a word ("عمار السرحان", "سند بني ياسين").
    Role and family words from name_exclusions.txt are never taken ("محمد المحاسب" keeps المحاسب visible).
    """

    def __init__(self, common_names: set[str] | None = None, *, name_model: "NameModel | None" = None,
                 model_error_policy: ModelErrorPolicy = "fail_closed", family_name_expansion: bool = False):
        self.names = common_names if common_names is not None else _load_names(_HERE / "common_names.txt")
        self.name_model = name_model
        self.family_name_expansion = bool(family_name_expansion)
        if model_error_policy not in ("fallback", "fail_closed"):
            raise ValueError(f"model_error_policy must be 'fallback' or 'fail_closed', got {model_error_policy!r}")
        self.model_error_policy = model_error_policy

    # -- public ---------------------------------------------------------------
    def spans(self, text: str) -> list[tuple[int, int, str]]:
        """(start, end, type) of everything mask() would replace, in text order."""
        return self._spans(text)[0]

    def mask(self, text: str) -> MaskResult:
        spans, model_error = self._spans(text)

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
        return MaskResult(text="".join(out), mapping=mapping, model_error=model_error)

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

    # -- span collection ------------------------------------------------------
    def _rule_spans(self, text: str) -> list[tuple[int, int, str]]:
        probe = to_latin_digits(text)  # same length as text, so spans line up
        spans: list[tuple[int, int, str]] = []
        for kind, pattern in PATTERNS:
            for m in pattern.finditer(probe):
                spans.append((m.start(), m.end(), kind))
        spans.extend(self._name_spans(text))
        return self._drop_overlaps(spans)

    def _spans(self, text: str) -> tuple[list[tuple[int, int, str]], str | None]:
        spans, model_error = self._detected_spans(text)
        if self.family_name_expansion:
            spans = self._expand_family_names(text, spans)
        return spans, model_error

    def _detected_spans(self, text: str) -> tuple[list[tuple[int, int, str]], str | None]:
        rule_spans = self._rule_spans(text)
        if self.name_model is None:
            return rule_spans, None
        try:
            model_spans = list(self.name_model.detect_names(text))
        except Exception as e:  # any failure of the optional model
            if self.model_error_policy == "fail_closed":
                raise PIIModelError(f"name model {getattr(self.name_model, 'name', '?')} failed: "
                                    f"{type(e).__name__}") from e
            log.warning("PII NAME MODEL FAILED (%s); this request is masked with rules only", type(e).__name__)
            return rule_spans, type(e).__name__
        return self._add_model_names(text, rule_spans, model_spans), None

    @staticmethod
    def _add_model_names(text: str, rule_spans: list[tuple[int, int, str]],
                         model_spans: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
        """Add model name spans to the rule spans.

        * Structured spans (email, IBAN, phone, national ID) always win: a model span touching one is dropped.
        * Name spans (rule and model) that overlap, or are separated only by whitespace, are merged into one,
          so "[rule: عمار] + [model: عمار السرحان]" masks "عمار السرحان" as a single name.
        """
        n = len(text)
        structured = [s for s in rule_spans if s[2] != "name"]
        names = [(s, e) for s, e, t in rule_spans if t == "name"]
        for s, e in model_spans:
            s, e = max(0, int(s)), min(n, int(e))
            if e <= s or not text[s:e].strip():
                continue
            if any(s < se and ss < e for ss, se, _ in structured):
                continue
            names.append((s, e))
        merged: list[list[int]] = []
        for s, e in sorted(names):
            if merged and (s <= merged[-1][1] or text[merged[-1][1]:s].strip() == ""):
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        out = structured + [(s, e, "name") for s, e in merged]
        return sorted(out)

    # -- family-name expansion (experimental, off by default) ------------------
    def _expand_family_names(self, text: str, spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        from app.pii.name_postprocess import is_excluded  # only needed when the option is on
        from app.text.arabic import normalize

        tokens = [(m.start(), m.end(), m.group(0)) for m in _TOKEN.finditer(text)]
        taken = [(s, e) for s, e, _ in spans]

        def next_token(pos: int) -> int | None:
            """Index of the token right after pos, if only spaces (no punctuation, no newline) lie between."""
            k = next((k for k, (s, _, _) in enumerate(tokens) if s >= pos), None)
            if k is None:
                return None
            gap = text[pos:tokens[k][0]]
            if not gap or gap.strip() or "\n" in gap:
                return None
            return k

        def usable(k: int) -> bool:
            s, e, tok = tokens[k]
            return (len(tok) >= 2 and tok not in NAME_STOP and not is_excluded(tok)
                    and not any(s < te and ts < e for ts, te in taken))

        out = []
        for s, e, kind in spans:
            if kind == "name":
                k = next_token(e)
                if k is not None and usable(k):
                    tok = normalize(tokens[k][2])
                    if tok in {"بني", "بنو", "ابو", "ال"}:          # بني ياسين / أبو زيد / آل ...
                        k2 = next_token(tokens[k][1])
                        if k2 is not None and usable(k2):
                            e = tokens[k2][1]
                    elif tok.startswith("ال") and len(tok) >= 4:    # السرحان
                        e = tokens[k][1]
            out.append((s, e, kind))
        merged: list[tuple[int, int, str]] = []          # join name + name only; other types never merge
        for s, e, kind in sorted(out):
            if (merged and kind == "name" and merged[-1][2] == "name"
                    and text[merged[-1][1]:s].strip() == "" and "\n" not in text[merged[-1][1]:s]):
                merged[-1] = (merged[-1][0], max(merged[-1][1], e), "name")
            else:
                merged.append((s, e, kind))
        return merged

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
