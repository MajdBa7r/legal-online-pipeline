"""Optional extra name detector for the PII masker.

PII_NAME_MODEL=none (default) never loads anything. PII_NAME_MODEL=camelbert-da loads
CAMeL-Lab/bert-base-arabic-camelbert-da-ner locally on CPU (transformers + torch from requirements-ml.txt).
The model runs in this process; text is never sent over the network.

CamelbertNameModel:
* keeps only person entities (PERS) with score >= min_score (min_score is tuned on the DEVELOPMENT set only);
* runs SENTENCE BY SENTENCE (see `windows`): the model was measured to lose names when they sit after a long
  stretch of text (scripts/diagnose_long_text.py), so it never sees more than CHUNK_PIECES pieces at once;
  consecutive short sentences share a call up to that limit, and a long unpunctuated sentence is cut into
  overlapping windows of CHUNK_PIECES pieces (OVERLAP_PIECES shared) snapped to whole words;
* passes raw spans through app/pii/name_postprocess.py (whole words, invalid-span filter, exclusion trimming).

The model is wrapped in two plain callables (predict, piece_offsets), so tests can use a fake pipeline.
"""
from __future__ import annotations

import bisect
import re
import time
from typing import Any, Callable, Iterable, Protocol, Sequence, runtime_checkable

from app.pii.name_postprocess import clean_model_spans, is_word_char, load_exclusions

SUPPORTED_NAME_MODELS = ("none", "camelbert-da")
PERSON_LABELS = frozenset({"PERS", "PER", "PERSON"})
MODEL_MAX_PIECES = 512
MAX_WINDOW_PIECES = 450   # leaves room for the 2 special tokens and for snapping window edges to whole words
CHUNK_PIECES = 128        # most pieces the model sees in one call (a design choice, not tuned on any data)
OVERLAP_PIECES = 32       # pieces shared by consecutive windows inside one long unpunctuated sentence
# A sentence ends at ؟ ? ! ؛ ; or a new line, always; at "." only before whitespace or the end of the text,
# so "3.5", "ali.k@example.com" and "www.site.jo" are not cut.
_SENTENCE_END = re.compile(r"[؟?!؛;\n]+|\.+(?=\s|$)")
_SNAP_LIMIT = 40   # never move a window edge more than this many characters to reach a word boundary


@runtime_checkable
class NameModel(Protocol):
    """Returns (start, end) character spans of person names in the ORIGINAL text.

    Runs locally; must never send text over the network.
    """

    name: str

    def detect_names(self, text: str) -> Iterable[tuple[int, int]]: ...


class NameModelUnavailable(RuntimeError):
    """The configured name model could not be loaded."""


class NameModelOutputError(RuntimeError):
    """The model returned something we cannot map safely to the text (e.g. an entity without offsets)."""


Predict = Callable[[str], Sequence[dict[str, Any]]]
PieceOffsets = Callable[[str], Sequence[tuple[int, int]]]


def _label(ent: dict[str, Any]) -> str:
    raw = ent.get("entity_group") or ent.get("entity") or ent.get("label") or ""
    return re.sub(r"^[BI]-", "", str(raw).upper())


class CamelbertNameModel:
    name = "camelbert-da"

    def __init__(self, predict: Predict, piece_offsets: PieceOffsets, *, min_score: float = 0.0,
                 max_pieces: int = CHUNK_PIECES, overlap_pieces: int = OVERLAP_PIECES, exclusions: frozenset[str] | None = None,
                 source: str = ""):
        if not 0 < overlap_pieces < max_pieces <= MAX_WINDOW_PIECES:
            raise ValueError(f"need 0 < overlap_pieces < max_pieces <= {MAX_WINDOW_PIECES}")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be in [0, 1], got {min_score}")
        self.predict, self.piece_offsets = predict, piece_offsets
        self.min_score, self.max_pieces, self.overlap = float(min_score), max_pieces, overlap_pieces
        self.exclusions = load_exclusions() if exclusions is None else exclusions
        self.source = source

    # -- chunks --------------------------------------------------------------------
    @staticmethod
    def sentences(text: str) -> list[tuple[int, int]]:
        """(start, end) of each sentence, terminator included, surrounding whitespace excluded."""
        out, a = [], 0
        for m in _SENTENCE_END.finditer(text):
            out.append((a, m.end()))
            a = m.end()
        out.append((a, len(text)))
        trimmed = []
        for a, b in out:
            while a < b and text[a].isspace():
                a += 1
            while b > a and text[b - 1].isspace():
                b -= 1
            if text[a:b].strip():
                trimmed.append((a, b))
        return trimmed

    def windows(self, text: str) -> list[tuple[int, int]]:
        """Character ranges to run the model on, each at most `max_pieces` pieces (plus word snapping).

        1. Split into sentences. 2. Join consecutive sentences while the joined chunk stays within max_pieces.
        3. A single sentence longer than max_pieces is cut into overlapping windows (overlap_pieces shared).
        Every non-space character of the text is in at least one range.
        """
        if not text.strip():
            return []
        offsets = [(int(s), int(e)) for s, e in self.piece_offsets(text) if e > s]

        offsets.sort()
        starts, ends = [o[0] for o in offsets], [o[1] for o in offsets]

        def pieces_in(a: int, b: int) -> list[tuple[int, int]]:
            return offsets[bisect.bisect_left(starts, a):bisect.bisect_right(ends, b)]

        out: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        cur_n = 0
        for a, b in self.sentences(text):
            n = len(pieces_in(a, b))
            if n > self.max_pieces:                       # long sentence: own overlapping windows
                if cur:
                    out.append(cur)
                    cur, cur_n = None, 0
                out.extend(self._split_long(text, pieces_in(a, b)))
            elif cur is not None and cur_n + n <= self.max_pieces:
                cur, cur_n = (cur[0], b), cur_n + n       # batch with the previous short sentences
            else:
                if cur:
                    out.append(cur)
                cur, cur_n = (a, b), n
        if cur:
            out.append(cur)
        return out

    def _split_long(self, text: str, offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
        out, step, i = [], self.max_pieces - self.overlap, 0
        while True:
            j = min(i + self.max_pieces, len(offsets))
            a, b = offsets[i][0], offsets[j - 1][1]
            a2 = a
            while a2 > 0 and a - a2 < _SNAP_LIMIT and is_word_char(text[a2 - 1]):
                a2 -= 1
            b2 = b
            while b2 < len(text) and b2 - b < _SNAP_LIMIT and is_word_char(text[b2]):
                b2 += 1
            out.append((a2, b2))
            if j == len(offsets):
                return out
            i += step

    # -- detection -------------------------------------------------------------
    def raw_spans(self, text: str) -> list[tuple[int, int, float]]:
        """Person spans with scores, before post-processing (for diagnostics and the benchmark)."""
        out = []
        for a, b in self.windows(text):
            for ent in self.predict(text[a:b]) or []:
                if _label(ent) not in PERSON_LABELS:
                    continue
                score = float(ent.get("score", 1.0))
                s, e = ent.get("start"), ent.get("end")
                if s is None or e is None:
                    raise NameModelOutputError("person entity without character offsets")
                out.append((a + int(s), a + int(e), score))
        return out

    def detect_names(self, text: str) -> list[tuple[int, int]]:
        kept = [(s, e) for s, e, score in self.raw_spans(text) if score >= self.min_score]
        return clean_model_spans(text, kept, self.exclusions)


# ------------------------------------------------------------------ loading
def load_camelbert(model_id: str, *, min_score: float, revision: str | None = None,
                   **model_kwargs: Any) -> CamelbertNameModel:
    """Load the Hugging Face model on CPU. Every failure is reported as NameModelUnavailable."""
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline
    except ImportError as e:  # torch / transformers not installed
        raise NameModelUnavailable("transformers/torch not installed (pip install -r requirements-ml.txt)") from e
    t0 = time.perf_counter()
    try:
        kw = {"revision": revision} if revision else {}
        tokenizer = AutoTokenizer.from_pretrained(model_id, **kw)
        model = AutoModelForTokenClassification.from_pretrained(model_id, **kw)
    except Exception as e:  # not downloaded and offline, wrong id, corrupt files ...
        raise NameModelUnavailable(f"could not load {model_id!r}: {type(e).__name__}: {e}") from e
    if not getattr(tokenizer, "is_fast", False):
        raise NameModelUnavailable("a fast tokenizer is required for character offsets")
    labels = {re.sub(r"^[BI]-", "", str(v).upper()) for v in model.config.id2label.values()}
    if not labels & PERSON_LABELS:
        raise NameModelUnavailable(f"{model_id!r} has no person label (labels: {sorted(labels)})")
    model.eval()
    ner = pipeline("token-classification", model=model, tokenizer=tokenizer, aggregation_strategy="simple",
                   device=-1)

    def piece_offsets(text: str) -> list[tuple[int, int]]:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
        return [tuple(o) for o in enc["offset_mapping"]]

    m = CamelbertNameModel(lambda t: ner(t), piece_offsets, min_score=min_score,
                           source=f"{model_id}@{revision or 'default'}", **model_kwargs)
    m.load_seconds = time.perf_counter() - t0  # type: ignore[attr-defined]
    m.labels = sorted(labels)                  # type: ignore[attr-defined]
    return m


def load_name_model(kind: str, *, model_id: str, min_score: float, revision: str | None = None) -> NameModel:
    if kind not in SUPPORTED_NAME_MODELS or kind == "none":
        raise ValueError(f"PII_NAME_MODEL must be one of {SUPPORTED_NAME_MODELS[1:]} to load a model, got {kind!r}")
    if kind == "camelbert-da":
        return load_camelbert(model_id, min_score=min_score, revision=revision)
    raise NameModelUnavailable(f"unknown name model {kind!r}")  # pragma: no cover
