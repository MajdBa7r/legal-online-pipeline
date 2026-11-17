"""CamelbertNameModel and post-processing with a FAKE Hugging Face pipeline.

These tests check OUR code (label filter, threshold, windows, offsets, whole words, filtering, trimming,
failure policy). They say nothing about how well the real CAMeLBERT-da model finds names: that is measured
with the real model by scripts/eval_pii.py (development set) and tests/test_name_model_real.py (sanity only).
"""
import re
import sys

import pytest

from app.pii.masker import PIIMasker, PIIModelError
from app.pii.name_model import (CamelbertNameModel, NameModel, NameModelOutputError, NameModelUnavailable,
                                load_name_model)
from app.pii.name_postprocess import clean_model_spans, expand_to_words, is_subword_span


def fake_offsets(text):
    """Word pieces like BERT's: every word is cut into pieces of at most 3 characters; punctuation is a piece."""
    out = []
    for m in re.finditer(r"\w+|[^\w\s]", text):
        for i in range(m.start(), m.end(), 3):
            out.append((i, min(i + 3, m.end())))
    return out


class FakeNER:
    """Behaves like pipeline('token-classification', aggregation_strategy='simple') on the text it is given.

    entities: list of (surface, label, score); every occurrence in the window text is returned with
    window-relative offsets. Records every window it was called with.
    """

    def __init__(self, entities=(), *, drop_offsets=False):
        self.entities, self.drop_offsets, self.calls = list(entities), drop_offsets, []

    def __call__(self, text):
        self.calls.append(text)
        out = []
        for surface, label, score in self.entities:
            for m in re.finditer(re.escape(surface), text):
                ent = {"entity_group": label, "score": score, "word": surface, "start": m.start(), "end": m.end()}
                if self.drop_offsets:
                    ent["start"] = ent["end"] = None
                out.append(ent)
        return out


def camel(entities=(), **kw):
    ner = FakeNER(entities, drop_offsets=kw.pop("drop_offsets", False))
    return CamelbertNameModel(ner, fake_offsets, **kw), ner


def masked(model, text, **kw):
    return PIIMasker(name_model=model, **kw).mask(text)


# ---------------------------------------------------------------- label and score
def test_satisfies_protocol():
    assert isinstance(camel()[0], NameModel)


def test_only_person_labels_are_used():
    m, _ = camel([("براء الفلاحات", "PERS", 0.99), ("اربد", "LOC", 0.99), ("أورانج", "ORG", 0.99)])
    text = "تواصلت مع براء الفلاحات من فرع اربد في أورانج"
    assert masked(m, text).text == "تواصلت مع [شخص_1] من فرع اربد في أورانج"


def test_bio_labels_are_accepted():
    m, _ = camel([("براء", "B-PERS", 0.9)])
    assert masked(m, "سألت براء عن العقد").text == "سألت [شخص_1] عن العقد"


def test_min_score_filters_low_confidence_spans():
    ents = [("براء", "PERS", 0.95), ("غيث", "PERS", 0.40)]
    text = "براء وغيث اشتغلوا عندي"
    assert len(camel(ents, min_score=0.0)[0].detect_names(text)) == 2
    assert camel(ents, min_score=0.5)[0].detect_names(text) == [(0, 4)]


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        camel(min_score=1.5)
    with pytest.raises(ValueError):
        camel(max_pieces=600)
    with pytest.raises(ValueError):
        camel(max_pieces=100, overlap_pieces=100)


# ---------------------------------------------------------------- post-processing
def test_subword_span_grows_to_the_whole_word():
    m, _ = camel([("عما", "PERS", 0.9)])          # model flags the first piece of "عمار" only
    text = "يا جماعة عمارة صاحب المحل"            # (not a name in the list, so rules leave it)
    assert masked(m, text).text == "يا جماعة [شخص_1] صاحب المحل"
    assert expand_to_words("ايميل", 0, 3) == (0, 5)
    assert is_subword_span("ايميل", 0, 3) and not is_subword_span("ايميل", 0, 5)


@pytest.mark.parametrize("surface", ["2023", "و", "،", "@", "١٢٣", " "])
def test_invalid_spans_are_dropped(surface):
    m, _ = camel([(surface, "PERS", 0.99)])
    text = "قدمت الشكية سنة 2023 ، ١٢٣ @ و ."      # each surface appears only on its own
    assert m.detect_names(text) == []


def test_span_is_split_at_punctuation_and_numbers():
    text = "براء، 0791234567 غيث"
    assert clean_model_spans(text, [(0, len(text))]) == [(0, 4), (17, 20)]


@pytest.mark.parametrize("flagged,expected", [
    ("المالك سند", "المالك [شخص_1]"),               # role word at the start is trimmed, the name stays
    ("سند المالك", "[شخص_1] المالك"),               # role word at the end is trimmed
    ("المالك", "المالك"),                            # role word alone: nothing masked
    ("زوجتي ريم", "زوجتي [شخص_1]"),
    ("والمالك سند", "والمالك [شخص_1]"),              # attached و is recognised
])
def test_exclusion_trimming_only_at_edges(flagged, expected):
    m, _ = camel([(flagged, "PERS", 0.99)])
    text = f"قال {flagged} إنه بدو الشقة"
    assert masked(m, text).text == f"قال {expected} إنه بدو الشقة"


def test_excluded_word_inside_a_span_is_kept_masked():
    m, _ = camel([("سند المالك براء", "PERS", 0.99)])
    assert masked(m, "قال سند المالك براء كلام").text == "قال [شخص_1] كلام"


# ---------------------------------------------------------------- long texts and windows
def _long_text(n_words=1500, tail=" وبعدها حكيت مع براء الفلاحات"):
    return " ".join(["الشقة"] * n_words) + tail


def test_short_text_is_one_window():
    m, ner = camel([("براء", "PERS", 0.9)])
    m.detect_names("حكيت مع براء")
    assert ner.calls == ["حكيت مع براء"]


def test_long_text_uses_windows_within_the_limit():
    m, ner = camel([("براء الفلاحات", "PERS", 0.9)], max_pieces=120, overlap_pieces=30)
    text = _long_text()
    spans = m.detect_names(text)
    assert len(ner.calls) > 5
    assert all(len(fake_offsets(w)) <= 120 + 10 for w in ner.calls)   # + a few pieces from word snapping
    start = text.index("براء")
    assert spans == [(start, start + len("براء الفلاحات"))]          # name at the very end is found
    r = masked(m, text)
    assert r.text.endswith("حكيت مع [شخص_1]") and r.mapping == {"[شخص_1]": "براء الفلاحات"}


def test_windows_cover_the_whole_text_and_overlap():
    m, _ = camel(max_pieces=60, overlap_pieces=20)
    text = _long_text(400)
    wins = m.windows(text)
    assert wins[0][0] == 0 and wins[-1][1] == len(text)
    for (a1, b1), (a2, b2) in zip(wins, wins[1:]):
        assert a2 < b1                                       # consecutive windows overlap
    for a, b in wins:                                        # edges never cut a word
        assert not is_subword_span(text, a, b)


def test_name_on_a_window_boundary_is_found_whole():
    m, _ = camel([("براء الفلاحات", "PERS", 0.9)], max_pieces=60, overlap_pieces=20)
    body = " ".join(["الشقة"] * 400)
    for cut in range(100, 300, 7):                           # slide the name across many positions
        text = body[:cut * 6] + " براء الفلاحات " + body[cut * 6:]
        s = text.index("براء")
        assert (s, s + len("براء الفلاحات")) in m.detect_names(text), cut


def test_offsets_are_shifted_to_the_original_text():
    m, _ = camel([("غيث", "PERS", 0.9)], max_pieces=60, overlap_pieces=20)
    text = _long_text(300, tail=" غيث")
    for s, e in m.detect_names(text):
        assert text[s:e] == "غيث"


def test_empty_text():
    m, ner = camel([("براء", "PERS", 0.9)])
    assert m.detect_names("") == [] and m.detect_names("   ") == [] and ner.calls == []


# ---------------------------------------------------------------- failures and integration
def test_entity_without_offsets_fails_closed():
    m, _ = camel([("براء", "PERS", 0.9)], drop_offsets=True)
    with pytest.raises(NameModelOutputError):
        m.detect_names("حكيت مع براء")
    with pytest.raises(PIIModelError):
        PIIMasker(name_model=m, model_error_policy="fail_closed").mask("حكيت مع براء")
    r = PIIMasker(name_model=m, model_error_policy="fallback").mask("حكيت مع محمد وبراء")
    assert r.model_error == "NameModelOutputError"                    # development: rules only, error reported
    assert r.text == "حكيت مع [شخص_1] وبراء"


def test_structured_detection_is_unchanged_by_the_model():
    m, _ = camel([("0791234567", "PERS", 0.99), ("ali.k", "PERS", 0.99), ("براء", "PERS", 0.99)])
    text = "براء رقمه 0791234567 وايميله ali.k@example.com ورقمه الوطني 9871234567"
    r = masked(m, text)
    assert r.text == "[شخص_1] رقمه [هاتف_1] وايميله [بريد_1] ورقمه الوطني [رقم_وطني_1]"
    assert PIIMasker.restore(r.text, r.mapping) == text


def test_model_with_no_detections_changes_nothing():
    m, _ = camel([])
    plain = PIIMasker()
    for text in ["اسمي محمد الزعبي ورقمي 0791234567", "المالك بدو يطلعني", "أبو خالد صاحب المحل"]:
        assert masked(m, text).text == plain.mask(text).text


def test_missing_ml_libraries_report_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)    # makes "import transformers" fail
    with pytest.raises(NameModelUnavailable, match="requirements-ml.txt"):
        load_name_model("camelbert-da", model_id="CAMeL-Lab/bert-base-arabic-camelbert-da-ner", min_score=0.0)


# ---------------------------------------------------------------- sentence-level inference
FILLER = [  # synthetic, no names
    "اشتغلت بالشركة خمس سنين وما أخذت ولا إجازة سنوية.",
    "العقد كان لمدة سنة وتجدد تلقائيا مرتين.",
    "صاحب العمل بطلب مني أشتغل يوم الجمعة بدون أجر إضافي.",
    "الراتب كان يتأخر كل شهر أسبوعين تقريبا؟",
    "قدمت شكوى لوزارة العمل وما حد رد علي لهلأ!",
    "بدي أعرف شو حقوقي إذا تركت الشغل قبل نهاية العقد\n",
]
NAME_SENTENCE = "وبعدين حكيت مع غيث القيسي عن الراتب."


def _natural(n, at=None, punctuated=True):
    sents = [FILLER[i % len(FILLER)] for i in range(n)]
    if at is not None:
        sents.insert(at, NAME_SENTENCE)
    if not punctuated:
        sents = [re.sub(r"[.؟!\n]", "", x) for x in sents]
    return " ".join(sents)


def test_sentences_split_at_the_documented_boundaries():
    text = "سؤال أول. سؤال ثاني؟ ثالث! رابع؛ خامس\nسادس"
    parts = [text[a:b] for a, b in CamelbertNameModel.sentences(text)]
    assert parts == ["سؤال أول.", "سؤال ثاني؟", "ثالث!", "رابع؛", "خامس", "سادس"]


def test_dots_inside_numbers_and_emails_do_not_split():
    text = "الأجرة 3.5 دينار وايميلي ali.k@example.com والموقع www.site.jo آخر جملة"
    assert CamelbertNameModel.sentences(text) == [(0, len(text))]


def test_short_sentences_are_batched_within_the_limit():
    m, _ = camel(max_pieces=60, overlap_pieces=16)
    text = _natural(40)
    wins = m.windows(text)
    assert 1 < len(wins) < 40                                 # batched, not one call per sentence
    sents = CamelbertNameModel.sentences(text)
    for a, b in wins:
        assert len(fake_offsets(text[a:b])) <= 60             # never more than the limit
        assert any(a == s for s, _ in sents) and any(b == e for _, e in sents)   # whole sentences only


def test_default_chunk_is_128_pieces_with_32_overlap():
    m, _ = camel()
    assert (m.max_pieces, m.overlap) == (128, 32)


@pytest.mark.parametrize("punctuated", [True, False])
def test_every_character_is_seen(punctuated):
    m, _ = camel(max_pieces=60, overlap_pieces=16)
    text = _natural(80, punctuated=punctuated)
    seen = {i for a, b in m.windows(text) for i in range(a, b)}
    assert all(i in seen for i, ch in enumerate(text) if not ch.isspace())


@pytest.mark.parametrize("punctuated", [True, False])
@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_name_found_at_start_middle_and_end_of_long_natural_text(where, punctuated):
    m, ner = camel([("غيث القيسي", "PERS", 0.9)])
    n = 150
    at = {"start": 0, "middle": n // 2, "end": n}[where]
    text = _natural(n, at=at, punctuated=punctuated)
    r = PIIMasker(name_model=m).mask(text)
    assert r.mapping == {"[شخص_1]": "غيث القيسي"} and "غيث" not in r.text
    assert all(len(fake_offsets(w)) <= 128 + 10 for w in ner.calls)
