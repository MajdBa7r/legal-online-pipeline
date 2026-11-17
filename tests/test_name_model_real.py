"""Sanity checks with the REAL CAMeLBERT-da model. Skipped unless you opt in:

    PII_REAL_MODEL=1 python -m pytest tests/test_name_model_real.py -v      (Windows: set PII_REAL_MODEL=1)

Needs requirements-ml.txt and the model downloaded by scripts/download_name_model.py.
These are NOT accuracy tests and set NO threshold: they check that the real model loads, has a person label,
and that its spans come out well-formed. Accuracy is measured with scripts/eval_pii.py on the development set.
"""
import os

import pytest

pytestmark = pytest.mark.real_model
if os.environ.get("PII_REAL_MODEL") != "1":
    pytest.skip("real-model checks are opt-in (PII_REAL_MODEL=1)", allow_module_level=True)

from app.config import Settings  # noqa: E402
from app.pii.masker import PIIMasker  # noqa: E402
from app.pii.name_model import load_name_model  # noqa: E402
from app.pii.name_postprocess import is_excluded, is_subword_span  # noqa: E402

SENTENCES = [  # synthetic, written for this check; not part of any evaluation set
    "تواصلت مع براء الفلاحات من الموارد البشرية وما رد علي",
    "المالك بدو يطلعني من الشقة قبل ما يخلص العقد ورقمه 0791234567",
    "قام صاحب العمل غيث القيسي بإنهاء عقدي دون سابق إنذار",
]


@pytest.fixture(scope="module")
def model():
    s = Settings.from_env()
    return load_name_model("camelbert-da", model_id=s.pii_name_model_id, min_score=0.0,
                           revision=s.pii_name_model_revision or None)


def test_has_person_label(model):
    assert "PERS" in model.labels


@pytest.mark.parametrize("text", SENTENCES)
def test_spans_are_well_formed(model, text):
    for s, e in model.detect_names(text):
        assert 0 <= s < e <= len(text)
        assert not is_subword_span(text, s, e)
        assert not any(ch.isdigit() for ch in text[s:e])
        words = text[s:e].split()
        assert not is_excluded(words[0]) and not is_excluded(words[-1])
    r = PIIMasker(name_model=model).mask(text)
    assert PIIMasker.restore(r.text, r.mapping) == text


FILLER = [  # synthetic, no names (same style as scripts/diagnose_long_text.py)
    "اشتغلت بالشركة خمس سنين وما أخذت ولا إجازة سنوية.",
    "العقد كان لمدة سنة وتجدد تلقائيا مرتين.",
    "صاحب العمل بطلب مني أشتغل يوم الجمعة بدون أجر إضافي.",
    "الراتب كان يتأخر كل شهر أسبوعين تقريبا.",
    "قدمت شكوى لوزارة العمل وما حد رد علي لهلأ.",
    "بدي أعرف شو حقوقي إذا تركت الشغل قبل نهاية العقد.",
]
NAME_SENTENCE = "وبعدين حكيت مع غيث القيسي عن الراتب."


@pytest.mark.parametrize("punctuated", [True, False], ids=["punctuated", "run-on"])
@pytest.mark.parametrize("where", ["start", "middle", "end"])
def test_name_fully_masked_anywhere_in_long_natural_text(model, where, punctuated):
    """The name must be masked completely wherever it sits in a long question (sentence-level inference).

    Known limitation, deliberately NOT tested here: unnatural text that repeats one word hundreds of times
    (scripts/diagnose_long_text.py, sections 1-2) can hide a name from the model.
    """
    n = 150
    sents = [FILLER[i % len(FILLER)] for i in range(n)]
    sents.insert({"start": 0, "middle": n // 2, "end": n}[where], NAME_SENTENCE)
    if not punctuated:
        sents = [x.rstrip(".") for x in sents]
    text = " ".join(sents)
    assert len(model.windows(text)) > 1
    r = PIIMasker(name_model=model).mask(text)
    assert "غيث" not in r.text and "القيسي" not in r.text, r.mapping
