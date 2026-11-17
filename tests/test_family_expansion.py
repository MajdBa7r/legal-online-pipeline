"""Family-name expansion: experimental, OFF by default, accepted or rejected in Phase 3D."""
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.pii.factory import build_masker
from app.pii.masker import PIIMasker

GOLDEN = Path(__file__).parent / "fixtures" / "masker_golden.jsonl"


class Names:
    name = "fake"

    def __init__(self, *words):
        self.words = words

    def detect_names(self, text):
        return [(text.index(w), text.index(w) + len(w)) for w in self.words if w in text]


def fam(model=None):
    return PIIMasker(name_model=model, family_name_expansion=True)


def test_off_by_default():
    assert PIIMasker().family_name_expansion is False
    assert Settings().pii_family_name_expansion is False
    m, st = build_masker(Settings())
    assert m.family_name_expansion is False and st.family_expansion is False


def test_setting_turns_it_on(monkeypatch):
    monkeypatch.setenv("PII_FAMILY_NAME_EXPANSION", "on")
    m, st = build_masker(Settings.from_env())
    assert m.family_name_expansion is True and st.family_expansion is True


@pytest.mark.parametrize("text,expected", [
    ("محمد المحاسب رفض يعطيني راتبي", "[شخص_1] المحاسب رفض يعطيني راتبي"),
    ("خالد المالك بدو يطلعني من الشقة", "[شخص_1] المالك بدو يطلعني من الشقة"),
    ("ريم الموظفة بالشركة قالت لا", "[شخص_1] الموظفة بالشركة قالت لا"),
    ("يزن المستأجر ما دفع", "[شخص_1] المستأجر ما دفع"),
])
def test_role_words_are_never_taken(text, expected):
    assert fam().mask(text).text == expected


def test_family_name_is_masked_with_the_first_name():
    text = "يا جماعة عمار السرحان صاحب المحل باعني تلفون خربان"
    assert PIIMasker().mask(text).text.startswith("يا جماعة [شخص_1] السرحان")     # off: family name visible
    r = fam().mask(text)
    assert r.text.startswith("يا جماعة [شخص_1] صاحب") and r.mapping == {"[شخص_1]": "عمار السرحان"}


def test_two_word_family_name():
    r = fam(Names("سند")).mask("سند بني ياسين ما دفعلي الأجرة")
    assert r.text == "[شخص_1] ما دفعلي الأجرة" and r.mapping == {"[شخص_1]": "سند بني ياسين"}
    r = fam().mask("محمد أبو زيد صاحب المحل")
    assert r.mapping == {"[شخص_1]": "محمد أبو زيد"}


@pytest.mark.parametrize("text", ["عمار، السرحان قال", "عمار. السرحان قال", "عمار\nالسرحان قال",
                                  "عمار: السرحان قال", "عمار (السرحان) قال"])
def test_punctuation_or_newline_stops_expansion(text):
    assert fam().mask(text).mapping == {"[شخص_1]": "عمار"}


def test_only_one_family_word_is_taken():
    r = fam().mask("عمار السرحان الطويل قال")
    assert r.mapping == {"[شخص_1]": "عمار السرحان"}


def test_clitic_words_and_stop_words_are_not_taken():
    assert fam().mask("عمار والمالك اتفقوا").mapping == {"[شخص_1]": "عمار"}
    assert fam().mask("عمار في الشركة").mapping == {"[شخص_1]": "عمار"}


def test_structured_data_untouched():
    text = "عمار 0791234567 السرحان"
    r = fam().mask(text)
    assert r.text == "[شخص_1] [هاتف_1] السرحان"


def test_expansion_never_unmasks_anything():
    """On all 1,663 golden inputs, expansion only adds masked characters (never removes any)."""
    off, on = PIIMasker(), fam()
    for line in GOLDEN.read_text(encoding="utf-8").splitlines():
        text = json.loads(line)["text"]
        cov_off = {i for s, e, _ in off.spans(text) for i in range(s, e)}
        cov_on = {i for s, e, _ in on.spans(text) for i in range(s, e)}
        assert cov_off <= cov_on, text
        restored = on.mask(text)
        assert PIIMasker.restore(restored.text, restored.mapping) == text
