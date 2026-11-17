"""The PII check of scripts/run_e2e.py (it reported a false leak for "براء" inside law words)."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("run_e2e", Path(__file__).parent.parent / "scripts" / "run_e2e.py")
run_e2e = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_e2e)


def calls(*contents):
    return [{"messages": [{"role": "user", "content": c} for c in contents]}]


def test_law_words_containing_the_letters_are_not_a_leak():
    law = '<article id="x">تعيين خبراء تأمين بعد الحصول على شهادة براءة ذمة</article>'
    assert run_e2e.pii_leaks(calls(law)) == []
    assert run_e2e.pii_leaks(calls("سؤال عن خبراء الضمان")) == []          # inside a word, outside articles


def test_a_real_leak_is_still_found():
    assert run_e2e.pii_leaks(calls("اسمي براء الفلاحات")) == ["براء الفلاحات", "براء", "الفلاحات"]
    assert run_e2e.pii_leaks(calls("رقمي 0791234567")) == ["0791234567"]
    assert run_e2e.pii_leaks(calls('<article id="x">نص</article> اسمي براء')) == ["براء"]
