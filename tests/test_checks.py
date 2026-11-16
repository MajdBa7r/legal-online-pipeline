from app.config import Settings
from app.schemas.llm import Claim
from app.stages.verify import code_checks
from app.text.arabic import build_fts_query, normalize
from app.verify_checks.numbers import extract_numbers, numbers_supported
from app.verify_checks.quotes import check_quote
from fakes import LEAVE, NOTICE, REPEALED


def test_number_words():
    assert extract_numbers("ثلاثين يوماً") == {30}
    assert extract_numbers("خمسة وعشرين بالمئة") == {25}
    assert extract_numbers("أربعة عشر يوماً") == {14}
    assert extract_numbers("مائة وخمسين ديناراً") == {150}
    assert extract_numbers("ثلاثة آلاف دينار") == {3000}
    assert extract_numbers("مدة شهرين") == {2}
    assert extract_numbers("١٤ يوماً") == {14}
    assert extract_numbers("[شخص_1] عمل") == set()


def test_numbers_supported_digits_vs_words():
    assert numbers_supported("إجازة 14 يوماً", LEAVE.text)[0]
    ok, missing = numbers_supported("إجازة 21 يوماً", LEAVE.text)
    assert not ok and missing == [21]
    # article number in the claim is allowed through metadata
    assert numbers_supported("وفق المادة 61", LEAVE.text, LEAVE.metadata_numbers())[0]


def test_quote_exact_and_normalized():
    r = check_quote("إجازة سنوية بأجر كامل", LEAVE.text)
    assert r.exact
    r = check_quote("اجازة سنوية باجر كامل", LEAVE.text)  # hamza dropped
    assert not r.exact and r.normalized
    assert check_quote("قصير", LEAVE.text).too_short


def test_code_checks():
    s = Settings()
    supplied = {a.id: a for a in (LEAVE, NOTICE, REPEALED)}
    good = Claim(id="c1", text="للعامل إجازة سنوية مدتها 14 يوماً", article_id="L1", quote="إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً")
    assert not code_checks(good, supplied, s).failures
    wrong_art = Claim(id="c2", text="x", article_id="ZZ", quote="إجازة سنوية بأجر كامل")
    assert code_checks(wrong_art, supplied, s).failures == ["not_supplied"]
    bad_quote = Claim(id="c3", text="الإشعار قبل شهر", article_id="L2", quote="قبل ثلاثين يوماً من الإنهاء")
    assert "quote_mismatch" in code_checks(bad_quote, supplied, s).failures
    repealed = Claim(id="c4", text="نص", article_id="R1", quote="نص مادة ملغاة")
    assert "not_in_force" in code_checks(repealed, supplied, s).failures


def test_fts_query_is_or_joined_without_stopwords():
    q = build_fts_query("هل يحق لصاحب العمل إنهاء عقد العمل في فترة التجربة؟")
    assert " or " in q
    assert "هل" not in q.split(" or ")
    assert normalize("إنهاء") in q
