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


# ---- ordinal numbers written as words (MVP step 1 fix) --------------------------------------------------
import pytest as _pytest  # noqa: E402

from app.verify_checks.numbers import extract_numbers as _x, numbers_supported as _ns  # noqa: E402


@_pytest.mark.parametrize("text,value", [
    ("سن الخامسة والخمسين", 55), ("سن الخامسة والأربعين", 45), ("الحادي عشر", 11),
    ("السادسة عشرة", 16), ("الثانية والستين", 62), ("العاشرة", 10),
])
def test_ordinal_words(text, value):
    assert value in _x(text)


def test_ordinal_age_supports_a_digit_claim():
    article = "ان يكمل المؤمن عليه الذكر سن الخامسة والخمسين من عمره وان تبلغ اشتراكاته (252) اشتراكاً"
    assert _ns("فيشترط للذكر إكمال 55 سنة وبلوغ 252 اشتراكا فعليا", article) == (True, [])
    assert _ns("فيشترط للذكر إكمال 56 سنة", article) == (False, [56.0])     # a wrong age is still caught


def test_existing_word_numbers_unchanged():
    assert _x("أربعة عشر يوماً") == {14.0} and _x("خمسة وعشرين بالمئة") == {25.0} and _x("الأول") == set()


# ---- source-text defect: dual glued to the previous word (Labour Law Art. 61) ---------------------------
def test_dual_glued_to_previous_word_in_the_article_supports_the_claim():
    article = "وفي حال رفض منحه الإجازة وانقضتالسنتان يعاقب بغرامة مقدارها مائة دينار"
    assert _ns("وإذا رفض منحك إجازتك وانقضت سنتان فعليه غرامة مقدارها مائة دينار", article) == (True, [])


def test_glued_fallback_is_narrow():
    article = "وفي حال رفض منحه الإجازة وانقضتالسنتان يعاقب بغرامة"
    assert _ns("وانقضت ثلاث سنوات", article) == (False, [3.0])          # other numbers still need support
    assert 2.0 not in _x("وانقضتالسنتان")                                   # claims never use the fallback
    assert _ns("خلال يومين", "خلال اليومين التاليين") == (True, [])          # normal duals work as before
