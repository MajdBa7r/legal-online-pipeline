from app.text.arabic import STOPWORDS, build_fts_query, clean_display, light_stem, normalize, search_tokens


def test_normalize_removes_invisible_and_unifies_forms():
    assert normalize("إجازة​ سنويّة") == "اجازه سنويه"
    assert normalize("قــانون حـمـايـة") == "قانون حمايه"
    assert normalize("١٤ يوماً") == "14 يوما"
    assert normalize("مسؤول على") == "مسوول علي"


def test_clean_display_keeps_letters_but_drops_tatweel():
    assert clean_display("قــانون حـمـايـة المستهلك") == "قانون حماية المستهلك"


def test_light_stem_matches_common_legal_variants():
    same = [
        ("الإجازة", "إجازته", "إجازات", "بالإجازة", "باجازة"),
        ("المستأجر", "المستأجرين", "للمستأجر"),
        ("مدة", "مدتها"),
        ("تعويض", "تعويضاً", "بالتعويض"),
        ("أجر", "أجره", "أجرها", "الأجر"),
        ("صاحب", "لصاحب"),
    ]
    for group in same:
        stems = {light_stem(normalize(w)) for w in group}
        assert len(stems) == 1, (group, stems)


def test_light_stem_does_not_eat_short_or_legal_words():
    for w in ("باطل", "بلاغ", "كامل", "بيان", "ضمان", "عقد", "بيع"):
        assert len(light_stem(normalize(w))) >= 3
    assert light_stem(normalize("باطل")) == "باطل"
    assert light_stem(normalize("الضمان")) == "ضمان"


def test_search_tokens_keeps_numbers_drops_stopwords():
    toks = search_tokens("هل يحق للعامل الحصول على إجازة مدتها ١٤ يوماً؟")
    assert "14" in toks and "اجاز" in toks and "عامل" in toks
    assert "هل" not in toks and "علي" not in toks


def test_stopwords_are_in_normalized_form():
    assert "علي" in STOPWORDS and "الي" in STOPWORDS
    assert all(w == normalize(w) for w in STOPWORDS)


def test_build_fts_query_unchanged_contract():
    q = build_fts_query("هل يحق لصاحب العمل إنهاء عقد العمل؟")
    assert q.split(" or ") == ["يحق", "لصاحب", "العمل", "انهاء", "عقد"]
