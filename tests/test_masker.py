from app.pii.masker import PIIMasker

m = PIIMasker()


def test_phone_email_iban_id():
    text = "رقمي 0791234567 وايميلي ali.k@example.com والآيبان JO94 CBJO 0010 0000 0000 0131 0003 02 ورقمي الوطني 9901234567"
    r = m.mask(text)
    assert "0791234567" not in r.text
    assert "ali.k@example.com" not in r.text
    assert "JO94" not in r.text
    assert "9901234567" not in r.text
    assert "[هاتف_1]" in r.text and "[بريد_1]" in r.text and "[آيبان_1]" in r.text and "[رقم_وطني_1]" in r.text


def test_international_phone_and_arabic_digits():
    r = m.mask("اتصل على +962 79 123 4567 أو ٠٧٨١٢٣٤٥٦٧")
    assert r.text.count("[هاتف_") == 2


def test_names_with_trigger_and_list():
    r = m.mask("اسمي خالد الزعبي وأشتغل عند أبو محمد من سنتين")
    assert "خالد" not in r.text and "الزعبي" not in r.text
    assert "محمد" not in r.text
    assert "[شخص_1]" in r.text


def test_same_value_same_placeholder_and_restore():
    r = m.mask("محمد قال إنه سيدفع، لكن محمد لم يدفع")
    assert r.text.count("[شخص_1]") == 2
    assert PIIMasker.restore(r.text, r.mapping) == "محمد قال إنه سيدفع، لكن محمد لم يدفع"


def test_amounts_and_dates_not_masked():
    text = "راتبي 450 دينار وبدأت العمل سنة 2021 ومدة العقد 12 شهر"
    assert m.mask(text).text == text


def test_ordinary_words_not_masked():
    text = "أنا موظف في شركة وصاحب العمل رفض يعطيني إجازة"
    assert m.mask(text).text == text


def test_leak_check():
    r = m.mask("اسمي خالد")
    assert PIIMasker.leaked(["سؤال عن خالد"], r.originals) == ["خالد"]
    assert PIIMasker.leaked([r.text], r.originals) == []
