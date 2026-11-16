"""Generate eval/pii_cases.jsonl: synthetic, labelled sentences for the PII benchmark.

    python eval/build_pii_cases.py        # rewrites eval/pii_cases.jsonl (deterministic, seed 17)

ALL DATA IS SYNTHETIC. Names are drawn from common-name pools, numbers are random but
correctly formatted (IBANs even carry a valid check digit). No real person is described.

Each line: {"id", "text", "group", "tags", "pii": [{"type", "value"}]}
  group "core"     -> name, phone, national_id, iban, email (the five blueprint categories)
  group "extra"    -> address, plate (benchmarked separately; NOT in the masking scope yet)
  group "negative" -> no personal data at all (catches false alarms)
Spans are computed by the harness from the values, so there are no hand-typed offsets.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

R = random.Random(17)
OUT = Path(__file__).with_name("pii_cases.jsonl")
INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")

# First names: some are in app/pii/common_names.txt, some deliberately are not.
FIRST_IN_LIST = ["محمد", "أحمد", "خالد", "يوسف", "رامي", "فادي", "سامر", "ليث", "معاذ", "يزن", "ريم", "لينا", "روان",
                 "مريم", "سارة", "هشام", "عمار", "ماجد", "زيد", "فاطمة"]
FIRST_NOT_IN_LIST = ["ثامر", "مجد", "قصي", "غيث", "براء", "ناصر", "حلا", "جنى", "هيا", "ضحى", "عدي", "شادي", "تالا",
                     "بشار", "إياد", "رزان", "سند", "مؤيد"]
FAMILY_IN_LIST = ["الزعبي", "العبادي", "الخطيب", "المومني", "الطراونة", "المجالي", "الشرمان", "العتوم"]
FAMILY_NOT_IN_LIST = ["الحياري", "النعيمات", "الجراح", "العواملة", "الخصاونة", "الدعجة", "السرحان", "القيسي",
                      "الفلاحات", "بني ياسين"]


def first(in_list: bool | None = None) -> str:
    if in_list is None:
        in_list = R.random() < 0.5
    return R.choice(FIRST_IN_LIST if in_list else FIRST_NOT_IN_LIST)


def family(in_list: bool | None = None) -> str:
    if in_list is None:
        in_list = R.random() < 0.4
    return R.choice(FAMILY_IN_LIST if in_list else FAMILY_NOT_IN_LIST)


def phone(fmt: str) -> str:
    d = "7" + R.choice("789") + "".join(R.choice("0123456789") for _ in range(7))
    return {
        "local": "0" + d,
        "spaced": f"0{d[:2]} {d[2:5]} {d[5:]}",
        "dashed": f"0{d[:2]}-{d[2:5]}-{d[5:]}",
        "intl": f"+962 {d[:2]} {d[2:5]} {d[5:]}",
        "intl_compact": "+962" + d,
        "00": "00962" + d,
        "indic": ("0" + d).translate(INDIC),
    }[fmt]


def national_id(indic: bool = False) -> str:
    v = R.choice("92") + "".join(R.choice("0123456789") for _ in range(9))
    return v.translate(INDIC) if indic else v


def iban(spaced: bool) -> str:
    bank = R.choice(["CBJO", "ARAB", "HBHO", "JONB", "UBSI", "ETHI"])
    acct = "".join(R.choice("0123456789") for _ in range(22))
    # ISO 13616 check digits
    rearranged = bank + acct + "JO00"
    num = "".join(str(int(c, 36)) for c in rearranged)
    check = 98 - int(num) % 97
    raw = f"JO{check:02d}{bank}{acct}"
    return " ".join(raw[i:i + 4] for i in range(0, len(raw), 4)) if spaced else raw


def email() -> str:
    user = R.choice(["ahmad.k", "m.zoubi", "rawan_99", "yazan2001", "lina.h", "sami.jo", "tala_m", "omar.ali88",
                     "hr.manager", "abu.zaid"])
    return f"{user}@{R.choice(['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.jo'])}"


ADDRESSES = [
    "عمان، جبل الحسين، شارع خالد بن الوليد، عمارة 12",
    "اربد - حي التركمان - بناية رقم 7 الطابق الثاني",
    "الزرقاء، حي معصوم، شارع 36، بيت رقم 14",
    "خلدا، شارع وصفي التل، مجمع النور، شقة 4",
    "طبربور، اسكان الأطباء، عمارة 22 شقة 6",
    "السلط، حي الجدعة، قرب مسجد الأنصار، الطابق الأرضي",
    "ماركا الشمالية شارع الملك عبدالله الأول عمارة 5",
    "العقبة، المنطقة العاشرة، فيلا رقم 31",
    "مادبا، حي الكرك، شارع 15، الطابق الثالث",
    "جرش، سوف، بجانب المدرسة الثانوية، بيت أبو سند",
    "عمان - الصويفية - شارع باريس - عمارة 9 - شقة 11",
    "المفرق، حي الحسين، شارع 20 بيت رقم 3",
]


def plate(indic: bool = False) -> str:
    v = f"{R.choice(['10', '11', '12', '13', '14', '17', '20', '21', '22', '50'])}-{R.randint(1000, 99999)}"
    return v.translate(INDIC) if indic else v


cases: list[dict] = []


def add(text: str, group: str, tags: list[str], pii: list[tuple[str, str]]):
    cases.append({"id": f"{group[:3]}-{len(cases) + 1:03d}", "text": text, "group": group, "tags": tags,
                  "pii": [{"type": t, "value": v} for t, v in pii]})


# ----------------------------------------------------------------------------- names
NAME_TEMPLATES = [
    ("اسمي {n} وبشتغل بشركة من خمس سنين، صاحب الشغل فصلني بدون إشعار، شو حقوقي؟", "dialect", "strong_trigger", "full"),
    ("اسمي {n}، المالك بدو يطلعني من الشقة قبل ما يخلص العقد", "dialect", "strong_trigger", "first"),
    ("أنا {n} عمري 30 سنة وبدي أعرف كم يوم إجازة سنوية إلي", "dialect", "weak_trigger", "first"),
    ("مديري {n} رفض يعطيني إجازة مرضية مع إنه معي تقرير", "dialect", "weak_trigger", "full"),
    ("صاحب البيت أبو {n} قطع عني المي عشان أطلع", "dialect", "kunya", "first"),
    ("أم {n} جارتنا أجرتني الشقة بدون عقد مكتوب، هل الإيجار صحيح؟", "dialect", "kunya", "first"),
    ("المدعو {n} استلم مني 300 دينار عربون وما رجعهم", "msa", "strong_trigger", "full"),
    ("زوجتي {n} حامل بالشهر السابع، قديش إجازة الأمومة الها؟", "dialect", "weak_trigger", "first"),
    ("يا جماعة {n} صاحب المحل باعني تلفون خربان ورافض يرجعه", "dialect", "no_trigger", "full"),
    ("تواصلت مع {n} من الموارد البشرية وقالي ما إلك تعويض", "dialect", "no_trigger", "first"),
    ("أعمل لدى السيد {n} منذ عام 2019 ولم يدفع أجري منذ ثلاثة أشهر", "msa", "strong_trigger", "full"),
    ("قام صاحب العمل {n} بإنهاء عقدي دون سابق إنذار", "msa", "no_trigger", "full"),
    ("ابني {n} عمره 17 سنة، بقدر يشتغل بالعطلة الصيفية؟", "dialect", "weak_trigger", "first"),
    ("الدكتور {n} كتبلي تقرير طبي بس الشركة رفضته", "dialect", "strong_trigger", "full"),
    ("أخوي {n} اشتغل بمطعم شهرين وما دفعوله ولا قرش", "dialect", "weak_trigger", "first"),
    ("بنتي {n} اشترت لابتوب أونلاين وطلع معيوب", "dialect", "weak_trigger", "first"),
    ("كلمت الحاج {n} صاحب العمارة عن الصيانة وما رد", "dialect", "strong_trigger", "first"),
    ("اسمها {n} وهي المستأجرة عندي من 2015 وما دفعت آخر ست شهور", "dialect", "strong_trigger", "full"),
    ("المستأجر {n} غير قفل الباب وما خلاني أدخل شقتي", "dialect", "weak_trigger", "first"),
    ("نرجو إفادتنا بخصوص الموظف {n} الذي تغيب عن العمل عشرة أيام متتالية", "msa", "no_trigger", "full"),
    ("{n} هو المدير المباشر تبعي وبطلب مني أشتغل 12 ساعة", "dialect", "no_trigger", "first"),
    ("بعتت رسالة للمحامي {n} وما رد علي لهلأ", "dialect", "no_trigger", "full"),
    ("السيدة {n} مالكة الشقة رفعت الأجرة مرتين بنفس السنة", "msa", "strong_trigger", "full"),
    ("أبو {n} صاحب المصنع بقول إنه ما في ضمان للعمال المياومة", "dialect", "kunya", "first"),
]
for i, (tpl, lang, trig, kind) in enumerate(NAME_TEMPLATES):
    in_list = i % 2 == 0  # alternate: half the names are in the masker's list, half are not
    f = first(in_list)
    n = f if kind == "first" else f"{f} {family()}"
    tags = [lang, trig, "name_in_list" if in_list else "name_not_in_list"]
    if kind == "full":
        tags.append("full_name")
    add(tpl.format(n=n), "core", tags, [("name", n)])

# ----------------------------------------------------------------------------- phones
PHONE_TEMPLATES = [
    ("رقمي {p} اتصلوا علي إذا في معلومة", "dialect"),
    ("هاد رقم المالك {p} ما بيرد من أسبوع", "dialect"),
    ("إذا بدكم تتواصلوا معي على الواتس {p}", "dialect"),
    ("يمكن التواصل معي على الرقم {p} لاستكمال البيانات", "msa"),
    ("صاحب المحل أعطاني رقمه {p} وقال بس يتصلوا عليه", "dialect"),
    ("الرجاء الاتصال بي على {p} بخصوص الشكوى", "msa"),
    ("موبايلي {p} وما وصلني ولا إشعار من الضمان", "dialect"),
    ("{p} هاد رقم الموارد البشرية بالشركة", "dialect"),
    ("رقم المؤجر {p} وبرفض يستلم الأجرة", "dialect"),
    ("اتصلت على {p} رقم خدمة العملاء وسكروا بوجهي", "dialect"),
    ("هاتفي المحمول {p}", "msa"),
    ("بعتولي رسالة من الرقم {p} إنه انفصلت", "dialect"),
    ("رقم المدير {p} لو بدكم تتأكدوا", "dialect"),
    ("تلفوني {p} وتلفون أخوي إذا ما رديت", "dialect"),
]
fmts = ["local", "spaced", "dashed", "intl", "intl_compact", "00", "indic"]
for i, (tpl, lang) in enumerate(PHONE_TEMPLATES):
    fmt = fmts[i % len(fmts)]
    p = phone(fmt)
    add(tpl.format(p=p), "core", [lang, f"phone_{fmt}"], [("phone", p)])

# ----------------------------------------------------------------------------- national IDs
NID_TEMPLATES = [
    ("رقمي الوطني {d} وبدي أعرف كم اشتراك إلي بالضمان", "dialect", False),
    ("الرقم الوطني تبعي {d}", "dialect", False),
    ("الرقم الوطني للمؤمن عليه {d} وقد تقدم بطلب تقاعد مبكر", "msa", False),
    ("{d} هاد رقمي الوطني إذا بدكم تفحصوا", "dialect", False),
    ("سجلوني بالضمان على رقم وطني {d} مش إلي", "dialect", False),
    ("رقم الهوية {d} والعامل مسجل منذ 2018", "msa", False),
    ("رقمي الوطني {d}", "dialect", True),
    ("الرقم الوطني {d} للمستأجر", "msa", True),
    ("بطاقتي رقمها {d} وانسرقت مع المحفظة", "dialect", False),
    ("أرجو التحقق من الرقم {d} في سجلات المؤسسة", "msa", False),
]
for tpl, lang, indic in NID_TEMPLATES:
    d = national_id(indic)
    add(tpl.format(d=d), "core", [lang] + (["indic_digits"] if indic else []), [("national_id", d)])

# ----------------------------------------------------------------------------- IBANs
IBAN_TEMPLATES = [
    ("حولولي الراتب على الآيبان {i} وما وصل", "dialect", True),
    ("رقم الآيبان {i}", "dialect", False),
    ("أرجو تحويل المستحقات إلى الحساب {i}", "msa", True),
    ("هاد الآيبان تبع المالك {i} بحولله الأجرة كل شهر", "dialect", False),
    ("حسابي البنكي {i} والشركة بتقول إنها حولت", "dialect", True),
    ("IBAN: {i}", "msa", False),
    ("الشركة خصمت من راتبي اللي بنزل على {i}", "dialect", True),
    ("رقم حسابي الدولي {i} لصرف بدل التعطل", "msa", False),
]
for tpl, lang, spaced in IBAN_TEMPLATES:
    i_ = iban(spaced)
    add(tpl.format(i=i_), "core", [lang, "iban_spaced" if spaced else "iban_compact"], [("iban", i_)])

# ----------------------------------------------------------------------------- emails
EMAIL_TEMPLATES = [
    ("ايميلي {e} ابعتولي الرد عليه", "dialect"),
    ("بعتولي على {e} إنه تم إنهاء خدمتي", "dialect"),
    ("يرجى المراسلة على البريد الإلكتروني {e}", "msa"),
    ("هاد ايميل الموارد البشرية {e} وما بردوا", "dialect"),
    ("{e}", "msa"),
    ("بعتت شكوى من {e} لحماية المستهلك", "dialect"),
    ("البريد الخاص بي هو {e}", "msa"),
    ("المالك بعتلي إنذار على الايميل {e}", "dialect"),
]
for tpl, lang in EMAIL_TEMPLATES:
    e = email()
    add(tpl.format(e=e), "core", [lang], [("email", e)])

# ----------------------------------------------------------------------------- mixed
for _ in range(8):
    n = f"{first()} {family()}"
    p, d = phone(R.choice(fmts[:4])), national_id()
    tpl = R.choice([
        ("اسمي {n}، رقمي {p}، والرقم الوطني {d}، بدي أعرف شو حقوقي بعد ما انفصلت", "dialect"),
        ("أنا {n} ورقمي الوطني {d} وتلفوني {p}، المالك بدو يطلعني", "dialect"),
        ("مقدم الطلب: {n}، الرقم الوطني: {d}، الهاتف: {p}", "msa"),
    ])
    add(tpl[0].format(n=n, p=p, d=d), "core", [tpl[1], "mixed"], [("name", n), ("phone", p), ("national_id", d)])
for _ in range(4):
    n, e, i_ = f"{first()} {family()}", email(), iban(True)
    add(f"اسمي {n}، ايميلي {e}، وحسابي {i_} وما نزل الراتب", "core", ["dialect", "mixed"],
        [("name", n), ("email", e), ("iban", i_)])

# ----------------------------------------------------------------------------- extra: addresses and plates
ADDR_TEMPLATES = [
    ("ساكن في {a} والمالك بدو يطلعني", "dialect"),
    ("عنواني {a}", "dialect"),
    ("المأجور يقع في {a} وقد انتهت مدة العقد", "msa"),
    ("المحل اللي اشتريت منه موقعه {a}", "dialect"),
    ("عنوان السكن: {a}", "msa"),
    ("الشقة اللي مستأجرها ب{a} فيها رطوبة والمالك ما بدو يصلح", "dialect"),
]
for i, a in enumerate(ADDRESSES):
    tpl, lang = ADDR_TEMPLATES[i % len(ADDR_TEMPLATES)]
    add(tpl.format(a=a), "extra", [lang], [("address", a)])
PLATE_TEMPLATES = [
    ("نمرة سيارتي {v} وصاحب الشغل حجزها", "dialect", False),
    ("رقم اللوحة {v} والسيارة مسجلة باسم الشركة", "msa", False),
    ("السيارة {v} صدمتني وأنا رايح عالشغل، هل بتعتبر إصابة عمل؟", "dialect", False),
    ("سيارة المالك نمرتها {v} صافة قدام باب المحل", "dialect", True),
    ("المركبة رقم {v} مملوكة للمؤسسة", "msa", False),
    ("شفت السيارة {v} بتكسر الزجاج", "dialect", True),
    ("بعت السيارة {v} وما دفعلي الشاري", "dialect", False),
    ("لوحة رقم {v}", "msa", False),
    ("الباص {v} اللي بنقل العمال عمل حادث", "dialect", False),
    ("نمرة الفان تبع الشركة {v}", "dialect", True),
    ("المخالفة كانت على السيارة {v}", "msa", False),
    ("سيارة توصيل رقمها {v} وصلت الطلب خربان", "dialect", False),
]
for tpl, lang, indic in PLATE_TEMPLATES:
    v = plate(indic)
    add(tpl.format(v=v), "extra", [lang] + (["indic_digits"] if indic else []), [("plate", v)])

# ----------------------------------------------------------------------------- negatives (no personal data)
NEGATIVES = [
    ("راتبي 450 دينار وبدأت الشغل سنة 2021 ومدة العقد 12 شهر", ["numbers"]),
    ("حسب المادة 23 من قانون العمل رقم 8 لسنة 1996 لازم إشعار شهر", ["law_numbers"]),
    ("رقم القضية 1234/2023 وما صار في جلسة لهلأ", ["case_number"]),
    ("دفعت 1500 دينار تأمين للشقة وعقد الإيجار من 1/1/2020 لمدة 3 سنوات", ["numbers", "dates"]),
    ("أمل تبعي إني أرجع حقي من الشركة", ["name_like_word"]),
    ("نور الشقة مقطوع من يومين والمالك مش سائل", ["name_like_word"]),
    ("سلام عليكم، سؤالي عن الإجازات المرضية", ["name_like_word"]),
    ("شركة الاتصالات خصمت من راتبي بدون سبب", ["company"]),
    ("اشتغلت 10 سنين وعمري 45 سنة، بقدر أطلع تقاعد مبكر؟", ["numbers"]),
    ("رقم الطلب 2024112233 ولسا ما ردوا علي", ["hard_negative_10_digits"]),
    ("اشتريت تلفون ب 350 دينار وطلع فيه عيب مصنعي", ["numbers"]),
    ("الإجازة السنوية 14 يوم وبعد 5 سنين بتصير 21 يوم", ["numbers"]),
    ("المالك رفع الأجرة من 250 إلى 300 دينار", ["numbers"]),
    ("كم نسبة الاقتطاع من الراتب للضمان الاجتماعي؟", []),
    ("هل بحقلي تعويض نهاية خدمة إذا استقلت؟", []),
    ("الشركة بتشغلنا 10 ساعات باليوم بدون إضافي", ["numbers"]),
    ("شو الفرق بين العقد محدد المدة وغير محدد المدة؟", []),
    ("بدي أرجع غسالة اشتريتها قبل أسبوع وطلعت خربانة", []),
    ("عقد الإيجار انتهى بتاريخ 31/8/2023 والمستأجر رافض يطلع", ["dates"]),
    ("المادة 28 من قانون العمل بتسمح بالفصل بدون إشعار بحالات معينة", ["law_numbers"]),
    ("حماية المستهلك بتقبل شكاوى أونلاين؟", []),
    ("أبوي بشتغل بالقطاع الخاص من 30 سنة وبدو يتقاعد", ["name_like_word"]),
    ("ساكن بعمان وبشتغل بالزرقاء، المواصلات على حسابي؟", ["place_names"]),
    ("شارع الملك عبدالله الثاني فيه محلات كثير بتبيع بدون فواتير", ["hard_negative_public_name"]),
    ("جامعة الحسين بن طلال بتعطي إجازة أمومة أطول؟", ["hard_negative_public_name"]),
    ("مستشفى الأمير حمزة رفض يعطيني تقرير إصابة عمل", ["hard_negative_public_name"]),
    ("قانون الضمان الاجتماعي رقم 1 لسنة 2014 هل لسا ساري؟", ["law_numbers"]),
    ("دفعت فاتورة 89.50 دينار وما أعطوني إيصال", ["numbers"]),
    ("كم مدة الإشعار إذا بدي أترك الشغل؟", []),
    ("صاحب العمل بدو يخصم 3 أيام من راتبي لأني تأخرت", ["numbers"]),
    ("انصابيت بالشغل يوم 12 آذار وما سجلوها إصابة", ["dates"]),
    ("في 3 عمال بالمحل ما إلهم ضمان", ["numbers"]),
    ("الأجرة السنوية 3600 دينار تدفع على أربع دفعات", ["numbers"]),
    ("المحكمة حكمت لي ب 2000 دينار وما نفذوا الحكم", ["numbers"]),
    ("رقم الشكوى في الوزارة 556677 ولسا ما في رد", ["case_number"]),
    ("أنا موظف بشركة خاصة وما بعطوني إجازة سنوية", ["weak_trigger_no_name"]),
    ("أنا مستأجر من 1995 هل بيطبق علي القانون القديم؟", ["weak_trigger_no_name"]),
    ("زوجتي بتشتغل بمدرسة خاصة وفصلوها وهي حامل", ["weak_trigger_no_name"]),
    ("ابني عمره 16 سنة، مسموح يشتغل؟", ["weak_trigger_no_name"]),
    ("صاحب المحل رفض يرجعلي المصاري", []),
    ("السلعة طلعت مغشوشة والكفالة سنة", []),
    ("هل الضمان بيصرف بدل تعطل إذا انفصلت؟", []),
    ("الراتب الأساسي 600 والعلاوات 150", ["numbers"]),
    ("قطعوا الكهرباء عن المحل لأني تأخرت بالأجرة شهر", []),
    ("وظيفتي محاسب ومديري بطلب أشتغل يوم الجمعة", []),
    ("هل بيجوز للمالك يدخل الشقة بدون إذني؟", []),
    ("وصلني إنذار عدلي بالإخلاء، شو بعمل؟", []),
    ("بيت أبوي مأجر من زمان بأجرة قليلة كثير", ["name_like_word"]),
    ("مكتب العمل في إربد قالولي أرفع قضية", ["place_names"]),
    ("الكفالة مكتوب فيها 24 شهر بس المحل رافض يصلح", ["numbers"]),
]
for text, tags in NEGATIVES:
    add(text, "negative", tags, [])

if __name__ == "__main__":
    for c in cases:
        for p in c["pii"]:
            assert p["value"] in c["text"], (c["id"], p)
    with open(OUT, "w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter

    print(f"{len(cases)} cases -> {OUT}")
    print(Counter(c["group"] for c in cases))
    print(Counter(p["type"] for c in cases for p in c["pii"]))
