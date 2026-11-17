"""Diagnose why a name at the end of a long text can be missed: windowing bug, or model context effect?

    python scripts/diagnose_long_text.py

Needs the real model (requirements-ml.txt + scripts/download_name_model.py). Synthetic text only.
Prints model outputs; changes nothing and tunes nothing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.pii.name_model import load_name_model  # noqa: E402

TAIL = "وبعدين حكيت مع غيث القيسي عن الراتب"
NAME = "غيث القيسي"
# Synthetic filler with no names, written for this check (not from any evaluation set).
FILLER = [
    "اشتغلت بالشركة خمس سنين وما أخذت ولا إجازة سنوية.",
    "العقد كان لمدة سنة وتجدد تلقائيا مرتين.",
    "صاحب العمل بطلب مني أشتغل يوم الجمعة بدون أجر إضافي.",
    "الراتب كان يتأخر كل شهر أسبوعين تقريبا.",
    "قدمت شكوى لوزارة العمل وما حد رد علي لهلأ.",
    "بدي أعرف شو حقوقي إذا تركت الشغل قبل نهاية العقد.",
]


def found(model, text):
    s = text.rindex(NAME)
    raw = [(a, b, sc) for a, b, sc in model.raw_spans(text) if a < s + len(NAME) and s < b]
    covered = {i for a, b in model.detect_names(text) for i in range(a, b)}
    hit = all(i in covered for i in range(s, s + len(NAME)) if not text[i].isspace())
    best = max((sc for _, _, sc in raw), default=None)
    return hit, best


def main():
    st = Settings.from_env()
    model = load_name_model("camelbert-da", model_id=st.pii_name_model_id, min_score=0.0,
                            revision=st.pii_name_model_revision or None)

    print("KNOWN LIMITATION (sections 1-2): unnatural text repeating one word hundreds of times can hide a name")
    print("from the model even inside one short call. Real questions do not look like this; it is documented,")
    print("not used to weaken the tests.\n")
    print("1) CHUNKS on repeated-word text (العقد x 1200 + tail)")
    text = " ".join(["العقد"] * 1200) + " " + TAIL
    wins = model.windows(text)
    a, b = wins[-1]
    print(f"   {len(wins)} chunks; last chunk = chars {a}-{b} of {len(text)}; "
          f"pieces in it: {len(model.piece_offsets(text[a:b]))}; contains the whole tail: {TAIL in text[a:b]}")
    print(f"   name found in full text: {found(model, text)[0]}")
    print(f"   name found when the model runs on the last window text directly: {found(model, text[a:b])}")

    print("\n2) REPEATED-WORD CONTEXT in a single model call (no chunking)")
    for n in (0, 10, 50, 100, 200, 300):
        t = (" ".join(["العقد"] * n) + " " + TAIL).strip()
        hit, best = found(model, t)
        print(f"   {n:>3} x العقد + tail: found={hit!s:5}  best PERS score near name={best}")

    print("\n3) NATURAL FILLER, increasing length (sentence-level chunks)")
    for n in (0, 5, 20, 60, 150):
        t = (" ".join(FILLER[i % len(FILLER)] for i in range(n)) + " " + TAIL).strip()
        hit, best = found(model, t)
        print(f"   {n:>3} sentences ({len(model.windows(t))} chunk(s)): found={hit!s:5}  best score={best}")

    print("\n4) NAME AT START / MIDDLE / END of 150 natural sentences (must all be found)")
    for punctuated in (True, False):
        for where, at in (("start", 0), ("middle", 75), ("end", 150)):
            sents = [FILLER[i % len(FILLER)] for i in range(150)]
            sents.insert(at, TAIL + ".")
            if not punctuated:
                sents = [x.rstrip(".") for x in sents]
            t = " ".join(sents)
            hit, best = found(model, t)
            label = "punctuated" if punctuated else "run-on    "
            print(f"   {label} {where:6s} ({len(model.windows(t))} chunks): found={hit!s:5}  best score={best}")


if __name__ == "__main__":
    main()
