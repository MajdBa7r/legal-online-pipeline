"""Record what the masker produces today, so refactors can prove they change nothing.

    python tests/fixtures/build_masker_golden.py

Writes tests/fixtures/masker_golden.jsonl: {"text", "masked", "mapping"} for every input below.
It was generated with the masker from Step 2 (masker.py sha256 c37eda56…ac23d9), BEFORE Phase 3A.
Do NOT regenerate it to make a failing test pass: a mismatch means masking behaviour changed.
Regenerate only when a behaviour change has been deliberately approved, and say so in the commit.

Inputs (all synthetic, seed 2026):
* every sentence of the PII benchmark development set (eval/pii_cases.jsonl);
* every article of the invented fixture laws (tests/fixtures/local_laws);
* 1,500 fuzz sentences mixing trigger words, names, kunyas, phones, IDs, IBANs, emails, plain numbers,
  Arabic-Indic digits, punctuation, English words and legal phrases in random orders.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.pii.masker import NAME_STOP, STRONG_NAME_TRIGGERS, WEAK_NAME_TRIGGERS, PIIMasker  # noqa: E402

R = random.Random(2026)
OUT = Path(__file__).with_name("masker_golden.jsonl")
INDIC = str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩")


def digits(n):
    return "".join(R.choice("0123456789") for _ in range(n))


def fragment(masker: PIIMasker) -> str:
    names = sorted(masker.names)
    other_names = ["ثامر", "قصي", "غيث", "براء", "حلا", "جنى", "الحياري", "النعيمات", "بني ياسين"]
    kind = R.randrange(24)
    if kind == 0:
        return R.choice(sorted(STRONG_NAME_TRIGGERS))
    if kind == 1:
        return R.choice(sorted(WEAK_NAME_TRIGGERS))
    if kind in (2, 3):
        return R.choice(names)
    if kind == 4:
        return R.choice(other_names)
    if kind == 5:
        return R.choice(sorted(NAME_STOP))
    if kind == 6:
        d = "7" + R.choice("789") + digits(7)
        return R.choice(["0" + d, f"0{d[:2]} {d[2:5]} {d[5:]}", f"+962 {d}", "00962" + d, f"0{d[:2]}-{d[2:5]}-{d[5:]}",
                         ("0" + d).translate(INDIC), "06" + digits(7)])
    if kind == 7:
        v = R.choice("29") + digits(9)
        return R.choice([v, v.translate(INDIC), v + digits(1), v[:9], f"{v[:3]}.{v[3:]}", f"{v},"])
    if kind == 8:
        raw = "JO" + digits(2) + R.choice(["CBJO", "ARAB", "HBHO"]) + digits(22)
        return R.choice([raw, raw.lower(), " ".join(raw[i:i + 4] for i in range(0, 30, 4)), raw[:-1]])
    if kind == 9:
        return R.choice(["a.b@gmail.com", "x_y99@yahoo.com", "hr@company.jo", "user@mail", "@handle", "test@@x.com"])
    if kind == 10:
        return R.choice([digits(R.randint(1, 12)), f"{digits(3)},{digits(3)}", f"{digits(2)}.{digits(2)}",
                         f"{digits(2)}/{digits(2)}/{digits(4)}", f"{digits(4)}/{digits(4)}"])
    if kind == 11:
        return R.choice(["،", ",", ".", "؟", "?", "!", ":", "؛", "(", ")", "-", "\n", "  "])
    if kind == 12:
        return R.choice(["salary", "WhatsApp", "ok", "Mr", "ID", "IBAN:", "email"])
    if kind == 13:
        return R.choice(["المادة 23 من قانون العمل", "قانون الضمان الاجتماعي رقم 1 لسنة 2014", "صاحب العمل",
                         "المستأجر", "المالك", "إجازة سنوية", "تعويض", "عقد الإيجار", "بدل التعطل"])
    if kind == 14:
        return "أبو " + R.choice(names + other_names)
    if kind == 15:
        return "أم " + R.choice(names + other_names)
    if kind == 16:
        return R.choice(sorted(STRONG_NAME_TRIGGERS)) + " " + R.choice(names + other_names) + " " + R.choice(
            names + other_names)
    return R.choice(["بدي", "أعرف", "شو", "حقي", "الشركة", "فصلني", "من", "سنة", "شهر", "دينار", "و", "في", "على",
                     "عندي", "مشكلة", "هل", "يحق", "لي", "الراتب", "تبعي", "اليوم", "مع"])


def main():
    masker = PIIMasker()
    texts: list[str] = []
    with open(ROOT / "eval" / "pii_cases.jsonl", encoding="utf-8") as f:
        texts += [json.loads(line)["text"] for line in f if line.strip()]
    for p in sorted((ROOT / "tests" / "fixtures" / "local_laws").glob("*.json")):
        texts += [a["text_ar"] for a in json.loads(p.read_text(encoding="utf-8"))["articles"]]
    for _ in range(1500):
        parts = [fragment(masker) for _ in range(R.randint(2, 12))]
        texts.append(" ".join(parts) if R.random() < 0.85 else "".join(parts))
    with open(OUT, "w", encoding="utf-8") as f:
        for t in texts:
            r = masker.mask(t)
            f.write(json.dumps({"text": t, "masked": r.text, "mapping": r.mapping}, ensure_ascii=False) + "\n")
    changed = sum(1 for t in texts if masker.mask(t).mapping)
    print(f"{len(texts)} inputs ({changed} with something masked) -> {OUT}")


if __name__ == "__main__":
    main()
