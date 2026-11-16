"""The PII benchmark harness and detectors. masker.py itself is not changed by Step 2."""
import json
from pathlib import Path

from app.pii.detectors import HybridDetector, ModelDetector, RuleDetector, Span
from app.pii.masker import PIIMasker
from app.pii.pii_eval import evaluate, load_cases, summarize

CASES = Path(__file__).resolve().parent.parent / "eval" / "pii_cases.jsonl"


def test_dataset_is_consistent():
    cases = load_cases(CASES)
    assert len(cases) == 150
    assert {c.group for c in cases} == {"core", "extra", "negative"}
    for c in cases:
        assert (c.group == "negative") == (not c.gold), c.id
        for g in c.gold:
            assert c.text[g.start:g.end] == g.value
            assert g.type in ("name", "phone", "national_id", "iban", "email", "address", "plate")
            assert (g.type in ("address", "plate")) == (c.group == "extra")


def test_rule_detector_is_exactly_the_production_masker():
    """Same spans as PIIMasker.mask(): rebuilding the masked text from the detector reproduces mask()."""
    m = PIIMasker()
    det = RuleDetector(m)
    for c in load_cases(CASES):
        spans = sorted(det.detect(c.text), key=lambda s: s.start)
        masked = m.mask(c.text)
        rebuilt, last, seen = [], 0, {}
        counters = {}
        from app.pii.masker import LABELS
        for s in spans:
            key = (s.type, c.text[s.start:s.end].strip())
            if key not in seen:
                counters[s.type] = counters.get(s.type, 0) + 1
                seen[key] = f"[{LABELS[s.type]}_{counters[s.type]}]"
            rebuilt += [c.text[last:s.start], seen[key]]
            last = s.end
        rebuilt.append(c.text[last:])
        assert "".join(rebuilt) == masked.text, c.id


def test_coverage_full_partial_missed_and_false_alarms(tmp_path):
    class Fixed:
        name = "fixed"

        def __init__(self, spans):
            self.spans = spans

        def detect(self, text):
            return self.spans

    case_text = "اسمي خالد الزعبي ورقمي 0791234567"
    path = tmp_path / "case.jsonl"
    path.write_text(json.dumps({"id": "t1", "text": case_text, "group": "core", "tags": [],
                                "pii": [{"type": "name", "value": "خالد الزعبي"},
                                        {"type": "phone", "value": "0791234567"}]}, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    if True:
        cases = load_cases(path)
        name_start = case_text.index("خالد")
        phone_start = case_text.index("079")
        det = Fixed([Span(name_start, name_start + 4, "name"),                 # only the first name
                     Span(phone_start, phone_start + 10, "national_id"),      # right value, other type
                     Span(0, 4, "name")])                                     # "اسمي": a false alarm
        r = evaluate(det, cases)
        status = {g["type"]: g["status"] for g in r["gold"]}
        assert status == {"name": "partial", "phone": "full"}
        assert [f["span"] for f in r["false_alarms"]] == ["اسمي"]
        s = summarize(r)
        assert s["core"]["sentence_leak_rate"] == 1.0
        assert s["type_confusions"] == 1


def test_model_detector_accepts_offsets_strings_and_bio_labels():
    text = "اتصلت مع غيث القيسي امبارح"
    preds = [{"entity_group": "PERS", "start": 9, "end": 19},
             {"label": "person", "text": "غيث"},
             ("B-PERS", "القيسي"),
             {"entity_group": "LOC", "start": 0, "end": 4}]
    det = ModelDetector("fake", lambda t: preds, {"pers": "name", "person": "name"})
    spans = det.detect(text)
    assert {(s.start, s.end, s.type) for s in spans} == {(9, 19, "name"), (9, 12, "name"), (13, 19, "name")}


def test_hybrid_adds_only_selected_model_types():
    text = "مديري براء الفلاحات رقمه 0791234567"
    model = ModelDetector("fake", lambda t: [{"label": "person", "start": 6, "end": 19},
                                             {"label": "phone", "start": 25, "end": 35}],
                          {"person": "name", "phone": "phone"})
    hyb = HybridDetector(RuleDetector(), model, model_types=("name",))
    spans = hyb.detect(text)
    assert any(s.type == "name" and s.source == "fake" for s in spans)
    assert not any(s.type == "phone" and s.source == "fake" for s in spans)
    assert any(s.type == "phone" and s.source == "rules" for s in spans)


def test_rules_baseline_regression_guard():
    """Measured 2026-11-17 on the synthetic set. Structured types must stay perfect; names are the known gap."""
    s = summarize(evaluate(RuleDetector(), load_cases(CASES)))
    for t in ("phone", "national_id", "iban", "email"):
        assert s["core"]["by_type"][t]["recall"] == 1.0, t
    assert s["core"]["by_type"]["name"]["recall"] >= 0.6
    assert s["false_alarms"]["negative_sentences_with_false_alarm"] <= 3
