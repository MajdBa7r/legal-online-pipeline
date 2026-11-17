"""Rules-only vs rules + CAMeLBERT-da (hybrid): one evaluation with every section needed for the decision.

    python scripts/eval_pii_camelbert.py                 # needs requirements-ml.txt + the downloaded model
    python scripts/eval_pii_camelbert.py --min-score 0.0 # default: no threshold (nothing tuned)

Data (all synthetic or public law text; nothing personal):
  1. eval/pii_cases.jsonl - 150 labelled cases (76 core, 24 address/plate, 50 with no personal data).
     This is the DEVELOPMENT set: it was used while building the exclusion list and the harness, so its numbers
     are optimistic. The independent held-out set is Phase 3C.
  2. The 301 law articles in data/dev (LocalStore): they hold no personal data of users, so EVERY name span
     found there is a false positive on legal text (listed for review: a law can mention a real person).
  3. The 1,663 masker regression inputs (tests/fixtures/masker_golden.jsonl): how many outputs the hybrid
     changes, with examples to review.

Report sections: names, phones, national IDs and other structured identifiers, false positives (ordinary
Arabic, role words, sub-word spans, legal text), overall leakage, time and resources. Saved to logs/.
Measures only; changes no configuration.
"""
import argparse
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_DIR, Settings  # noqa: E402
from app.pii.detectors import MaskerDetector  # noqa: E402
from app.pii.masker import PIIMasker  # noqa: E402
from app.pii.pii_eval import evaluate, load_cases, summarize  # noqa: E402

GOLDEN = PROJECT_DIR / "tests" / "fixtures" / "masker_golden.jsonl"
STRUCTURED = ("phone", "national_id", "iban", "email")


def rss_mb():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0.0


def law_texts(settings) -> list[tuple[str, str]]:
    from app.db.local_store import LocalStore
    store = LocalStore(settings.local_data_dir)
    return [(a.id, a.text) for a in store.articles.values()]


def legal_text_fp(masker: PIIMasker, texts) -> dict:
    spans, times = [], []
    for aid, t in texts:
        t0 = time.perf_counter()
        found = masker.spans(t)
        times.append(time.perf_counter() - t0)
        spans += [{"article": aid, "type": ty, "text": t[s:e]} for s, e, ty in found]
    by_type = {}
    for s in spans:
        by_type[s["type"]] = by_type.get(s["type"], 0) + 1
    return {"articles": len(texts), "spans": len(spans), "by_type": by_type,
            "articles_with_name_span": len({s["article"] for s in spans if s["type"] == "name"}),
            "name_examples": [s for s in spans if s["type"] == "name"][:40],
            "ms_per_article_median": 1000 * statistics.median(times) if times else 0.0,
            "ms_per_article_p95": 1000 * pct(times, 0.95)}


def golden_changes(rules: PIIMasker, hybrid: PIIMasker) -> dict:
    rows = [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines()]
    changed = []
    for r in rows:
        a, b = rules.mask(r["text"]), hybrid.mask(r["text"])
        if a.text != b.text:
            changed.append({"text": r["text"], "rules": a.text, "hybrid": b.text})
    return {"inputs": len(rows), "changed": len(changed), "examples": changed[:40]}


def long_text_latency(masker: PIIMasker, n: int = 20) -> dict:
    base = ("اشتغلت بالشركة خمس سنين وما أخذت ولا إجازة سنوية، وصاحب العمل بطلب مني أشتغل يوم الجمعة. "
            "حكيت مع غيث القيسي من الموارد البشرية وما رد علي. ")
    text = (base * (2000 // len(base) + 1))[:2000]
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        masker.mask(text)
        times.append(time.perf_counter() - t0)
    return {"chars": len(text), "ms_median": 1000 * statistics.median(times), "ms_p95": 1000 * pct(times, 0.95)}


def diff_cases(r_rules, r_hybrid) -> dict:
    key = lambda g: (g["case"], g["type"], g["value"])  # noqa: E731
    before = {key(g): g["status"] for g in r_rules["gold"]}
    fixed = [g for g in r_hybrid["gold"] if g["status"] == "full" and before.get(key(g)) != "full"]
    broke = [g for g in r_hybrid["gold"] if g["status"] != "full" and before.get(key(g)) == "full"]
    fa_before = {(f["case"], f["span"]) for f in r_rules["false_alarms"]}
    new_fa = [f for f in r_hybrid["false_alarms"] if (f["case"], f["span"]) not in fa_before]
    return {"fixed_by_hybrid": [{k: g[k] for k in ("case", "type", "value", "text")} for g in fixed],
            "worse_with_hybrid": [{k: g[k] for k in ("case", "type", "value", "status", "text")} for g in broke],
            "new_false_alarms": [{k: f[k] for k in ("case", "type", "span", "text")} for f in new_fa]}


def section(summary: dict, legal: dict) -> dict:
    bt = summary["core"]["by_type"]
    return {
        "names": bt.get("name"), "names_by_tag": summary["core"]["names_by_tag"],
        "phones": bt.get("phone"), "phones_by_format": summary["core"]["phones_by_format"],
        "structured": {t: bt.get(t) for t in STRUCTURED if t != "phone"},
        "false_positives": {
            "negative_sentences_with_false_alarm": f"{summary['false_alarms']['negative_sentences_with_false_alarm']}"
                                                   f"/{summary['false_alarms']['negative_sentences']}",
            "false_alarm_spans_all_cases": summary["false_alarms"]["total_false_alarm_spans"],
            "role_word_false_alarms": summary["role_word_false_alarms"],
            "subword_spans": summary["subword_spans"],
            "legal_text_spans": legal["spans"], "legal_text_by_type": legal["by_type"],
            "legal_articles_with_a_name_span": f"{legal['articles_with_name_span']}/{legal['articles']}"},
        "leakage": {"core_sentence_leak_rate": summary["core"]["sentence_leak_rate"],
                    "core_values_fully_masked": summary["core"]["recall_full"],
                    "address_plate_leak_rate_not_in_scope": summary["extra_not_in_scope"]["sentence_leak_rate"]},
        "latency_ms_per_sentence": summary["latency_ms"],
    }


def print_table(rows):
    w = max(len(r[0]) for r in rows)
    for r in rows:
        print(f"  {r[0]:<{w}}  {r[1]:>22}  {r[2]:>22}")


def fmt_type(r):
    return f"{r['full']}/{r['n']} full ({r['recall']:.0%}), {r['partial']} part" if r else "-"


def main():
    s = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-score", type=float, default=s.pii_name_model_min_score)
    ap.add_argument("--model-id", default=s.pii_name_model_id)
    ap.add_argument("--revision", default=s.pii_name_model_revision or None)
    args = ap.parse_args()

    cases = load_cases(PROJECT_DIR / "eval" / "pii_cases.jsonl")
    laws = law_texts(s)
    print(f"{len(cases)} labelled cases (development set) · {len(laws)} law articles · "
          f"{sum(1 for _ in GOLDEN.open(encoding='utf-8'))} regression inputs\n")

    rules = PIIMasker()
    m0 = rss_mb()
    t0 = time.perf_counter()
    from app.pii.name_model import load_name_model
    model = load_name_model("camelbert-da", model_id=args.model_id, min_score=args.min_score, revision=args.revision)
    load_s = time.perf_counter() - t0
    m1 = rss_mb()
    hybrid = PIIMasker(name_model=model, model_error_policy="fail_closed")
    print(f"CAMeLBERT loaded in {load_s:.1f} s ({getattr(model, 'source', '')}), min_score={args.min_score}"
          + (f", process memory {m0:.0f} -> {m1:.0f} MB (+{m1 - m0:.0f} MB)" if m0 else "") + "\n")

    out = {"settings": {"min_score": args.min_score, "model": getattr(model, "source", args.model_id)},
           "resources": {"load_seconds": load_s, "rss_before_mb": m0, "rss_after_mb": m1}}
    results = {}
    for name, masker in (("rules", rules), ("hybrid", hybrid)):
        print(f"evaluating {name} ...")
        res = evaluate(MaskerDetector(masker, name=name), cases)
        legal = legal_text_fp(masker, laws)
        results[name] = (res, summarize(res), legal)
        out[name] = {"sections": section(results[name][1], legal), "legal_text": legal,
                     "long_text_2000_chars": long_text_latency(masker)}
    out["diff"] = diff_cases(results["rules"][0], results["hybrid"][0])
    out["regression_inputs"] = golden_changes(rules, hybrid)

    R, H = out["rules"]["sections"], out["hybrid"]["sections"]
    print("\n                                  rules-only               rules + CAMeLBERT")
    print_table([
        ("Names", fmt_type(R["names"]), fmt_type(H["names"])),
        ("Phones", fmt_type(R["phones"]), fmt_type(H["phones"])),
        *[(f"{t}", fmt_type(R["structured"][t]), fmt_type(H["structured"][t])) for t in R["structured"]],
        ("Core sentence leak rate", f"{R['leakage']['core_sentence_leak_rate']:.1%}",
         f"{H['leakage']['core_sentence_leak_rate']:.1%}"),
        ("Core values fully masked", f"{R['leakage']['core_values_fully_masked']:.1%}",
         f"{H['leakage']['core_values_fully_masked']:.1%}"),
        ("No-PII sentences w/ false alarm", R["false_positives"]["negative_sentences_with_false_alarm"],
         H["false_positives"]["negative_sentences_with_false_alarm"]),
        ("False-alarm spans (all cases)", str(R["false_positives"]["false_alarm_spans_all_cases"]),
         str(H["false_positives"]["false_alarm_spans_all_cases"])),
        ("Role/family words hidden", str(R["false_positives"]["role_word_false_alarms"]),
         str(H["false_positives"]["role_word_false_alarms"])),
        ("Sub-word spans", str(R["false_positives"]["subword_spans"]), str(H["false_positives"]["subword_spans"])),
        ("Law articles with a name span", R["false_positives"]["legal_articles_with_a_name_span"],
         H["false_positives"]["legal_articles_with_a_name_span"]),
        ("Spans on law text (all types)", str(R["false_positives"]["legal_text_spans"]),
         str(H["false_positives"]["legal_text_spans"])),
        ("Address/plate leak (not in scope)", f"{R['leakage']['address_plate_leak_rate_not_in_scope']:.1%}",
         f"{H['leakage']['address_plate_leak_rate_not_in_scope']:.1%}"),
        ("ms per sentence (median/p95)", f"{R['latency_ms_per_sentence']['median']:.1f}/"
                                         f"{R['latency_ms_per_sentence']['p95']:.1f}",
         f"{H['latency_ms_per_sentence']['median']:.1f}/{H['latency_ms_per_sentence']['p95']:.1f}"),
        ("ms per 2,000 chars (median/p95)", f"{out['rules']['long_text_2000_chars']['ms_median']:.1f}/"
                                            f"{out['rules']['long_text_2000_chars']['ms_p95']:.1f}",
         f"{out['hybrid']['long_text_2000_chars']['ms_median']:.1f}/"
         f"{out['hybrid']['long_text_2000_chars']['ms_p95']:.1f}"),
    ])
    d = out["diff"]
    print(f"\nHybrid fixes {len(d['fixed_by_hybrid'])} values the rules missed; worse on "
          f"{len(d['worse_with_hybrid'])}; adds {len(d['new_false_alarms'])} false-alarm spans.")
    for g in d["fixed_by_hybrid"][:12]:
        print(f"  + {g['case']} {g['type']:8s} {g['value']}")
    for g in d["worse_with_hybrid"]:
        print(f"  - {g['case']} {g['type']:8s} {g['value']} ({g['status']})")
    for f in d["new_false_alarms"][:15]:
        print(f"  ! {f['case']} '{f['span']}'  in: {f['text'][:70]}")
    names_on_law = out["hybrid"]["legal_text"]["name_examples"][:15]
    if names_on_law:
        print("\nName spans the hybrid finds in LAW text (false positives unless a law names a real person):")
        for x in names_on_law:
            print(f"  {x['article']:16s} '{x['text']}'")
    g = out["regression_inputs"]
    print(f"\nRegression inputs: hybrid changes {g['changed']} of {g['inputs']} outputs (examples in the JSON).")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    path = PROJECT_DIR / "logs" / f"pii_camelbert_eval_{stamp}.json"
    path.parent.mkdir(exist_ok=True)
    out["full_results"] = {k: {"summary": v[1]} for k, v in results.items()}
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\nSaved {path}")
    print("Development-set numbers are optimistic (the set informed the design); the held-out set (Phase 3C) decides.")


if __name__ == "__main__":
    main()
