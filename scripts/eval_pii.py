"""PII benchmark on eval/pii_cases.jsonl (synthetic data only).

    python scripts/eval_pii.py                                   # rules (what production runs today)
    python scripts/eval_pii.py --show-errors                     # also list misses and false alarms
    python scripts/eval_pii.py --family-expansion on             # rules + family-name expansion
    python scripts/eval_pii.py --name-model camelbert-da         # rules + CAMeLBERT-da (needs requirements-ml.txt)
    python scripts/eval_pii.py --name-model camelbert-da --min-score 0.6 --family-expansion on
    python scripts/eval_pii.py --configs all --save              # the four configurations, one model load
    python scripts/eval_pii.py --compare logs/pii_eval_rules.json logs/pii_eval_rules+camelbert-da.json

The default --min-score comes from PII_NAME_MODEL_MIN_SCORE (0.0 = keep every PERS span).
Thresholds are chosen on the DEVELOPMENT cases only. A held-out file is refused unless --final-heldout is
given, which is meant to be used once, after every setting is frozen.
This script measures only; it does not change what the pipeline does.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_DIR, Settings  # noqa: E402
from app.pii.detectors import MaskerDetector, RuleDetector  # noqa: E402
from app.pii.masker import PIIMasker  # noqa: E402
from app.pii.pii_eval import evaluate, format_report, load_cases, misses, summarize  # noqa: E402


def compare(paths: list[str]) -> None:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            s = json.load(f)["summary"]
        bt = s["core"]["by_type"]
        rows.append((s["detector"], s["core"]["sentence_leak_rate"], bt.get("name", {}).get("recall", 0.0),
                     min(bt[t]["recall"] for t in ("phone", "national_id", "iban", "email") if t in bt),
                     s["extra_not_in_scope"]["sentence_leak_rate"], s["false_alarms"]["rate"],
                     s["false_alarms"]["total_false_alarm_spans"], s.get("role_word_false_alarms", "-"),
                     s.get("subword_spans", "-"), s["latency_ms"]["median"]))
    print(f"{'detector':42s} {'core leak':>9s} {'names':>6s} {'min struct.':>11s} {'extra leak':>10s} "
          f"{'neg FA':>7s} {'FA spans':>8s} {'role FA':>7s} {'subword':>7s} {'ms/sent':>8s}")
    for r in sorted(rows, key=lambda r: (r[1], r[5])):
        print(f"{r[0]:42s} {r[1]:>9.1%} {r[2]:>6.0%} {r[3]:>11.0%} {r[4]:>10.1%} {r[5]:>7.0%} {r[6]:>8d} "
              f"{str(r[7]):>7s} {str(r[8]):>7s} {r[9]:>8.2f}")


def run(detector, cases, args) -> Path | None:
    result = evaluate(detector, cases)
    summary = summarize(result)
    print(format_report(summary))
    if args.show_errors:
        print("\nNot fully masked:")
        for m in misses(result):
            print(f"  {m['case']}  {m['type']:12s} {m['status']:8s} {m['value']}   [{', '.join(m['tags'])}]")
        print("\nFalse alarms:")
        for f in result["false_alarms"]:
            print(f"  {f['case']}  {f['type']:12s} '{f['span']}'   in: {f['text']}")
        print("\nRole/family words hidden:")
        for f in result["role_word_false_alarms"]:
            print(f"  {f['case']}  '{f['word']}' inside '{f['span']}'")
        print("\nSub-word spans:")
        for f in result["subword_spans"]:
            print(f"  {f['case']}  '{f['span']}'   in: {f['text']}")
    if not args.save:
        return None
    out = PROJECT_DIR / "logs" / f"pii_eval_{detector.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "result": result, "settings": vars(args)}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved {out}")
    return out


def main():
    settings = Settings.from_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(PROJECT_DIR / "eval" / "pii_cases.jsonl"))
    ap.add_argument("--name-model", choices=["none", "camelbert-da"], default="none")
    ap.add_argument("--model-id", default=settings.pii_name_model_id)
    ap.add_argument("--revision", default=settings.pii_name_model_revision or None)
    ap.add_argument("--min-score", type=float, default=settings.pii_name_model_min_score)
    ap.add_argument("--family-expansion", choices=["on", "off"], default="off")
    ap.add_argument("--configs", choices=["one", "all"], default="one",
                    help="all = rules, rules+family, rules+model, rules+model+family")
    ap.add_argument("--final-heldout", action="store_true",
                    help="required to score a held-out file; use once, after all settings are frozen")
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return
    if "heldout" in Path(args.cases).name.lower() and not args.final_heldout:
        sys.exit("Refusing to score a held-out file without --final-heldout (it must not be used for tuning).")
    if args.configs == "all" and args.name_model == "none":
        args.name_model = "camelbert-da"

    cases = load_cases(args.cases)
    print(f"{len(cases)} synthetic cases from {Path(args.cases).name}: "
          f"{sum(c.group == 'core' for c in cases)} core, {sum(c.group == 'extra' for c in cases)} extra, "
          f"{sum(c.group == 'negative' for c in cases)} negative\n")

    model = None
    if args.name_model != "none":
        from app.pii.name_model import load_name_model
        t0 = time.perf_counter()
        model = load_name_model(args.name_model, model_id=args.model_id, min_score=args.min_score,
                                revision=args.revision)
        print(f"loaded {args.name_model} ({getattr(model, 'source', args.model_id)}) in "
              f"{time.perf_counter() - t0:.1f} s, min_score={args.min_score}\n")

    if args.configs == "all":
        plan = [(None, False), (None, True), (model, False), (model, True)]
    else:
        plan = [(model, args.family_expansion == "on")]

    saved = []
    for m, family in plan:
        if m is None and not family:
            detector = RuleDetector()
        else:
            name = "rules" + (f"+{m.name}" if m else "") + ("+family" if family else "")
            if m is not None and args.min_score:
                name += f"@{args.min_score:g}"
            detector = MaskerDetector(PIIMasker(name_model=m, model_error_policy="fail_closed",
                                                family_name_expansion=family), name=name)
        print("=" * 78)
        out = run(detector, cases, args)
        if out:
            saved.append(str(out))
        print()
    if len(saved) > 1:
        compare(saved)


if __name__ == "__main__":
    main()
