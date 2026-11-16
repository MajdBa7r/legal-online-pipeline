"""PII benchmark: score the current production rules on eval/pii_cases.jsonl (synthetic data only).

    python scripts/eval_pii.py                 # report for the rules
    python scripts/eval_pii.py --show-errors   # also list every miss, partial mask and false alarm
    python scripts/eval_pii.py --save          # also write logs/pii_eval_rules.json (for comparison)
    python scripts/eval_pii.py --compare logs/pii_eval_rules.json logs/pii_eval_<model>.json

Model detectors (CAMeLBERT, GLiNER, ...) run in notebooks/pii_benchmark.ipynb on Colab with the same
harness; their saved JSON files can be compared here with --compare.
This script measures only. It does not change masker.py or anything the pipeline does.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import PROJECT_DIR  # noqa: E402
from app.pii.detectors import RuleDetector  # noqa: E402
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
                     s["false_alarms"]["total_false_alarm_spans"], s["latency_ms"]["median"]))
    print(f"{'detector':45s} {'core leak':>9s} {'names':>6s} {'min struct.':>11s} {'extra leak':>10s} "
          f"{'neg FA':>7s} {'FA spans':>8s} {'ms/sent':>8s}")
    for r in sorted(rows, key=lambda r: (r[1], r[5])):
        print(f"{r[0]:45s} {r[1]:>9.1%} {r[2]:>6.0%} {r[3]:>11.0%} {r[4]:>10.1%} {r[5]:>7.0%} {r[6]:>8d} {r[7]:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(PROJECT_DIR / "eval" / "pii_cases.jsonl"))
    ap.add_argument("--show-errors", action="store_true")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--compare", nargs="+")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return

    cases = load_cases(args.cases)
    result = evaluate(RuleDetector(), cases)
    summary = summarize(result)
    print(f"{len(cases)} synthetic cases: "
          f"{sum(c.group == 'core' for c in cases)} core, {sum(c.group == 'extra' for c in cases)} extra, "
          f"{sum(c.group == 'negative' for c in cases)} negative\n")
    print(format_report(summary))
    if args.show_errors:
        print("\nNot fully masked:")
        for m in misses(result):
            print(f"  {m['case']}  {m['type']:12s} {m['status']:8s} {m['value']}   [{', '.join(m['tags'])}]")
        print("\nFalse alarms:")
        for f in result["false_alarms"]:
            print(f"  {f['case']}  {f['type']:12s} '{f['span']}'   in: {f['text']}")
    if args.save:
        out = PROJECT_DIR / "logs" / "pii_eval_rules.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "result": result}, f, ensure_ascii=False, indent=1)
        print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
