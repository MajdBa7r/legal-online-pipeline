"""PII benchmark harness: score any detector on eval/pii_cases.jsonl.

What is measured (characters, not tokens, so rules and models are comparable):
* A gold value is FULLY MASKED when every non-space character of it is covered by detected spans,
  PARTIAL when some are, MISSED when none are. Any detected type counts: for privacy, masking a
  national ID as a "phone" still protects it (type confusion is reported separately).
* Sentence LEAK: a sentence leaks if any gold value of the group is not fully masked.
  Core (name, phone, national_id, iban, email) and extra (address, plate) are reported separately;
  extra types are outside the current masking scope.
* FALSE ALARM: a detected span that overlaps no gold value. Reported overall and as the share of
  negative (no-PII) sentences with at least one false alarm.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.pii.detectors import CORE_TYPES, EXTRA_TYPES, Span


@dataclass
class Gold:
    case_id: str
    type: str
    value: str
    start: int
    end: int
    group: str
    tags: list[str]


@dataclass
class Case:
    id: str
    text: str
    group: str
    tags: list[str]
    gold: list[Gold] = field(default_factory=list)


def load_cases(path: str | Path) -> list[Case]:
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            raw = json.loads(line)
            c = Case(raw["id"], raw["text"], raw["group"], raw.get("tags", []))
            for p in raw["pii"]:
                start, found = 0, False
                while (i := c.text.find(p["value"], start)) != -1:
                    c.gold.append(Gold(c.id, p["type"], p["value"], i, i + len(p["value"]), c.group, c.tags))
                    start, found = i + len(p["value"]), True
                if not found:
                    raise ValueError(f"{c.id}: value {p['value']!r} not found in text")
            cases.append(c)
    return cases


def _covered(text: str, g: Gold, spans: list[Span]) -> tuple[float, set[str]]:
    chars = [i for i in range(g.start, g.end) if not text[i].isspace()]
    if not chars:
        return 1.0, set()
    hit, types = 0, set()
    for i in chars:
        for s in spans:
            if s.start <= i < s.end:
                hit += 1
                types.add(s.type)
                break
    return hit / len(chars), types


def evaluate(detector, cases: list[Case]) -> dict[str, Any]:
    gold_rows, fp_rows, sentence_rows, times = [], [], [], []
    for c in cases:
        t0 = time.perf_counter()
        spans = detector.detect(c.text)
        times.append(time.perf_counter() - t0)
        for g in c.gold:
            cov, types = _covered(c.text, g, spans)
            status = "full" if cov >= 0.999 else ("partial" if cov > 0 else "missed")
            gold_rows.append({"case": c.id, "type": g.type, "value": g.value, "group": g.group, "tags": g.tags,
                              "coverage": round(cov, 3), "status": status, "detected_types": sorted(types),
                              "text": c.text})
        for s in spans:
            if not any(s.start < g.end and g.start < s.end for g in c.gold):
                fp_rows.append({"case": c.id, "group": c.group, "type": s.type, "span": c.text[s.start:s.end],
                                "tags": c.tags, "text": c.text})
        core = [r for r in gold_rows if r["case"] == c.id and r["type"] in CORE_TYPES]
        extra = [r for r in gold_rows if r["case"] == c.id and r["type"] in EXTRA_TYPES]
        sentence_rows.append({"case": c.id, "group": c.group,
                              "core_leak": any(r["status"] != "full" for r in core) if core else None,
                              "extra_leak": any(r["status"] != "full" for r in extra) if extra else None,
                              "false_alarm": any(f["case"] == c.id for f in fp_rows)})
    return {"detector": getattr(detector, "name", type(detector).__name__), "gold": gold_rows,
            "false_alarms": fp_rows, "sentences": sentence_rows, "seconds": times}


def _rate(rows, key):
    rows = [r for r in rows if r[key] is not None]
    return (sum(1 for r in rows if r[key]) / len(rows), len(rows)) if rows else (0.0, 0)


def _by(rows, keyfn):
    groups = defaultdict(list)
    for r in rows:
        for k in keyfn(r):
            groups[k].append(r)
    return {k: {"n": len(v), "full": sum(r["status"] == "full" for r in v),
                "partial": sum(r["status"] == "partial" for r in v),
                "missed": sum(r["status"] == "missed" for r in v),
                "recall": sum(r["status"] == "full" for r in v) / len(v)} for k, v in sorted(groups.items())}


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    gold, sents = result["gold"], result["sentences"]
    core = [r for r in gold if r["type"] in CORE_TYPES]
    extra = [r for r in gold if r["type"] in EXTRA_TYPES]
    core_leak, n_core = _rate([s for s in sents if s["group"] == "core"], "core_leak")
    extra_leak, n_extra = _rate([s for s in sents if s["group"] == "extra"], "extra_leak")
    neg = [s for s in sents if s["group"] == "negative"]
    neg_fa = sum(s["false_alarm"] for s in neg)
    n_pred_true = sum(1 for r in gold if r["status"] != "missed")
    secs = sorted(result["seconds"])
    type_confusions = [r for r in gold if r["status"] != "missed" and r["detected_types"]
                       and r["type"] not in r["detected_types"]]
    return {
        "detector": result["detector"],
        "core": {
            "sentence_leak_rate": core_leak, "sentences": n_core,
            "recall_full": sum(r["status"] == "full" for r in core) / len(core) if core else 0.0,
            "by_type": _by(core, lambda r: [r["type"]]),
            "names_by_tag": _by([r for r in core if r["type"] == "name"],
                                lambda r: [t for t in r["tags"] if t not in ("mixed",)]),
            "phones_by_format": _by([r for r in core if r["type"] == "phone"],
                                    lambda r: [t for t in r["tags"] if t.startswith("phone_")] or ["mixed"]),
        },
        "extra_not_in_scope": {
            "sentence_leak_rate": extra_leak, "sentences": n_extra,
            "by_type": _by(extra, lambda r: [r["type"]]),
        },
        "false_alarms": {
            "negative_sentences_with_false_alarm": neg_fa, "negative_sentences": len(neg),
            "rate": neg_fa / len(neg) if neg else 0.0,
            "total_false_alarm_spans": len(result["false_alarms"]),
            "by_type": dict(sorted({t: sum(1 for f in result["false_alarms"] if f["type"] == t)
                                    for t in {f["type"] for f in result["false_alarms"]}}.items())),
        },
        "type_confusions": len(type_confusions),
        "latency_ms": {"median": 1000 * secs[len(secs) // 2] if secs else 0.0,
                       "p95": 1000 * secs[int(len(secs) * 0.95) - 1] if secs else 0.0},
        "gold_detected_any": n_pred_true,
    }


def format_report(summary: dict[str, Any]) -> str:
    s = summary
    lines = [f"Detector: {s['detector']}", "",
             "CORE (name, phone, national_id, iban, email) - in masking scope",
             f"  sentences leaking any core value: {s['core']['sentence_leak_rate']:.1%} "
             f"of {s['core']['sentences']}",
             f"  values fully masked: {s['core']['recall_full']:.1%}",
             f"  {'type':14s} {'n':>3s} {'full':>5s} {'partial':>8s} {'missed':>7s} {'recall':>7s}"]
    for t, r in s["core"]["by_type"].items():
        lines.append(f"  {t:14s} {r['n']:>3d} {r['full']:>5d} {r['partial']:>8d} {r['missed']:>7d} {r['recall']:>7.1%}")
    lines.append("  names by tag:")
    for t, r in s["core"]["names_by_tag"].items():
        lines.append(f"    {t:22s} n={r['n']:>2d}  full={r['full']:>2d}  partial={r['partial']:>2d}  "
                     f"missed={r['missed']:>2d}  recall={r['recall']:.0%}")
    lines.append("  phones by format:")
    for t, r in s["core"]["phones_by_format"].items():
        lines.append(f"    {t:22s} n={r['n']:>2d}  recall={r['recall']:.0%}")
    lines += ["", "EXTRA (address, plate) - NOT in masking scope, reported separately",
              f"  sentences leaking: {s['extra_not_in_scope']['sentence_leak_rate']:.1%} "
              f"of {s['extra_not_in_scope']['sentences']}"]
    for t, r in s["extra_not_in_scope"]["by_type"].items():
        lines.append(f"  {t:14s} n={r['n']:>2d}  full={r['full']:>2d}  partial={r['partial']:>2d}  "
                     f"missed={r['missed']:>2d}  recall={r['recall']:.0%}")
    fa = s["false_alarms"]
    lines += ["", "FALSE ALARMS",
              f"  negative sentences with a false alarm: {fa['negative_sentences_with_false_alarm']}/"
              f"{fa['negative_sentences']} ({fa['rate']:.0%})",
              f"  false-alarm spans in all sentences: {fa['total_false_alarm_spans']}  by type: {fa['by_type']}",
              f"  values detected but with a different type: {s['type_confusions']}",
              "", f"LATENCY per sentence: median {s['latency_ms']['median']:.2f} ms, "
                  f"p95 {s['latency_ms']['p95']:.2f} ms"]
    return "\n".join(lines)


def misses(result: dict[str, Any], *, include_extra: bool = True) -> list[dict]:
    return [r for r in result["gold"] if r["status"] != "full" and (include_extra or r["type"] in CORE_TYPES)]
