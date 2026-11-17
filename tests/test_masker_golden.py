"""The default masker (no name model) must reproduce, byte for byte, the outputs recorded before Phase 3A.

tests/fixtures/masker_golden.jsonl was written by tests/fixtures/build_masker_golden.py with the Step 2
masker (masker.py sha256 c37eda56…ac23d9). A failure here means default masking behaviour changed.
"""
import json
from pathlib import Path

from app.config import Settings
from app.pii.factory import build_masker
from app.pii.masker import PIIMasker

GOLDEN = Path(__file__).parent / "fixtures" / "masker_golden.jsonl"


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def test_golden_file_is_substantial():
    rows = _golden()
    assert len(rows) >= 1600
    assert sum(1 for r in rows if r["mapping"]) >= 1000  # most inputs exercise masking


def test_default_masker_reproduces_every_recorded_output():
    m = PIIMasker()
    mismatches = []
    for r in _golden():
        out = m.mask(r["text"])
        if out.text != r["masked"] or out.mapping != r["mapping"] or out.model_error is not None:
            mismatches.append(r["text"])
    assert mismatches == [], f"{len(mismatches)} outputs changed, first: {mismatches[:3]}"


def test_masker_built_from_default_settings_is_the_same():
    masker, status = build_masker(Settings())
    assert status.configured == "none" and status.active == "rules" and status.serving and not status.degraded
    assert masker.name_model is None
    for r in _golden()[:300]:
        assert masker.mask(r["text"]).text == r["masked"]


def test_restore_round_trips_every_recorded_output():
    for r in _golden():
        assert PIIMasker.restore(r["masked"], r["mapping"]) == r["text"]
