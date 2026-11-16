"""Checks on the real scraped data in data/dev/ (skipped when the files are not there, e.g. in CI).

The four focus-domain files come from the scraper team's Drive folder and are not committed.
Expected values below were read from those files on 2026-11-17; if the scraper team re-scrapes,
update them deliberately rather than loosening the tests.
"""
import asyncio
import hashlib
from pathlib import Path

import pytest

from app.config import PROJECT_DIR
from app.db.local_store import LocalStore
from app.retrieval_eval import load_queries, run_queries, summarize

DATA = PROJECT_DIR / "data" / "dev"
FOCUS = {"lob-2-3392": "labour", "lob-2-2803": "rentals", "lob-2-3391": "social_security",
         "lob-2-3169": "consumer_protection"}
pytestmark = pytest.mark.skipif(not all((DATA / f"{k}.json").exists() for k in FOCUS),
                                reason="real dev data not in data/dev/")


@pytest.fixture(scope="module")
def store():
    return LocalStore(DATA)


def test_four_focus_laws_and_their_actual_status(store):
    r = store.quality_report()
    expected = {  # law_id: (status as displayed, mapped status, amended, articles)
        "lob-2-3392": ("ساري معدل", "in_force", True, 142),
        "lob-2-2803": ("ساري معدل", "in_force", True, 22),
        "lob-2-3391": ("ساري معدل", "in_force", True, 110),
        "lob-2-3169": ("ساري", "in_force", False, 27),
    }
    for lid, (raw, status, amended, n) in expected.items():
        assert r[lid]["domain"] == FOCUS[lid]
        assert (r[lid]["status_raw"], r[lid]["status"], r[lid]["law_amended"]) == (raw, status, amended), lid
        assert r[lid]["articles_loaded"] == r[lid]["articles_displayed"] == n, lid
        # the source has no article-level status; none of these articles reads as repealed
        assert r[lid]["article_level_status_in_source"] is False
        assert r[lid]["articles_repealed_by_text"] == []


def test_known_scraper_issues_are_detected(store):
    r = store.quality_report()
    assert any("85 مكررة" in m for m in r["lob-2-3391"]["merged_article_headings"])
    assert r["lob-2-3391"]["possible_appended_schedules"] == ["110"]
    assert r["lob-2-3392"]["possible_appended_schedules"] == ["142"]
    assert len(r["lob-2-3391"]["question_mark_or_replacement_char"]) > 0


def test_raw_files_unchanged_by_loading():
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.json")}
    s = LocalStore(DATA)
    s.quality_report()
    asyncio.run(s.hybrid_search("اجازه", None, 5))
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in DATA.glob("*.json")}


def test_keyword_baseline_on_developer_smoke_set(store):
    """Regression guard for the keyword baseline (measured 2026-11-17: formal hit@3 0.97, hit@10 1.00).

    Dialect queries are reported but not asserted: closing that gap is the job of the
    understanding rewrites and embeddings, measured separately.
    """
    queries = load_queries(PROJECT_DIR / "eval" / "retrieval_smoke.jsonl")
    table = summarize(asyncio.run(run_queries(store, queries)))
    formal = table["style=formal"]
    assert formal["n"] >= 30
    assert formal["hit@3"] >= 0.90
    assert formal["hit@10"] >= 0.95
