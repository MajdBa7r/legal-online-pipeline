"""LocalStore on invented fixture laws (tests/fixtures/local_laws), in the scraper's exact JSON format."""
import asyncio
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.db.local_store import LocalStore, split_into_chunks
from app.db.store import Store, SupabaseStore
from app.schemas.domain import Article, SearchHit
from app.text.arabic import build_fts_query

FIX = Path(__file__).parent / "fixtures"
LAWS = FIX / "local_laws"


def run(coro):
    return asyncio.run(coro)


@pytest.fixture()
def store(tmp_path):
    return LocalStore(LAWS, domains_file=FIX / "local_domains_test.yaml", traces_path=tmp_path / "traces.jsonl",
                      glossary_file=FIX / "glossary_test.csv")


def search(store, text, n=10, domain=None, embedding=None):
    hits = run(store.hybrid_search(build_fts_query(text), embedding, n, domain))
    return list(dict.fromkeys(h.article_id for h in hits))


# ---- loading and mapping ----------------------------------------------------
def test_source_files_are_never_modified(tmp_path):
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LAWS.glob("*.json")}
    s = LocalStore(LAWS, domains_file=FIX / "local_domains_test.yaml", traces_path=tmp_path / "t.jsonl")
    run(s.hybrid_search("اجازه", None, 5))
    run(s.insert_trace({"request_id": "x"}))
    s.quality_report()
    after = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in LAWS.glob("*.json")}
    assert before == after
    assert sorted(p.name for p in LAWS.iterdir()) == sorted(before)  # nothing new written into the data folder


def test_law_status_mapping_is_explicit_and_fails_closed(store):
    assert store.laws["lob-2-9001"].status == "in_force" and store.laws["lob-2-9001"].amended
    assert store.laws["lob-2-9002"].status == "in_force" and not store.laws["lob-2-9002"].amended
    assert store.laws["lob-2-9003"].status == "repealed"
    assert store.laws["lob-2-9004"].status == "unknown"
    assert not store.articles["lob-2-9004:1"].in_force()  # unknown status is NOT treated as in force
    assert any("unknown status" in w for w in store.warnings)


def test_article_mapping(store):
    a = store.articles["lob-2-9001:3"]
    assert isinstance(a, Article)
    assert a.id == "lob-2-9001:3" and a.legislation_id == "lob-2-9001" and a.article_no == "3"
    assert a.law_title == "قانون العمل التجريبي رقم 1 لسنة 2001"   # tatweel removed for display only
    assert a.text.startswith("لكل عامل الحق")                      # original text unchanged
    assert a.law_type == "law" and a.last_amendment == "2019" and a.domain == "labour"
    assert a.law_status_raw == "ساري معدل" and a.chapter == "الفصل الثاني · الإجازات"
    assert a.source_url.startswith("https://lob.gov.jo")
    assert a.label() == "قانون العمل التجريبي رقم 1 لسنة 2001، المادة 3"


def test_article_repeal_only_from_its_own_text(store):
    assert store.articles["lob-2-9001:5"].repealed
    assert store.articles["lob-2-9001:5"].repeal_evidence
    assert not store.articles["lob-2-9001:3"].repealed


def test_chunking_keeps_article_id_and_header(store):
    chunks = [c for c in store.chunks if c.article_id == "lob-2-9001:6"]
    assert len(chunks) > 1
    assert all(c.id.startswith("lob-2-9001:6#") for c in chunks)
    assert all(c.text.startswith("[قانون العمل التجريبي رقم 1 لسنة 2001 – قانون – المادة 6]") for c in chunks)
    assert all(len(c.text) < 1400 for c in chunks)


def test_split_into_chunks_handles_one_huge_line():
    pieces = split_into_chunks("كلمة " * 2000, max_chars=500)
    assert len(pieces) > 5 and all(len(p) <= 500 for p in pieces)


# ---- search -----------------------------------------------------------------
def test_keyword_search_ranks_the_right_article(store):
    assert search(store, "مدة الإجازة السنوية للعامل")[0] == "lob-2-9001:3"
    assert search(store, "اجازة مرضية بتقرير طبي")[0] == "lob-2-9001:4"
    assert search(store, "إخلاء المأجور لعدم دفع بدل الإجارة")[0] == "lob-2-9002:2"
    assert search(store, "المالك قطع المياه عن المستأجر")[0] == "lob-2-9002:3"


def test_search_excludes_repealed_unknown_and_repealed_articles(store):
    ids = search(store, "إخلاء المأجور دفع بدل الإجارة", n=30)
    assert "lob-2-9003:1" not in ids  # law repealed
    assert "lob-2-9004:1" not in ids  # unknown status
    assert "lob-2-9001:5" not in search(store, "ملغاة", n=30)


def test_domain_filter(store):
    ids = search(store, "الإجازة السنوية بدل الإجارة", n=30, domain="rentals")
    assert ids and all(i.startswith("lob-2-9002") for i in ids)


def test_query_or_syntax_and_free_text_give_same_ranking(store):
    a = run(store.hybrid_search("اجازه or سنويه or عامل", None, 10))
    b = run(store.hybrid_search("اجازه سنويه عامل", None, 10))
    assert [h.chunk_id for h in a] == [h.chunk_id for h in b]


def test_no_matching_terms_returns_empty(store):
    assert run(store.hybrid_search(build_fts_query("زيزفون برتقال قرنفل"), None, 10)) == []


def test_results_are_chunk_level_hits_capped_like_the_sql_contract(store):
    hits = run(store.hybrid_search("التقرير السنوي المنشاه", None, 500))
    assert all(isinstance(h, SearchHit) and h.chunk_id and h.article_id for h in hits)
    assert len(hits) <= 50
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_vector_side_is_fused_with_rrf(store):
    # fake 3-d vectors: only the rentals "قطع المياه" chunk points along x
    for cid in store.chunks_by_id:
        store.vectors[cid] = [0.0, 1.0, 0.0]
    target = next(c.id for c in store.chunks if c.article_id == "lob-2-9002:3")
    store.vectors[target] = [1.0, 0.0, 0.0]
    # a query with no keyword overlap is found by the vector side alone
    hits = run(store.hybrid_search(build_fts_query("زيزفون"), [1.0, 0.0, 0.0], 5))
    assert hits[0].chunk_id == target
    # keyword + vector agreement beats keyword alone
    hits = run(store.hybrid_search(build_fts_query("قطع المياه المالك"), [1.0, 0.0, 0.0], 5))
    assert hits[0].chunk_id == target
    # wrong dimension: vectors ignored, keyword ranking still works
    hits = run(store.hybrid_search(build_fts_query("الإجازة السنوية"), [1.0, 0.0], 5))
    assert hits[0].article_id == "lob-2-9001:3"


# ---- other Store methods ----------------------------------------------------
def test_packs_definitions_glossary_version_traces(store, tmp_path):
    packs = run(store.load_domain_packs())
    assert packs["labour"][:3] == ["lob-2-9001:1", "lob-2-9001:2", "lob-2-9001:3"]
    assert run(store.definitions_for(["lob-2-9001", "lob-2-9002"])) == ["lob-2-9001:2"]
    gl = run(store.load_glossary())
    assert [g.phrase for g in gl] == ["فنشني", "الاجار"] and gl[0].phrase_norm == "فنشني"
    dv = run(store.latest_data_version())
    assert dv.version.startswith("local-") and dv.scraped_at == "2026-11-16"
    assert dv.version == run(LocalStore(LAWS, domains_file=FIX / "local_domains_test.yaml")
                             .latest_data_version()).version  # stable across loads
    run(store.insert_trace({"request_id": "abc", "outcome": "answer"}))
    rows = [json.loads(l) for l in (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows == [{"request_id": "abc", "outcome": "answer"}]
    assert run(store.ping())


def test_quality_report_flags_known_scraper_issues(store):
    r = store.quality_report()
    assert r["lob-2-9002"]["merged_article_headings"] == ["3 contains 'المادة 3 مكررة' at char 92"]
    assert r["lob-2-9001"]["possible_appended_schedules"] == ["7"]
    assert r["lob-2-9002"]["question_mark_or_replacement_char"] == ["4"]
    assert r["lob-2-9001"]["articles_repealed_by_text"] == ["5"]
    assert r["lob-2-9001"]["article_level_status_in_source"] is False
    assert r["lob-2-9001"]["definitions_article"] == "lob-2-9001:2"


# ---- compatibility with the future Supabase store ---------------------------
def test_same_interface_as_supabase_store():
    for name, proto in inspect.getmembers(Store, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        local, remote = getattr(LocalStore, name), getattr(SupabaseStore, name)
        assert inspect.iscoroutinefunction(local) and inspect.iscoroutinefunction(remote), name
        proto_params = list(inspect.signature(proto).parameters)
        assert list(inspect.signature(local).parameters)[: len(proto_params)] == proto_params, name
        assert list(inspect.signature(remote).parameters)[: len(proto_params)] == proto_params, name


def test_factory_picks_backend(tmp_path):
    import httpx

    from app.db.factory import make_store

    s = Settings(store_backend="local", local_data_dir=str(LAWS), local_traces_path=str(tmp_path / "t.jsonl"),
                 local_embeddings_dir=str(tmp_path / "none"))
    assert isinstance(make_store(s, httpx.AsyncClient()), LocalStore)
    with pytest.raises(ValueError):
        make_store(Settings(store_backend="mongo"), httpx.AsyncClient())
