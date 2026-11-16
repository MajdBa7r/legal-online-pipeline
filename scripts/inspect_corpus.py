"""Report what the local store loaded from data/dev/: status per law, article counts, and known data issues.

    python scripts/inspect_corpus.py            # readable report
    python scripts/inspect_corpus.py --json     # full report as JSON

Nothing in the data folder is modified. Issues are reported, not fixed: fixes belong to the scraper team.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings  # noqa: E402
from app.db.local_store import LocalStore  # noqa: E402
from app.text.arabic import build_fts_query  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    s = Settings.from_env()
    store = LocalStore(args.data_dir or s.local_data_dir, embeddings_dir=s.local_embeddings_dir,
                       chunk_max_chars=s.chunk_max_chars)
    report = store.quality_report()
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    dv = asyncio.run(store.latest_data_version())
    print(f"Data folder: {store.data_dir}")
    print(f"Data version: {dv.version}  scraped: {dv.scraped_at}  laws: {len(store.laws)}  "
          f"articles: {len(store.articles)}  chunks: {len(store.chunks)}  vectors: {len(store.vectors)}")
    for w in store.warnings:
        print(f"WARNING: {w}")
    print()
    print(f"{'law_id':12s} {'domain':20s} {'status as displayed':20s} {'mapped':10s} {'last amend.':11s} "
          f"{'articles':9s} {'in force':9s} {'chars':>8s}")
    for lid, r in report.items():
        print(f"{lid:12s} {str(r['domain']):20s} {str(r['status_raw']):20s} {r['status']:10s} "
              f"{str(r['last_amendment']):11s} {r['articles_loaded']:>3}/{str(r['articles_displayed']):<5s} "
              f"{r['articles_in_force']:>8} {r['total_chars']:>8}")
    print()
    print("Article-level status: the source gives no per-article repeal/amendment status. Articles are only "
          "marked repealed when their own text says so.")
    for lid, r in report.items():
        print(f"\n== {lid}  {r['title']}")
        print(f"   definitions article: {r['definitions_article']}   chunks: {r['chunks']}   "
              f"longest: {r['longest_articles']}")
        issues = {
            "articles repealed by their own text": r["articles_repealed_by_text"],
            "merged article headings (another article inside this one)": r["merged_article_headings"],
            "schedules/tables appended to an article": r["possible_appended_schedules"],
            "chapter heading text inside article text": r["chapter_heading_inside_text"],
            "'?' where a letter is missing (e.g. ذ)": r["question_mark_or_replacement_char"],
            "'0' used as a full stop": r["zero_used_as_full_stop"],
            "attachment URLs inside text": r["articles_with_urls"],
            "articles with invisible characters": r["articles_with_invisible_chars"],
            "articles with no chapter label": r["articles_without_chapter"],
            "empty articles": r["empty_articles"],
            "duplicate article numbers": r["duplicate_numbers"],
        }
        for name, val in issues.items():
            if val:
                print(f"   - {name}: {val}")
    print("\nSample searches (keyword baseline):")
    for q in ("اجازة سنوية", "اخلاء المأجور", "تقاعد مبكر", "السلعة المعيبة"):
        hits = asyncio.run(store.hybrid_search(build_fts_query(q), None, 10))
        print(f"   {q}: {list(dict.fromkeys(h.article_id for h in hits))[:5]}")


if __name__ == "__main__":
    main()
