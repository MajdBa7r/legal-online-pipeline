"""Retrieval smoke evaluation on the local store (developer set in eval/retrieval_smoke.jsonl).

    python scripts/eval_retrieval.py                  # keyword baseline (no API key needed)
    python scripts/eval_retrieval.py --domain-filter  # search only inside the expected domain
    python scripts/eval_retrieval.py --hybrid         # keyword + vectors (needs OPENAI_API_KEY and
                                                      # data/dev/embeddings built by build_local_embeddings.py)
    python scripts/eval_retrieval.py --show-misses

Only law text and the developer queries are sent to the embedding model; no user data.
"""
import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import PROJECT_DIR, Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.local_store import LocalStore  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.retrieval_eval import load_queries, run_queries, summarize  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(PROJECT_DIR / "eval" / "retrieval_smoke.jsonl"))
    ap.add_argument("--domain-filter", action="store_true")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--show-misses", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    store = LocalStore(s.local_data_dir, embeddings_dir=s.local_embeddings_dir, chunk_max_chars=s.chunk_max_chars)
    queries = load_queries(args.queries)

    embed_fn = None
    http = httpx.AsyncClient()
    if args.hybrid:
        if not store.vectors:
            sys.exit("No local chunk vectors found. Run scripts/build_local_embeddings.py first.")
        gw = ModelGateway.from_file(s.models_file, http)
        gw.roles["embed"].timeout_s = 20

        async def embed_fn(text: str):
            return (await gw.embed([text], RequestCtx(deadline=Deadline(60))))[0]

    try:
        results = await run_queries(store, queries, use_domain_filter=args.domain_filter, embed_fn=embed_fn)
    finally:
        await http.aclose()

    mode = "hybrid (keyword + vectors)" if args.hybrid else "keyword baseline"
    print(f"{mode}{' with domain filter' if args.domain_filter else ''} | {len(queries)} queries | "
          f"{len(store.articles)} articles, {len(store.chunks)} chunks")
    print(f"{'group':28s} {'n':>3s} {'hit@1':>6s} {'hit@3':>6s} {'hit@5':>6s} {'hit@10':>7s} {'MRR':>6s}")
    for name, row in summarize(results).items():
        print(f"{name:28s} {int(row['n']):>3d} {row['hit@1']:>6.2f} {row['hit@3']:>6.2f} {row['hit@5']:>6.2f} "
              f"{row['hit@10']:>7.2f} {row['mrr']:>6.2f}")
    if args.show_misses:
        print("\nQueries whose expected article is not in the top 3:")
        for r in results:
            if not r.first_rank or r.first_rank > 3:
                print(f"  {r.id:7s} rank={r.first_rank}  {r.query}\n          got {r.ranked[:5]}")


if __name__ == "__main__":
    asyncio.run(main())
