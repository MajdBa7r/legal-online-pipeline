"""Day 1 check: Supabase reachability, latency from this machine, data version, pack sizes,
and one hybrid_search call (full-text only and with an embedding if keys are set).

    python scripts/check_supabase.py
"""
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.store import SupabaseStore  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.resources import Resources  # noqa: E402
from app.text.arabic import build_fts_query  # noqa: E402

QUESTION = "ما مدة الإجازة السنوية للعامل؟"


async def main():
    s = Settings.from_env()
    async with httpx.AsyncClient() as http:
        store = SupabaseStore(s.supabase_url, s.supabase_service_key, http)

        times = []
        for _ in range(5):
            t0 = time.monotonic()
            ok = await store.ping()
            times.append((time.monotonic() - t0) * 1000)
        print(f"ping ok={ok}  median={statistics.median(times):.0f} ms  max={max(times):.0f} ms")

        dv = await store.latest_data_version()
        print("data version:", dv)

        res = Resources.from_settings(s)
        await res.load(store)
        for d, arts in res.packs.items():
            chars = res.pack_chars.get(d, 0)
            print(f"pack {d:20s} articles={len(arts):4d} chars={chars:7d} ~tokens={chars // 3:6d} "
                  f"fits={res.pack_fits(d)}")
        print("glossary entries:", len(res.glossary))

        q = build_fts_query(QUESTION)
        t0 = time.monotonic()
        hits = await store.hybrid_search(q, None, 10)
        print(f"hybrid_search fts-only: {len(hits)} hits in {(time.monotonic() - t0) * 1000:.0f} ms")

        gw = ModelGateway.from_file(s.models_file, http)
        if not gw.missing_keys():
            ctx = RequestCtx(deadline=Deadline(30))
            vec = (await gw.embed([QUESTION], ctx))[0]
            t0 = time.monotonic()
            hits = await store.hybrid_search(q, vec, 10)
            print(f"hybrid_search hybrid:   {len(hits)} hits in {(time.monotonic() - t0) * 1000:.0f} ms")
            arts = await store.get_articles([h.article_id for h in hits[:3]])
            for a in arts.values():
                print("  ", a.header(), a.text[:80].replace("\n", " "))


if __name__ == "__main__":
    asyncio.run(main())
