"""Data access. `Store` is the interface the pipeline depends on; `SupabaseStore`
talks to Supabase's REST API (PostgREST) with the service key, backend only.
Tests use an in-memory fake with the same methods.
"""
from __future__ import annotations

import logging
from typing import Any, Protocol

import httpx

from app.db import schema_map as S
from app.errors import StoreError
from app.schemas.domain import Article, DataVersion, GlossaryEntry, SearchHit
from app.text.arabic import normalize

log = logging.getLogger(__name__)


class Store(Protocol):
    async def hybrid_search(self, query_text: str, embedding: list[float] | None, match_count: int,
                            domain: str | None = None, timeout: float = 3.0) -> list[SearchHit]: ...
    async def get_articles(self, ids: list[str]) -> dict[str, Article]: ...
    async def load_domain_packs(self) -> dict[str, list[str]]: ...
    async def load_glossary(self) -> list[GlossaryEntry]: ...
    async def definitions_for(self, legislation_ids: list[str]) -> list[str]: ...
    async def latest_data_version(self) -> DataVersion | None: ...
    async def insert_trace(self, row: dict[str, Any]) -> None: ...
    async def ping(self) -> bool: ...


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


class SupabaseStore:
    def __init__(self, url: str, service_key: str, http: httpx.AsyncClient):
        if not url or not service_key:
            raise StoreError("SUPABASE_URL and SUPABASE_SERVICE_KEY must be set")
        self.base = url.rstrip("/") + "/rest/v1"
        self.http = http
        self.headers = {
            "apikey": service_key,
            "Authorization": f"Bearer {service_key}",
            "Content-Type": "application/json",
        }

    # ---- low level ---------------------------------------------------------
    async def _get(self, path: str, params: dict[str, str], timeout: float = 5.0) -> list[dict]:
        try:
            r = await self.http.get(f"{self.base}/{path}", params=params, headers=self.headers, timeout=timeout)
        except httpx.HTTPError as e:
            raise StoreError(f"GET {path}: {type(e).__name__}") from e
        if r.status_code >= 400:
            raise StoreError(f"GET {path}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    async def _rpc(self, fn: str, body: dict, timeout: float) -> list[dict]:
        try:
            r = await self.http.post(f"{self.base}/rpc/{fn}", json=body, headers=self.headers, timeout=timeout)
        except httpx.HTTPError as e:
            raise StoreError(f"rpc {fn}: {type(e).__name__}") from e
        if r.status_code >= 400:
            raise StoreError(f"rpc {fn}: HTTP {r.status_code}: {r.text[:200]}")
        return r.json()

    # ---- search ------------------------------------------------------------
    async def hybrid_search(self, query_text, embedding, match_count, domain=None, timeout=3.0):
        h = S.HYBRID_SEARCH
        body = {
            h["query_text"]: query_text,
            h["query_embedding"]: embedding,  # None -> full-text-only mode (needs contract support)
            h["match_count"]: match_count,
            h["filter_domain"]: domain,
        }
        rows = await self._rpc(h["function"], body, timeout)
        return [
            SearchHit(article_id=str(r[h["out_article_id"]]), score=float(r.get(h["out_score"], 0) or 0),
                      chunk_id=str(r.get(h["out_chunk_id"])) if r.get(h["out_chunk_id"]) is not None else None)
            for r in rows
        ]

    # ---- articles ----------------------------------------------------------
    def _article_select(self) -> str:
        a, l = S.ARTICLES, S.LEGISLATION
        leg_cols = ",".join(l[k] for k in ("id", "title", "type", "number", "year", "status", "last_amendment",
                                           "gazette", "domain", "coverage_level", "source_url"))
        return (f"{a['id']},{a['legislation_id']},{a['article_no']},{a['text']},{a['repealed']},"
                f"legislation:{a['legislation_id']}({leg_cols})")

    def _to_article(self, r: dict) -> Article:
        a, l = S.ARTICLES, S.LEGISLATION
        leg = r.get("legislation") or {}
        return Article(
            id=str(r[a["id"]]),
            article_no=str(r.get(a["article_no"]) or ""),
            text=r.get(a["text"]) or "",
            repealed=bool(r.get(a["repealed"]) or False),
            legislation_id=str(r.get(a["legislation_id"]) or ""),
            law_title=leg.get(l["title"]) or "",
            law_type=leg.get(l["type"]) or "law",
            law_number=str(leg.get(l["number"]) or ""),
            law_year=str(leg.get(l["year"]) or ""),
            law_status=leg.get(l["status"]) or "in_force",
            last_amendment=str(leg.get(l["last_amendment"])) if leg.get(l["last_amendment"]) else None,
            source_url=leg.get(l["source_url"]),
            gazette=str(leg.get(l["gazette"])) if leg.get(l["gazette"]) else None,
            domain=leg.get(l["domain"]),
            coverage_level=leg.get(l["coverage_level"]),
        )

    async def get_articles(self, ids):
        out: dict[str, Article] = {}
        unique = list(dict.fromkeys(str(i) for i in ids))
        for batch in _chunks(unique, 100):  # keep the URL short
            rows = await self._get(S.ARTICLES["table"], {
                "select": self._article_select(),
                S.ARTICLES["id"]: f"in.({','.join(batch)})",
            })
            for r in rows:
                art = self._to_article(r)
                out[art.id] = art
        return out

    # ---- startup resources -------------------------------------------------
    async def load_domain_packs(self):
        p = S.DOMAIN_PACKS
        rows = await self._get(p["table"], {
            "select": f"{p['domain']},{p['article_id']},{p['position']}",
            "order": f"{p['domain']}.asc,{p['position']}.asc",
            "limit": "20000",
        }, timeout=15)
        packs: dict[str, list[str]] = {}
        for r in rows:
            packs.setdefault(r[p["domain"]], []).append(str(r[p["article_id"]]))
        return packs

    async def load_glossary(self):
        g = S.GLOSSARY
        rows = await self._get(g["table"], {"select": f"{g['phrase']},{g['legal_term']},{g['domain']}",
                                            "limit": "5000"})
        return [
            GlossaryEntry(phrase=r[g["phrase"]], legal_term=r[g["legal_term"]], domain=r.get(g["domain"]),
                          phrase_norm=normalize(r[g["phrase"]]))
            for r in rows if r.get(g["phrase"])
        ]

    async def definitions_for(self, legislation_ids):
        """Definitions articles of these laws and of their parent laws (only where `relations` states it)."""
        if not legislation_ids:
            return []
        rel = S.RELATIONS
        ids = ",".join(dict.fromkeys(legislation_ids))
        try:
            parents = await self._get(rel["table"], {
                "select": rel["target_legislation_id"],
                rel["type"]: "eq.parent_law",
                rel["legislation_id"]: f"in.({ids})",
            })
            all_ids = list(dict.fromkeys(list(legislation_ids) +
                                         [str(r[rel["target_legislation_id"]]) for r in parents
                                          if r.get(rel["target_legislation_id"])]))
            rows = await self._get(rel["table"], {
                "select": rel["target_article_id"],
                rel["type"]: "eq.definitions",
                rel["legislation_id"]: f"in.({','.join(all_ids)})",
            })
            return [str(r[rel["target_article_id"]]) for r in rows if r.get(rel["target_article_id"])]
        except StoreError as e:
            log.warning("definitions_for failed, skipping expansion: %s", e)
            return []

    async def latest_data_version(self):
        d = S.DATA_VERSIONS
        try:
            rows = await self._get(d["table"], {"select": "*", "order": f"{d['created_at']}.desc", "limit": "1"})
        except StoreError:
            rows = await self._get(d["table"], {"select": "*", "limit": "1"})
        if not rows:
            return None
        r = rows[0]
        return DataVersion(version=str(r.get(d["version"])), scraped_at=str(r.get(d["scraped_at"]) or "") or None,
                           embedding_model=r.get(d["embedding_model"]), extra=r)

    async def insert_trace(self, row):
        try:
            r = await self.http.post(f"{self.base}/{S.TRACES['table']}", json=row,
                                     headers={**self.headers, "Prefer": "return=minimal"}, timeout=5)
            if r.status_code >= 400:
                log.warning("trace insert failed: HTTP %s %s", r.status_code, r.text[:200])
        except httpx.HTTPError as e:
            log.warning("trace insert failed: %s", e)

    async def ping(self):
        try:
            await self._get(S.DATA_VERSIONS["table"], {"select": "*", "limit": "1"}, timeout=3)
            return True
        except StoreError:
            return False
