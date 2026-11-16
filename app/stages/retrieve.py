"""Stage 4: get the articles the generator may use.

4a  focus domain -> the in-memory domain pack (or a domain-filtered search if the pack is too big)
4b  other domain -> embed 3 wordings, hybrid_search each in parallel, RRF merge, rerank, expand
"""
from __future__ import annotations

import asyncio
import logging

from app.cache import LRU
from app.config import Settings
from app.context import RequestCtx
from app.db.store import Store
from app.errors import ModelCallError, StoreError
from app.gateway.client import ModelGateway
from app.resources import Resources
from app.schemas.domain import Article, SearchHit
from app.schemas.llm import Understanding
from app.text.arabic import build_fts_query, normalize

log = logging.getLogger(__name__)
RRF_K = 60
RERANK_DOC_CHARS = 4000


def rrf_merge(result_lists: list[list[SearchHit]], k: int = RRF_K) -> list[str]:
    """Reciprocal Rank Fusion across wordings, by article ID (an article's best chunk counts once per list)."""
    scores: dict[str, float] = {}
    for hits in result_lists:
        seen: set[str] = set()
        rank = 0
        for h in hits:
            if h.article_id in seen:
                continue
            seen.add(h.article_id)
            rank += 1
            scores[h.article_id] = scores.get(h.article_id, 0.0) + 1.0 / (k + rank)
    return [a for a, _ in sorted(scores.items(), key=lambda x: -x[1])]


def _wordings(masked_question: str, u: Understanding) -> list[str]:
    out: list[str] = []
    for w in (masked_question, u.msa_rewrite, u.legal_rewrite):
        w = (w or "").strip()
        if w and normalize(w) not in {normalize(x) for x in out}:
            out.append(w)
    return out


class Retriever:
    def __init__(self, gw: ModelGateway, store: Store, res: Resources, settings: Settings):
        self.gw, self.store, self.res, self.s = gw, store, res, settings
        self.embed_cache: LRU[str, list[float]] = LRU(1024)

    async def get_articles(self, masked_question: str, u: Understanding, domain: str, focus: bool,
                           ctx: RequestCtx) -> tuple[list[Article], str]:
        """Returns (articles, path) where path is 'pack' | 'pack_search' | 'search'."""
        if focus and self.res.packs.get(domain):
            if self.res.pack_fits(domain):
                return list(self.res.packs[domain]), "pack"
            arts = await self.search(masked_question, u, ctx, domain_filter=domain,
                                     top_n=self.s.pack_fallback_top_n)
            return arts, "pack_search"
        return await self.search(masked_question, u, ctx), "search"

    async def _embed(self, texts: list[str], ctx: RequestCtx) -> list[list[float] | None]:
        result: list[list[float] | None] = [self.embed_cache.get(t) for t in texts]
        todo = [i for i, v in enumerate(result) if v is None]
        if todo:
            try:
                vecs = await self.gw.embed([texts[i] for i in todo], ctx)
                for i, v in zip(todo, vecs):
                    result[i] = v
                    self.embed_cache.put(texts[i], v)
            except ModelCallError as e:
                log.warning("embedding failed, full-text only: %s", e)
                ctx.trace.set(embed_fallback=True)
        return result

    async def search(self, masked_question: str, u: Understanding, ctx: RequestCtx,
                     domain_filter: str | None = None, top_n: int | None = None) -> list[Article]:
        top_n = top_n or self.s.rerank_top_n
        wordings = _wordings(masked_question, u)

        with ctx.trace.stage("embed"):
            vectors = await self._embed(wordings, ctx)

        timeout = max(1.0, min(3.0, ctx.deadline.remaining() - 10))
        with ctx.trace.stage("hybrid_search"):
            results = await asyncio.gather(
                *(self.store.hybrid_search(build_fts_query(w), v, self.s.match_count, domain_filter, timeout)
                  for w, v in zip(wordings, vectors)),
                return_exceptions=True,
            )
        lists = [r for r in results if isinstance(r, list)]
        for r in results:
            if isinstance(r, Exception):
                log.warning("hybrid_search failed for one wording: %s", r)
        if not lists:
            raise StoreError("hybrid_search failed for every wording")

        candidate_ids = rrf_merge(lists)[: self.s.candidates]
        arts_by_id = await self.store.get_articles(candidate_ids)
        candidates = [arts_by_id[i] for i in candidate_ids if i in arts_by_id and arts_by_id[i].in_force()]
        ctx.trace.set(retrieved_ids=[a.id for a in candidates])
        if not candidates:
            return []

        ranked = await self._rerank(masked_question, u, candidates, top_n, ctx)
        return await self._expand(ranked, ctx)

    async def _rerank(self, masked_question, u, candidates: list[Article], top_n: int, ctx) -> list[Article]:
        query = u.legal_rewrite if self.s.rerank_query == "legal_rewrite" else masked_question
        docs = [f"{a.header()}\n{a.text[:RERANK_DOC_CHARS]}" for a in candidates]
        try:
            with ctx.trace.stage("rerank"):
                hits = await self.gw.rerank(query, docs, top_n, ctx)
            ctx.trace.set(rerank_fallback=False)
            return [candidates[h.index] for h in hits if 0 <= h.index < len(candidates)]
        except ModelCallError as e:
            log.warning("rerank failed, keeping RRF order: %s", e)
            ctx.trace.set(rerank_fallback=True)
            return candidates[:top_n]

    async def _expand(self, ranked: list[Article], ctx: RequestCtx) -> list[Article]:
        if self.s.expansion_max <= 0 or not ranked:
            return ranked
        have = {a.id for a in ranked}
        try:
            def_ids = await self.store.definitions_for(list(dict.fromkeys(a.legislation_id for a in ranked)))
            new_ids = [i for i in def_ids if i not in have][: self.s.expansion_max]
            if not new_ids:
                return ranked
            extra = await self.store.get_articles(new_ids)
            added = [extra[i] for i in new_ids if i in extra and extra[i].in_force()]
            ctx.trace.set(expanded_ids=[a.id for a in added])
            return ranked + added
        except StoreError as e:
            log.warning("expansion skipped: %s", e)
            return ranked
