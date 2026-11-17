"""Test doubles: a scripted gateway and an in-memory store.

The articles below are INVENTED fixtures for tests only; they are not real law text.
"""
from __future__ import annotations

import asyncio

from typing import Any, Callable

from app.config import Settings
from app.errors import ModelCallError, PIILeakError
from app.gateway.client import RerankHit
from app.pii.masker import PIIMasker
from app.resources import GateRules, Prompts, Resources
from app.schemas.domain import Article, DataVersion, GlossaryEntry, SearchHit
from app.text.arabic import normalize

LEAVE = Article(id="L1", article_no="61", law_title="قانون تجريبي للعمل", law_type="law", law_number="1",
                law_year="2000", legislation_id="LAW1", domain="labour", last_amendment="2019",
                text="يستحق العامل إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً عن كل سنة خدمة.")
NOTICE = Article(id="L2", article_no="23", law_title="قانون تجريبي للعمل", law_type="law", law_number="1",
                 law_year="2000", legislation_id="LAW1", domain="labour", last_amendment="2019",
                 text="على الطرف الذي يرغب في إنهاء عقد العمل غير محدود المدة أن يشعر الطرف الآخر خطياً قبل شهر واحد على الأقل.")
DEFS = Article(id="L0", article_no="2", law_title="قانون تجريبي للعمل", law_type="law", law_number="1",
               law_year="2000", legislation_id="LAW1", domain="labour",
               text="العامل: كل شخص ذكراً كان أو أنثى يؤدي عملاً لقاء أجر.")
TRAFFIC = Article(id="T1", article_no="10", law_title="قانون تجريبي للسير", law_type="law", law_number="5",
                  law_year="2008", legislation_id="LAW2", domain=None,
                  text="يعاقب كل من قاد مركبة دون رخصة قيادة سارية بغرامة لا تقل عن خمسين ديناراً.")
REPEALED = Article(id="R1", article_no="3", law_title="قانون ملغى", legislation_id="LAW3", law_status="repealed",
                   text="نص مادة ملغاة لا يجوز الاستناد إليها في أي إجابة.")

ALL = {a.id: a for a in (LEAVE, NOTICE, DEFS, TRAFFIC, REPEALED)}


class FakeStore:
    def __init__(self, search_ids: list[str] | None = None):
        self.search_ids = search_ids if search_ids is not None else ["T1", "R1"]
        self.traces: list[dict] = []
        self.search_calls: list[dict] = []

    async def hybrid_search(self, query_text, embedding, match_count, domain=None, timeout=3.0):
        self.search_calls.append({"q": query_text, "has_embedding": embedding is not None, "domain": domain})
        return [SearchHit(article_id=i, score=1.0 / (n + 1)) for n, i in enumerate(self.search_ids)]

    async def get_articles(self, ids):
        return {i: ALL[i] for i in ids if i in ALL}

    async def load_domain_packs(self):
        return {"labour": ["L1", "L2"]}

    async def load_glossary(self):
        return [GlossaryEntry(phrase="فنشني", legal_term="إنهاء عقد العمل", domain="labour",
                              phrase_norm=normalize("فنشني"))]

    async def definitions_for(self, legislation_ids):
        return ["L0"] if "LAW1" in legislation_ids else []

    async def latest_data_version(self):
        return DataVersion(version="test-1", scraped_at="2026-10-13")

    async def insert_trace(self, row):
        self.traces.append(row)

    async def ping(self):
        return True


Script = Callable[[list[dict], int], Any]


class Delay:
    """Scripted reply that takes `seconds` before returning `item` (or raising it, if it is an Exception)."""

    def __init__(self, seconds: float, item: Any):
        self.seconds, self.item = seconds, item


class FakeGateway:
    """Each role returns the next scripted item: a pydantic object, a dict, or an Exception to raise."""

    def __init__(self, scripts: dict[str, list[Any]]):
        self.scripts = {k: list(v) for k, v in scripts.items()}
        self.calls: list[tuple[str, list[dict]]] = []
        self.status = {r: "unknown" for r in ("understand", "generate", "verify", "embed", "rerank")}

    def _guard(self, texts, ctx):
        if PIIMasker.leaked(texts, ctx.pii_values):
            raise PIILeakError("leak")

    def _next(self, role):
        queue = self.scripts.get(role) or []
        if not queue:
            raise ModelCallError(f"no scripted reply for {role}")
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def chat_json(self, role, messages, schema, ctx):
        self.calls.append((role, messages))
        self._guard([m["content"] for m in messages if m["role"] != "system"], ctx)
        queue = self.scripts.get(role) or []
        if queue and isinstance(queue[0], Delay):
            d = queue.pop(0)
            await asyncio.sleep(d.seconds)
            queue.insert(0, d.item)
        item = self._next(role)
        return item if isinstance(item, schema) else schema.model_validate(item)

    async def embed(self, texts, ctx):
        self.calls.append(("embed", [{"content": t} for t in texts]))
        self._guard(texts, ctx)
        item = self._next("embed") if self.scripts.get("embed") else None
        if item is None:
            return [[0.0] * 4 for _ in texts]
        return item

    async def rerank(self, query, docs, top_n, ctx):
        self.calls.append(("rerank", [{"content": query}]))
        if self.scripts.get("rerank"):
            self._next("rerank")
        return [RerankHit(index=i, score=1.0 - i / 10) for i in range(min(top_n, len(docs)))]


def make_resources(settings: Settings | None = None) -> Resources:
    s = settings or Settings()
    return Resources(settings=s, prompts=Prompts.from_dir(s.prompts_dir), rules=GateRules.from_file(s.rules_file))
