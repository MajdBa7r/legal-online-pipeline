"""Runs one request through the seven stages inside the time budget."""
from __future__ import annotations

import asyncio
import logging
import time

from app import messages as M
from app.cache import LRU
from app.config import Settings
from app.context import Deadline, RequestCtx
from app.db.store import Store
from app.errors import ModelCallError, ModelOutputError, PIILeakError, StoreError
from app.gateway.client import ModelGateway
from app.pii.masker import MaskResult, PIIMasker
from app.resources import Resources
from app.schemas.api import (AnswerOut, AnswerSections, AskRequest, AskResponse, Citation, ClaimOut,
                             DataVersionOut, Referral)
from app.schemas.domain import Article
from app.schemas.llm import SECTION_NAMES, GeneratedAnswer
from app.stages.decide import Decision, decide
from app.stages.gate import GateDecision, decide_gate
from app.stages.generate import generate
from app.stages.retrieve import Retriever
from app.stages.understand import understand
from app.stages.verify import verify
from app.text.arabic import normalize

log = logging.getLogger(__name__)


def compose_question(req: AskRequest) -> str:
    """Clarification is stateless: the website sends back the original question and the user's answer."""
    if not req.clarification:
        return req.question
    c = req.clarification
    return f"{c.original_question}\nسؤال توضيحي: {c.clarifying_question}\nجواب المستخدم: {c.answer}"


class Pipeline:
    def __init__(self, gw: ModelGateway, store: Store, res: Resources, masker: PIIMasker, settings: Settings):
        self.gw, self.store, self.res, self.masker, self.s = gw, store, res, masker, settings
        self.retriever = Retriever(gw, store, res, settings)
        self.demo_cache: LRU[tuple, AskResponse] = LRU(256)

    # ---------------------------------------------------------------------
    async def handle(self, req: AskRequest, ctx: RequestCtx | None = None) -> AskResponse:
        ctx = ctx or RequestCtx(deadline=Deadline(self.s.budget_s))
        await self.res.refresh_if_stale(self.store)
        mask: MaskResult | None = None
        try:
            with ctx.trace.stage("mask"):
                mask = self.masker.mask(compose_question(req))
            ctx.pii_values = mask.originals
            ctx.trace.set(pii_count=len(mask.mapping), clarification_round=req.clarification is not None)

            cache_key = (normalize(mask.text), self._version())
            if self.s.demo_cache and (cached := self.demo_cache.get(cache_key)):
                ctx.trace.set(outcome="cache_hit")
                return self._restore(cached.model_copy(update={"request_id": ctx.request_id}), mask)

            try:   # hard backstop: the whole request, retries and rewrite included, ends at the deadline
                resp = await asyncio.wait_for(self._run(mask, req, ctx),
                                              timeout=max(0.1, ctx.deadline.remaining()))
            except asyncio.TimeoutError:
                log.warning("request hit the %.0fs budget", self.s.budget_s)
                ctx.trace.set(outcome="timeout", error="budget")
                return self._simple(ctx, "error", M.TIMEOUT)
            if self.s.demo_cache and resp.type == "answer":
                self.demo_cache.put(cache_key, resp)
            return self._restore(resp, mask)
        except PIILeakError as e:
            log.error("PII guard blocked a call: %s", e)
            ctx.trace.set(outcome="error", error="pii_guard")
            return self._simple(ctx, "error", M.ERROR)
        except Exception as e:  # last line of defence; never leak internals to the user
            log.exception("pipeline error")
            ctx.trace.set(outcome="error", error=type(e).__name__)
            return self._simple(ctx, "error", M.ERROR)
        finally:
            ctx.trace.set(total_ms=ctx.deadline.elapsed_ms())

    # ---------------------------------------------------------------------
    async def _run(self, mask: MaskResult, req: AskRequest, ctx: RequestCtx) -> AskResponse:
        q = mask.text

        await ctx.emit("understanding")
        with ctx.trace.stage("understand"):
            u = await understand(q, self.gw, self.res, ctx)

        gate: GateDecision = decide_gate(u, q, self.res.rules, clarification_round=req.clarification is not None)
        ctx.trace.set(domain=gate.domain, model_domain=u.domain, gate=gate.kind, risk=u.risk_level)
        if gate.kind == "out_of_scope":
            return self._simple(ctx, "out_of_scope", M.OUT_OF_SCOPE)
        if gate.kind == "referral":
            ctx.trace.set(outcome="referral")
            return self._simple(ctx, "referral", None, domain=gate.domain, referral=Referral(reason=M.REFERRAL))
        if gate.kind == "clarify":
            ctx.trace.set(outcome="clarification")
            return self._simple(ctx, "clarification", None, domain=gate.domain,
                                clarifying_question=u.clarifying_question)

        await ctx.emit("searching")
        try:
            with ctx.trace.stage("retrieve"):
                articles, path = await self.retriever.get_articles(q, u, gate.domain, gate.focus, ctx)
        except StoreError as e:
            log.error("retrieval failed: %s", e)
            ctx.trace.set(outcome="error", error="retrieval")
            return self._simple(ctx, "error", M.ERROR)
        ctx.trace.set(path=path, supplied_ids=[a.id for a in articles][:200])
        if not articles:
            ctx.trace.set(outcome="no_support")
            return self._simple(ctx, "no_support", M.NO_SUPPORT, domain=gate.domain)

        coverage = "focus" if gate.focus else "initial"
        feedback: str | None = None
        previous: GeneratedAnswer | None = None
        decision: Decision | None = None
        first: tuple[GeneratedAnswer, object] | None = None      # attempt 1 (answer, report), kept for a fallback
        for attempt in (1, 2):
            await ctx.emit("writing")
            t0 = time.monotonic()
            try:
                with ctx.trace.stage("generate"):
                    ans = await generate(q, u, articles, self.gw, self.res, ctx, feedback, previous)
            except (ModelCallError, ModelOutputError) as e:
                log.warning("generation attempt %d failed: %s", attempt, e)
                ctx.trace.set(generations=attempt)
                if first is None:
                    ctx.trace.set(outcome="timeout")
                    return self._simple(ctx, "error", M.TIMEOUT, domain=gate.domain)
                # the rewrite failed: decide on the first answer, with no further rewrite
                decision = decide(first[0], first[1], u, 2, 0.0, float("inf"), articles)
                ctx.trace.set(rewrite_failed=True)
                break
            gen_s = time.monotonic() - t0

            await ctx.emit("checking")
            t1 = time.monotonic()
            with ctx.trace.stage("verify"):
                report = await verify(ans, articles, self.gw, self.res, self.s, ctx,
                                      question=f"{q}\n{u.msa_rewrite}")
            ver_s = time.monotonic() - t1
            # a rewrite costs about another generate + verify: require that (with margin) to be left
            retry_min = self.retry_needed_s(gen_s, ver_s)
            decision = decide(ans, report, u, attempt, ctx.deadline.remaining(), retry_min, articles)
            ctx.trace.set(generations=attempt, verification=report.counts(),
                          semantic_skipped=report.semantic_skipped, retry_min_s=round(retry_min, 1))
            if decision.kind != "retry":
                break
            first = (ans, report)
            feedback, previous = report.feedback(ans), ans

        assert decision is not None
        ctx.trace.set(outcome=decision.kind, partial=decision.partial,
                      cited_ids=[c.article_id for _, c in decision.answer.claims()] if decision.answer else [])
        if decision.kind == "no_support":
            return self._simple(ctx, "no_support", M.NO_SUPPORT, domain=gate.domain)
        if decision.kind == "unverified":          # abstain rather than show an incomplete or misleading answer
            return self._simple(ctx, "no_support", M.UNVERIFIED, domain=gate.domain)
        if decision.force_initial_label:
            coverage = "initial"
        answer = self._answer_out(decision, articles) if decision.answer else None
        if decision.kind == "referral":
            return self._simple(ctx, "referral", None, domain=gate.domain, coverage=coverage,
                                referral=Referral(reason=M.REFERRAL), answer=answer)
        return self._simple(ctx, "answer", None, domain=gate.domain, coverage=coverage, answer=answer)

    def retry_needed_s(self, gen_s: float, ver_s: float) -> float:
        """Seconds that must be left before a rewrite starts: this request's own generate + verify times with a
        25% margin plus the gateway reserve, and never less than RETRY_MIN_REMAINING_S."""
        return max(self.s.retry_min_remaining_s, 1.25 * (gen_s + ver_s) + self.s.reserve_s)

    # ---------------------------------------------------------------------
    def _answer_out(self, d: Decision, articles: list[Article]) -> AnswerOut:
        by_id = {a.id: a for a in articles}
        sections = AnswerSections()
        for name in SECTION_NAMES:
            items = []
            for c in getattr(d.answer.sections, name):
                art = by_id.get(c.article_id)
                items.append(ClaimOut(text=c.text, citation=Citation(
                    article_id=c.article_id, quote=c.quote, label=art.label() if art else c.article_id)))
            setattr(sections, name, items)
        caveats = list(d.answer.caveats)
        if d.partial:
            caveats.append(M.PARTIAL_NOTE)
        if d.amendment_warning:
            caveats.append(M.AMENDMENT_NOTE)
        if d.force_initial_label:
            caveats.append(M.SEMANTIC_SKIPPED_NOTE)
        return AnswerOut(summary=d.answer.summary, sections=sections, caveats=caveats,
                         partial=d.partial, amendment_warning=d.amendment_warning)

    def _version(self) -> str | None:
        return self.res.data_version.version if self.res.data_version else None

    def _simple(self, ctx: RequestCtx, type_: str, message: str | None, **kw) -> AskResponse:
        dv = self.res.data_version
        return AskResponse(
            request_id=ctx.request_id, type=type_, message=message,
            data_version=DataVersionOut(version=dv.version if dv else None, scraped_at=dv.scraped_at if dv else None),
            timings_ms={**ctx.trace.timings_ms, "total": ctx.deadline.elapsed_ms()},
            **kw,
        )

    def _restore(self, resp: AskResponse, mask: MaskResult) -> AskResponse:
        """Put the user's personal data back, only in text the user reads (never in quotes from the law)."""
        if not mask.mapping:
            return resp
        r = self.masker.restore
        out = resp.model_copy(deep=True)
        out.clarifying_question = r(out.clarifying_question, mask.mapping)
        out.message = r(out.message, mask.mapping)
        if out.answer:
            out.answer.summary = r(out.answer.summary, mask.mapping)
            out.answer.caveats = [r(c, mask.mapping) for c in out.answer.caveats]
            for name in SECTION_NAMES:
                for item in getattr(out.answer.sections, name):
                    item.text = r(item.text, mask.mapping)
        return out
