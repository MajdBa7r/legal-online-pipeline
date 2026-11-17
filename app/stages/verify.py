"""Stage 6: code checks on every claim, then one semantic check call for the claims that pass."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.config import Settings
from app.context import RequestCtx
from app.errors import ModelCallError, ModelOutputError
from app.gateway.client import ModelGateway
from app.resources import Resources
from app.schemas.domain import Article
from app.schemas.llm import Claim, GeneratedAnswer, VerifierOutput
from app.verify_checks.numbers import numbers_supported
from app.verify_checks.quotes import check_quote

log = logging.getLogger(__name__)


@dataclass
class ClaimCheck:
    claim_id: str
    failures: list[str] = field(default_factory=list)   # machine codes: not_supplied, quote_mismatch, ...
    reasons: list[str] = field(default_factory=list)    # human-readable, fed back to the generator
    quote_normalized_only: bool = False
    semantic: str | None = None                         # supported | partial | unsupported | None (not run)

    @property
    def passed(self) -> bool:
        return not self.failures and self.semantic in (None, "supported")


SUMMARY_ID = "summary"   # id of the summary item in the checker request (claim ids are c1, c2, ...)


@dataclass
class VerificationReport:
    checks: dict[str, ClaimCheck]
    semantic_skipped: bool = False
    # checker verdict on the summary: "supported" | "partial" | "unsupported" | None (not checked / no verdict).
    # Anything but "supported" means the summary is replaced by a neutral sentence (app/stages/decide.py).
    summary_verdict: str | None = None
    summary_reason: str = ""
    # answer-level problems as (part, message); part is "summary" or "caveat:<index>"
    answer_failures: list[tuple[str, str]] = field(default_factory=list)

    def passed_ids(self) -> set[str]:
        return {cid for cid, c in self.checks.items() if c.passed}

    def failed(self) -> list[ClaimCheck]:
        return [c for c in self.checks.values() if not c.passed]

    def all_passed(self) -> bool:
        return bool(self.checks) and not self.failed() and not self.answer_failures

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"claims": len(self.checks), "passed": len(self.passed_ids()),
                               "answer_failures": len(self.answer_failures)}
        for c in self.checks.values():
            for f in c.failures:
                out[f] = out.get(f, 0) + 1
            if c.quote_normalized_only:
                out["quote_normalized_only"] = out.get("quote_normalized_only", 0) + 1
        return out

    def feedback(self, answer: GeneratedAnswer) -> str:
        by_id = {c.id: c for _, c in answer.claims()}
        lines = []
        for chk in self.failed():
            claim = by_id.get(chk.claim_id)
            text = claim.text if claim else ""
            lines.append(f"- {chk.claim_id} ({text[:120]}): {'; '.join(chk.reasons)}")
        lines += [f"- {part}: {msg}" for part, msg in self.answer_failures]
        return "\n".join(lines)


def code_checks(claim: Claim, supplied: dict[str, Article], settings: Settings) -> ClaimCheck:
    chk = ClaimCheck(claim.id)
    art = supplied.get(claim.article_id)
    if art is None:
        chk.failures.append("not_supplied")
        chk.reasons.append(f"article {claim.article_id} was not among the supplied articles")
        return chk
    q = check_quote(claim.quote, art.text, settings.min_quote_chars)
    if q.too_short:
        chk.failures.append("quote_too_short")
        chk.reasons.append("the quote is empty or too short")
    elif not q.exact:
        chk.failures.append("quote_mismatch")
        chk.reasons.append("the quote is not an exact span of the article text")
        chk.quote_normalized_only = q.normalized
    ok, missing = numbers_supported(claim.text, art.text, art.metadata_numbers())
    if not ok:
        chk.failures.append("number_mismatch")
        chk.reasons.append("numbers not found in the article: " + ", ".join(f"{m:g}" for m in missing))
    if not art.in_force():
        chk.failures.append("not_in_force")
        chk.reasons.append("the article or its law is not in force")
    return chk


def _semantic_messages(items: list[tuple[Claim, Article]], res: Resources, summary: str = "",
                       question: str = "") -> list[dict]:
    """One request: every UNIQUE article once, then each claim pointing to its article by id, then the summary.

    Each claim still gets its own verdict against its own article (see app/prompts/verify.md)."""
    articles: dict[str, Article] = {}
    for _, a in items:
        articles.setdefault(a.id, a)
    payload: dict = {
        "articles": [{"id": a.id, "text": f"<article id=\"{a.id}\">{a.text}</article>"} for a in articles.values()],
        "claims": [{"id": c.id, "article_id": a.id, "claim": c.text, "quote": c.quote} for c, a in items],
    }
    if summary.strip():
        payload["summary"] = {"id": SUMMARY_ID, "question": question, "text": summary,
                              "claim_ids": [c.id for c, _ in items]}
    return [
        {"role": "system", "content": res.prompts.verify},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def answer_level_checks(answer: GeneratedAnswer, supplied: dict[str, Article],
                        question: str) -> list[tuple[str, str]]:
    """Summary and caveats carry no claims of their own, but they may restate the user's facts next to the rules.
    Every number in them must come from the user's question or from an article cited by a claim."""
    cited = [supplied[c.article_id] for _, c in answer.claims() if c.article_id in supplied]
    source = "\n".join([question] + [a.text for a in cited])
    allowed: set[float] = set()
    for a in cited:
        allowed |= a.metadata_numbers()
    failures = []
    for part, text in [("summary", answer.summary)] + [(f"caveat:{i}", c) for i, c in enumerate(answer.caveats)]:
        ok, missing = numbers_supported(text, source, allowed)
        if not ok:
            failures.append((part, "numbers in neither the question nor a cited article: "
                             + ", ".join(f"{m:g}" for m in missing)))
    return failures


async def verify(answer: GeneratedAnswer, articles: list[Article], gw: ModelGateway, res: Resources,
                 settings: Settings, ctx: RequestCtx, question: str = "") -> VerificationReport:
    supplied = {a.id: a for a in articles}
    checks: dict[str, ClaimCheck] = {}
    to_semantic: list[tuple[Claim, Article]] = []
    for _, claim in answer.claims():
        chk = code_checks(claim, supplied, settings)
        checks[claim.id] = chk
        if not chk.failures:
            to_semantic.append((claim, supplied[claim.article_id]))

    report = VerificationReport(checks, answer_failures=answer_level_checks(answer, supplied, question))
    if not to_semantic:
        return report
    try:
        out = await gw.chat_json("verify", _semantic_messages(to_semantic, res, answer.summary, question),
                                 VerifierOutput, ctx)
        verdicts = {v.id: v for v in out.verdicts}
        if (sv := verdicts.get(SUMMARY_ID)) is not None:
            report.summary_verdict, report.summary_reason = sv.verdict, sv.reason
        for claim, _ in to_semantic:
            v = verdicts.get(claim.id)
            chk = checks[claim.id]
            if v is None:
                chk.semantic = "unsupported"
                chk.failures.append("semantic_missing")
                chk.reasons.append("the checker returned no verdict for this claim")
            else:
                chk.semantic = v.verdict
                if v.verdict != "supported":
                    chk.failures.append(f"semantic_{v.verdict}")
                    chk.reasons.append(v.reason or f"checker verdict: {v.verdict}")
    except (ModelCallError, ModelOutputError) as e:
        log.warning("semantic verification skipped: %s", e)
        report.semantic_skipped = True
    return report
