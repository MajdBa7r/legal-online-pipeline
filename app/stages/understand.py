"""Stage 2: one call turns the masked dialect question into rewrites + gate facts."""
from __future__ import annotations

import json
import logging

from app.context import RequestCtx
from app.errors import ModelCallError, ModelOutputError
from app.gateway.client import ModelGateway
from app.resources import Resources
from app.schemas.llm import Understanding

log = logging.getLogger(__name__)
DOMAINS = '"labour", "rentals", "consumer_protection", "social_security", "other"'


def build_messages(masked_question: str, res: Resources) -> list[dict]:
    hits = res.glossary_hits(masked_question)
    glossary = "\n".join(f"- {g.phrase} -> {g.legal_term}" for g in hits) or "(none)"
    system = res.prompts.understand.replace("{{DOMAINS}}", DOMAINS).replace("{{GLOSSARY}}", glossary)
    messages = [{"role": "system", "content": system}]
    for ex in res.prompts.examples:
        messages.append({"role": "user", "content": ex["question"]})
        messages.append({"role": "assistant", "content": json.dumps(ex["output"], ensure_ascii=False)})
    messages.append({"role": "user", "content": masked_question})
    return messages


def fallback(masked_question: str) -> Understanding:
    """Used when the model fails: the request still goes through the search path."""
    return Understanding(
        is_legal_question=True,
        msa_rewrite=masked_question,
        legal_rewrite=masked_question,
        domain="other",
        domain_confidence=0.0,
        risk_level="low",
    )


async def understand(masked_question: str, gw: ModelGateway, res: Resources, ctx: RequestCtx) -> Understanding:
    try:
        u = await gw.chat_json("understand", build_messages(masked_question, res), Understanding, ctx)
        ctx.trace.set(understand_fallback=False)
        return u
    except (ModelCallError, ModelOutputError) as e:
        log.warning("understand failed, using fallback: %s", e)
        ctx.trace.set(understand_fallback=True)
        return fallback(masked_question)
