"""Stage 5: write the answer from the supplied articles only, as JSON claims with quotes."""
from __future__ import annotations

from html import escape

from app.context import RequestCtx
from app.gateway.client import ModelGateway
from app.resources import Resources
from app.schemas.domain import Article
from app.schemas.llm import GeneratedAnswer, Understanding


def format_articles(articles: list[Article]) -> str:
    """Articles first and in a stable order, so repeated requests share a prompt prefix (provider caching)."""
    blocks = []
    for a in articles:
        attrs = (f'id="{escape(a.id)}" law="{escape(a.law_title)}" type="{escape(a.law_type)}" '
                 f'number="{escape(a.law_number)}" year="{escape(a.law_year)}" article_no="{escape(a.article_no)}" '
                 f'last_amended="{escape(a.last_amendment or "")}"')
        blocks.append(f"<article {attrs}>\n{a.text}\n</article>")
    return "\n".join(blocks)


def build_messages(masked_question: str, u: Understanding, articles: list[Article], res: Resources,
                   feedback: str | None = None, previous: GeneratedAnswer | None = None) -> list[dict]:
    issues = "، ".join(u.legal_issues) if u.legal_issues else "-"
    user = (
        f"<articles>\n{format_articles(articles)}\n</articles>\n\n"
        f"User question (as written, personal data masked):\n{masked_question}\n\n"
        f"Standard Arabic version:\n{u.msa_rewrite}\n\n"
        f"Legal issues: {issues}"
    )
    messages = [{"role": "system", "content": res.prompts.generate}, {"role": "user", "content": user}]
    if feedback:
        if previous is not None:
            messages.append({"role": "assistant", "content": previous.model_dump_json()})
        messages.append({"role": "user", "content": (
            "Your previous answer failed verification. Fix or remove these claims and return the full JSON again. "
            "Copy quotes exactly from the article text.\n" + feedback)})
    return messages


async def generate(masked_question: str, u: Understanding, articles: list[Article], gw: ModelGateway,
                   res: Resources, ctx: RequestCtx, feedback: str | None = None,
                   previous: GeneratedAnswer | None = None) -> GeneratedAnswer:
    ans = await gw.chat_json("generate", build_messages(masked_question, u, articles, res, feedback, previous),
                             GeneratedAnswer, ctx)
    ans.ensure_unique_ids()
    return ans
