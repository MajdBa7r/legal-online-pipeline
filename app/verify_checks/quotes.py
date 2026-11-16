"""Quote check: the cited quote must be a real span of the article's original text."""
from __future__ import annotations

from dataclasses import dataclass

from app.text.arabic import collapse_ws, normalize

_STRIP = " \t\n\"'«»“”„‚‘’.،,:؛;…"


@dataclass
class QuoteResult:
    exact: bool        # passes the rule in force: substring after collapsing whitespace only
    normalized: bool   # would pass if we compared normalized text (logged to decide strictness)
    too_short: bool


def check_quote(quote: str, article_text: str, min_chars: int = 8) -> QuoteResult:
    q = collapse_ws(quote).strip(_STRIP)
    if len(q) < min_chars:
        return QuoteResult(exact=False, normalized=False, too_short=True)
    exact = q in collapse_ws(article_text)
    normalized = exact or normalize(q) in normalize(article_text)
    return QuoteResult(exact=exact, normalized=normalized, too_short=False)
