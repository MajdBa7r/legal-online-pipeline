"""Stage 7: plain code decides what the user gets."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.schemas.domain import Article
from app.schemas.llm import GeneratedAnswer, Understanding
from app.stages.verify import VerificationReport

DecisionKind = Literal["deliver", "retry", "no_support", "referral"]

# What to do when the semantic checker could not run. Open decision with the legal members:
#   "deliver_labelled" -> deliver claims that passed every code check, labelled for verification
#   "abstain"          -> treat as no support
SEMANTIC_SKIPPED_POLICY: Literal["deliver_labelled", "abstain"] = "deliver_labelled"


@dataclass
class Decision:
    kind: DecisionKind
    answer: GeneratedAnswer | None = None
    partial: bool = False
    amendment_warning: bool = False
    force_initial_label: bool = False
    reasons: list[str] = field(default_factory=list)


def _amendment_warning(answer: GeneratedAnswer, u: Understanding, supplied: dict[str, Article]) -> bool:
    """The user's situation happened before the last amendment of a cited law."""
    if not u.event_year:
        return False
    for _, c in answer.claims():
        art = supplied.get(c.article_id)
        year = art.amendment_year() if art else None
        if year and u.event_year < year:
            return True
    return False


def decide(answer: GeneratedAnswer, report: VerificationReport, u: Understanding, attempt: int,
           time_left: float, retry_min_s: float, articles: list[Article]) -> Decision:
    supplied = {a.id: a for a in articles}

    if not answer.claims():
        if answer.status == "needs_lawyer":
            return Decision("referral", reasons=["generator asked for a lawyer"])
        return Decision("no_support", reasons=["generator returned no claims"])

    if report.semantic_skipped and SEMANTIC_SKIPPED_POLICY == "abstain":
        return Decision("no_support", reasons=["semantic check unavailable"])

    if report.all_passed():
        kind: DecisionKind = "referral" if answer.status == "needs_lawyer" else "deliver"
        return Decision(kind, answer=answer, amendment_warning=_amendment_warning(answer, u, supplied),
                        force_initial_label=report.semantic_skipped)

    if attempt == 1 and time_left >= retry_min_s:
        return Decision("retry", reasons=[f"{len(report.failed())} claim(s) failed"])

    kept = answer.keep_only(report.passed_ids())
    if not kept.claims():
        if answer.status == "needs_lawyer":
            return Decision("referral", reasons=["no verified claim; generator asked for a lawyer"])
        return Decision("no_support", reasons=["no claim survived verification"])
    kind = "referral" if answer.status == "needs_lawyer" else "deliver"
    return Decision(kind, answer=kept, partial=True, amendment_warning=_amendment_warning(kept, u, supplied),
                    force_initial_label=report.semantic_skipped)
