"""Stage 7: plain code decides what the user gets.

Safety rules for the final answer:
* A rewrite is requested only on the first attempt and only when `time_left >= retry_min_s` (the orchestrator
  sets retry_min_s from this request's measured generate + verify times).
* A claim that failed verification is dropped only if it is NOT essential. If an essential claim failed, or more
  than half of the claims failed, the system abstains ("unverified") instead of showing an answer that could be
  incomplete or misleading.
* A summary or caveat that failed the answer-level number check is never shown: the summary is replaced by a
  neutral sentence and the caveat is removed. Dropped claims are announced with PARTIAL_NOTE, never silently.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app import messages as M
from app.schemas.domain import Article
from app.schemas.llm import GeneratedAnswer, Understanding
from app.stages.verify import VerificationReport

DecisionKind = Literal["deliver", "retry", "no_support", "unverified", "referral"]

# What to do when the semantic checker could not run. Open decision with the legal members:
#   "deliver_labelled" -> deliver claims that passed every code check, labelled for verification
#   "abstain"          -> treat as no support
SEMANTIC_SKIPPED_POLICY: Literal["deliver_labelled", "abstain"] = "deliver_labelled"
MAX_DROPPED_SHARE = 0.5   # more than this share of claims failing -> abstain, even if none is essential


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


def _without_failed_text(answer: GeneratedAnswer, report: VerificationReport) -> GeneratedAnswer:
    """Remove a summary or caveat that failed the answer-level check (never show unchecked numbers)."""
    parts = {p for p, _ in report.answer_failures}
    if not parts:
        return answer
    out = answer.model_copy(deep=True)
    if "summary" in parts:
        out.summary = M.NEUTRAL_SUMMARY
    out.caveats = [c for i, c in enumerate(answer.caveats) if f"caveat:{i}" not in parts]
    return out


def decide(answer: GeneratedAnswer, report: VerificationReport, u: Understanding, attempt: int,
           time_left: float, retry_min_s: float, articles: list[Article]) -> Decision:
    supplied = {a.id: a for a in articles}
    lawyer = answer.status == "needs_lawyer"

    if not answer.claims():
        if lawyer:
            return Decision("referral", reasons=["generator asked for a lawyer"])
        return Decision("no_support", reasons=["generator returned no claims"])

    if report.semantic_skipped and SEMANTIC_SKIPPED_POLICY == "abstain":
        return Decision("no_support", reasons=["semantic check unavailable"])

    if report.all_passed():
        return Decision("referral" if lawyer else "deliver", answer=answer,
                        amendment_warning=_amendment_warning(answer, u, supplied),
                        force_initial_label=report.semantic_skipped)

    if attempt == 1 and time_left >= retry_min_s:
        return Decision("retry", reasons=[f"{len(report.failed())} claim(s) failed"
                                          + (", answer-level check failed" if report.answer_failures else "")])

    # Final attempt (or no time for a rewrite): decide what can be shown safely.
    claims = {c.id: c for _, c in answer.claims()}
    failed = [chk.claim_id for chk in report.failed() if chk.claim_id in claims]
    essential_failed = [cid for cid in failed if claims[cid].essential]
    if essential_failed or len(failed) > MAX_DROPPED_SHARE * len(claims):
        why = (f"essential claim(s) failed: {', '.join(essential_failed)}" if essential_failed
               else f"{len(failed)} of {len(claims)} claims failed")
        if lawyer:
            return Decision("referral", reasons=[why, "generator asked for a lawyer"])
        return Decision("unverified", reasons=[why])

    kept = _without_failed_text(answer.keep_only(report.passed_ids()), report)
    if not kept.claims():                                   # defensive: covered by the share rule above
        return Decision("referral" if lawyer else "unverified", reasons=["no claim survived verification"])
    return Decision("referral" if lawyer else "deliver", answer=kept, partial=bool(failed),
                    amendment_warning=_amendment_warning(kept, u, supplied),
                    force_initial_label=report.semantic_skipped)
