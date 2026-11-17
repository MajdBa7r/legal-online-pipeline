"""Output schema for each model role. The gateway validates every reply against one of these."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["labour", "rentals", "consumer_protection", "social_security", "other"]
SECTION_NAMES = ("rights", "obligations", "steps", "conditions")


# ---- role: understand -------------------------------------------------------
class MissingFact(BaseModel):
    fact: str
    blocking: bool = False


class Understanding(BaseModel):
    is_legal_question: bool = True
    msa_rewrite: str
    legal_rewrite: str
    domain: Domain = "other"
    domain_confidence: float = Field(0.0, ge=0.0, le=1.0)
    other_domain_hint: str | None = None
    legal_issues: list[str] = Field(default_factory=list)
    missing_facts: list[MissingFact] = Field(default_factory=list)
    clarifying_question: str | None = None
    risk_level: Literal["low", "medium", "high"] = "low"
    risk_reasons: list[str] = Field(default_factory=list)
    event_year: int | None = None


# ---- role: generate ---------------------------------------------------------
class Claim(BaseModel):
    id: str
    text: str
    article_id: str
    quote: str
    # True when the direct answer depends on this claim (the main rule, or a condition/exception that changes the
    # outcome for this user). Missing -> True: an unmarked claim is never assumed safe to drop.
    essential: bool = True


class Sections(BaseModel):
    rights: list[Claim] = Field(default_factory=list)
    obligations: list[Claim] = Field(default_factory=list)
    steps: list[Claim] = Field(default_factory=list)
    conditions: list[Claim] = Field(default_factory=list)


class GeneratedAnswer(BaseModel):
    status: Literal["answered", "no_support", "needs_lawyer"]
    summary: str = ""
    sections: Sections = Field(default_factory=Sections)
    caveats: list[str] = Field(default_factory=list)

    def claims(self) -> list[tuple[str, Claim]]:
        return [(name, c) for name in SECTION_NAMES for c in getattr(self.sections, name)]

    def ensure_unique_ids(self) -> None:
        seen: set[str] = set()
        n = 0
        for _, c in self.claims():
            n += 1
            if not c.id or c.id in seen:
                c.id = f"c{n}"
                while c.id in seen:
                    n += 1
                    c.id = f"c{n}"
            seen.add(c.id)

    def keep_only(self, claim_ids: set[str]) -> "GeneratedAnswer":
        copy = self.model_copy(deep=True)
        for name in SECTION_NAMES:
            setattr(copy.sections, name, [c for c in getattr(copy.sections, name) if c.id in claim_ids])
        return copy


# ---- role: verify -----------------------------------------------------------
class ClaimVerdict(BaseModel):
    id: str
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str = ""


class VerifierOutput(BaseModel):
    verdicts: list[ClaimVerdict] = Field(default_factory=list)
