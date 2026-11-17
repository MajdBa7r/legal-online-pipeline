"""Public API contract shared with the Web and Accuracy team.

Change this file only together with them: the website is built on this shape.
Sample responses live in docs/sample_responses/.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class Clarification(BaseModel):
    original_question: str = Field(..., max_length=2000)
    clarifying_question: str = Field(..., max_length=1000)
    answer: str = Field(..., max_length=1000)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    clarification: Clarification | None = None


class Citation(BaseModel):
    article_id: str
    quote: str
    label: str


class ClaimOut(BaseModel):
    text: str
    citation: Citation
    # the claim with its citation inline, ready to display: "<text> (<law title>، المادة <n>)".
    # Built from the LocalStore article, never from model text.
    display_text: str = ""


class AnswerSections(BaseModel):
    rights: list[ClaimOut] = Field(default_factory=list)
    obligations: list[ClaimOut] = Field(default_factory=list)
    steps: list[ClaimOut] = Field(default_factory=list)
    conditions: list[ClaimOut] = Field(default_factory=list)


class AnswerOut(BaseModel):
    summary: str
    sections: AnswerSections
    caveats: list[str] = Field(default_factory=list)
    partial: bool = False
    amendment_warning: bool = False
    # "verified": every shown claim passed the code checks AND the semantic check.
    # "not_semantically_verified": the semantic checker could not run; shown claims passed the code checks only
    # (real article, exact quote, numbers). The website must show this status visibly.
    verification: Literal["verified", "not_semantically_verified"] = "verified"


class Referral(BaseModel):
    reason: str


class DataVersionOut(BaseModel):
    version: str | None = None
    scraped_at: str | None = None


ResponseType = Literal["answer", "clarification", "referral", "no_support", "out_of_scope", "error"]


class AskResponse(BaseModel):
    request_id: str
    type: ResponseType
    domain: str | None = None
    coverage: Literal["focus", "initial"] | None = None
    answer: AnswerOut | None = None
    clarifying_question: str | None = None
    referral: Referral | None = None
    message: str | None = None
    data_version: DataVersionOut = Field(default_factory=DataVersionOut)
    timings_ms: dict[str, int] = Field(default_factory=dict)


class ArticleOut(BaseModel):
    id: str
    law_title: str
    type: str
    number: str | None = None
    year: str | None = None
    article_no: str
    text: str
    status: str
    last_amendment: str | None = None
    gazette: str | None = None
    source_url: str | None = None


class HealthOut(BaseModel):
    status: Literal["ok", "degraded", "unavailable"]
    database: bool
    models: dict[str, str]
    data_version: DataVersionOut
    pii_name_model: str = "none"  # added in Phase 3A (additive): "none", a model name, or why it is unavailable
