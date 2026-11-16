"""Stage 3: plain code + the legal members' rules file. First match wins."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.resources import GateRules
from app.schemas.domain import FOCUS_DOMAINS
from app.schemas.llm import Understanding
from app.text.arabic import normalize

GateKind = Literal["continue", "out_of_scope", "referral", "clarify"]


@dataclass
class GateDecision:
    kind: GateKind
    domain: str             # effective domain after the confidence check
    focus: bool             # True -> stage 4a (domain pack), False -> 4b (search)
    reason: str = ""

    @property
    def stop(self) -> bool:
        return self.kind != "continue"


def decide_gate(u: Understanding, masked_question: str, rules: GateRules, clarification_round: bool) -> GateDecision:
    domain = u.domain
    if domain in FOCUS_DOMAINS and u.domain_confidence < rules.min_domain_confidence:
        domain = "other"
    focus = domain in FOCUS_DOMAINS

    if not u.is_legal_question:
        return GateDecision("out_of_scope", domain, focus, "not a legal question")

    q = normalize(masked_question)
    hit = next((k for k in rules.keywords_norm if k and k in q), None)
    if u.risk_level in rules.referral_risk_levels or hit:
        reason = "; ".join(u.risk_reasons) if u.risk_reasons else (f"keyword: {hit}" if hit else "high risk")
        return GateDecision("referral", domain, focus, reason)

    blocking = [m for m in u.missing_facts if m.blocking]
    if (rules.clarify_on_blocking_missing_fact and blocking and not clarification_round
            and u.clarifying_question):
        return GateDecision("clarify", domain, focus, blocking[0].fact)

    return GateDecision("continue", domain, focus)
