from app.resources import GateRules
from app.schemas.llm import Claim, GeneratedAnswer, MissingFact, Sections, Understanding
from app.stages.decide import decide
from app.stages.gate import decide_gate
from app.stages.verify import ClaimCheck, VerificationReport
from app.config import Settings
from fakes import LEAVE, NOTICE

RULES = GateRules.from_file(Settings().rules_file)


def U(**kw):
    base = dict(msa_rewrite="س", legal_rewrite="س", domain="labour", domain_confidence=0.9)
    base.update(kw)
    return Understanding(**base)


def test_gate_order():
    assert decide_gate(U(is_legal_question=False), "مرحبا", RULES, False).kind == "out_of_scope"
    assert decide_gate(U(risk_level="high"), "سؤال", RULES, False).kind == "referral"
    assert decide_gate(U(), "ابني موقوف في المخفر", RULES, False).kind == "referral"
    u = U(missing_facts=[MissingFact(fact="نوع العقد", blocking=True)], clarifying_question="ما نوع عقدك؟")
    assert decide_gate(u, "سؤال", RULES, False).kind == "clarify"
    assert decide_gate(u, "سؤال", RULES, clarification_round=True).kind == "continue"


def test_gate_low_confidence_goes_to_search():
    g = decide_gate(U(domain_confidence=0.3), "سؤال", RULES, False)
    assert g.kind == "continue" and g.domain == "other" and not g.focus
    g = decide_gate(U(), "سؤال", RULES, False)
    assert g.focus and g.domain == "labour"


def _answer(*claims, status="answered"):
    return GeneratedAnswer(status=status, summary="ملخص", sections=Sections(rights=list(claims)))


def _report(**verdicts):
    checks = {}
    for cid, ok in verdicts.items():
        c = ClaimCheck(cid)
        if not ok:
            c.failures.append("quote_mismatch")
            c.reasons.append("bad quote")
        checks[cid] = c
    return VerificationReport(checks)


C1 = Claim(id="c1", text="إجازة 14 يوماً", article_id="L1", quote="مدتها أربعة عشر يوماً")
C2 = Claim(id="c2", text="إشعار قبل شهر", article_id="L2", quote="قبل شهر واحد على الأقل")
ARTS = [LEAVE, NOTICE]


def test_decide_paths():
    assert decide(_answer(C1, C2), _report(c1=True, c2=True), U(), 1, 20, 12, ARTS).kind == "deliver"
    assert decide(_answer(C1, C2), _report(c1=True, c2=False), U(), 1, 20, 12, ARTS).kind == "retry"
    d = decide(_answer(C1, C2), _report(c1=True, c2=False), U(), 2, 20, 12, ARTS)
    assert d.kind == "deliver" and d.partial and [c.id for _, c in d.answer.claims()] == ["c1"]
    # not enough time for a rewrite -> drop the failing claim now
    assert decide(_answer(C1, C2), _report(c1=True, c2=False), U(), 1, 5, 12, ARTS).partial
    assert decide(_answer(C1), _report(c1=False), U(), 2, 20, 12, ARTS).kind == "no_support"
    assert decide(_answer(status="no_support"), _report(), U(), 1, 20, 12, ARTS).kind == "no_support"
    assert decide(_answer(C1, status="needs_lawyer"), _report(c1=True), U(), 1, 20, 12, ARTS).kind == "referral"


def test_amendment_warning():
    d = decide(_answer(C1), _report(c1=True), U(event_year=2015), 1, 20, 12, ARTS)
    assert d.amendment_warning
    d = decide(_answer(C1), _report(c1=True), U(event_year=2022), 1, 20, 12, ARTS)
    assert not d.amendment_warning
