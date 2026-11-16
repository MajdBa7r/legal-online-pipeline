"""End-to-end runs of the orchestrator with a scripted gateway and an in-memory store."""
import asyncio

from app.config import Settings
from app.context import Deadline, RequestCtx
from app.errors import ModelCallError
from app.orchestrator import Pipeline
from app.pii.masker import PIIMasker
from app.schemas.api import AskRequest, Clarification
from fakes import FakeGateway, FakeStore, make_resources

S = Settings()

UNDERSTAND_LABOUR = {
    "msa_rewrite": "كم مدة الإجازة السنوية للعامل؟", "legal_rewrite": "مدة الإجازة السنوية للعامل",
    "domain": "labour", "domain_confidence": 0.9, "legal_issues": ["الإجازة السنوية"], "event_year": None,
}
UNDERSTAND_OTHER = {**UNDERSTAND_LABOUR, "domain": "other", "domain_confidence": 0.8,
                    "msa_rewrite": "ما عقوبة القيادة دون رخصة؟", "legal_rewrite": "عقوبة قيادة مركبة دون رخصة"}
GOOD = {"status": "answered", "summary": "إليك ما ينص عليه القانون.",
        "sections": {"rights": [{"id": "c1", "text": "يحق لـ [شخص_1] إجازة سنوية مدتها 14 يوماً",
                                 "article_id": "L1", "quote": "إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً"}]}}
BAD_QUOTE = {"status": "answered", "summary": "ملخص",
             "sections": {"rights": [{"id": "c1", "text": "إجازة 14 يوماً", "article_id": "L1",
                                      "quote": "إجازة مدفوعة لمدة أسبوعين"}]}}
SUPPORTED = {"verdicts": [{"id": "c1", "verdict": "supported"}]}


def run(pipeline, req):
    async def go():
        await pipeline.res.load(pipeline.store)
        ctx = RequestCtx(deadline=Deadline(S.budget_s))
        resp = await pipeline.handle(req, ctx)
        return resp, ctx
    return asyncio.run(go())


def make(scripts, store=None):
    store = store or FakeStore()
    gw = FakeGateway(scripts)
    return Pipeline(gw, store, make_resources(S), PIIMasker(), S), gw, store


def test_focus_domain_answer_with_pii_restored():
    p, gw, _ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="اسمي خالد، كم يوم إجازة سنوية إلي؟"))
    assert resp.type == "answer" and resp.coverage == "focus"
    claim = resp.answer.sections.rights[0]
    assert "خالد" in claim.text                       # restored for the user
    assert claim.citation.label == "قانون تجريبي للعمل، المادة 61"
    assert ctx.trace.fields["path"] == "pack"
    # nothing personal reached any model call
    assert all("خالد" not in m["content"] for _, msgs in gw.calls for m in msgs)
    # the definitions article was added to the pack
    gen_prompt = next(msgs for role, msgs in gw.calls if role == "generate")[1]["content"]
    assert 'id="L0"' in gen_prompt


def test_rewrite_once_then_deliver():
    p, gw, _ = make({"understand": [UNDERSTAND_LABOUR], "generate": [BAD_QUOTE, GOOD], "verify": [SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "answer"
    assert ctx.trace.fields["generations"] == 2
    second = [msgs for role, msgs in gw.calls if role == "generate"][1]
    assert "failed verification" in second[-1]["content"]


def test_two_failures_means_no_support():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [BAD_QUOTE, BAD_QUOTE]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "no_support" and resp.message


def test_other_domain_search_path_and_initial_label():
    answer = {"status": "answered", "summary": "ملخص", "sections": {"conditions": [
        {"id": "c1", "text": "غرامة لا تقل عن 50 ديناراً", "article_id": "T1",
         "quote": "بغرامة لا تقل عن خمسين ديناراً"}]}}
    p, gw, store = make({"understand": [UNDERSTAND_OTHER], "generate": [answer], "verify": [SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="شو عقوبة اللي بسوق بدون رخصة؟"))
    assert resp.type == "answer" and resp.coverage == "initial"
    assert ctx.trace.fields["path"] == "search"
    assert len(store.search_calls) == 3                       # one per wording
    assert "R1" not in ctx.trace.fields["retrieved_ids"]      # repealed filtered out


def test_understand_failure_falls_back_to_search():
    answer = {"status": "no_support", "summary": "", "sections": {}}
    p, *_ = make({"understand": [ModelCallError("down")], "generate": [answer]})
    resp, ctx = run(p, AskRequest(question="سؤال قانوني ما"))
    assert ctx.trace.fields["understand_fallback"] is True
    assert ctx.trace.fields["path"] == "search"
    assert resp.type == "no_support"


def test_clarification_then_answer():
    u = {**UNDERSTAND_LABOUR, "missing_facts": [{"fact": "نوع العقد", "blocking": True}],
         "clarifying_question": "هل عقدك محدد المدة؟"}
    p, *_ = make({"understand": [u]})
    resp, _ = run(p, AskRequest(question="بدي أترك الشغل، شو لازم أعمل؟"))
    assert resp.type == "clarification" and resp.clarifying_question

    p, *_ = make({"understand": [u], "generate": [GOOD], "verify": [SUPPORTED]})
    resp, _ = run(p, AskRequest(question="بدي أترك الشغل", clarification=Clarification(
        original_question="بدي أترك الشغل، شو لازم أعمل؟", clarifying_question="هل عقدك محدد المدة؟",
        answer="غير محدد")))
    assert resp.type == "answer"


def test_referral_keyword():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR]})
    resp, _ = run(p, AskRequest(question="ابني موقوف بالمخفر بسبب خلاف مع صاحب العمل"))
    assert resp.type == "referral"


def test_semantic_check_down_delivers_labelled():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [ModelCallError("down")]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "answer" and resp.coverage == "initial"


def test_generation_failure_returns_timeout_message():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [ModelCallError("timeout")]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "error" and resp.message
