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
# CHANGED (P0 summary verification): the checker now also returns a verdict for the summary; without a
# "supported" summary verdict the summary is replaced by the neutral sentence.
SUPPORTED = {"verdicts": [{"id": "c1", "verdict": "supported"}, {"id": "summary", "verdict": "supported"}]}
CLAIM_ONLY = {"verdicts": [{"id": "c1", "verdict": "supported"}]}


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


# ---- MVP step 1: time budget, rewrite timing, safe abstention -----------------------------------------
import asyncio as _asyncio  # noqa: E402

from app import messages as M  # noqa: E402
from fakes import Delay  # noqa: E402

TWO_CLAIMS_BAD = {"status": "answered", "summary": "ملخص", "sections": {"rights": [
    {"id": "c1", "text": "يحق لك إجازة سنوية مدتها 14 يوماً", "article_id": "L1",
     "quote": "إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً"},
    {"id": "c2", "text": "الإجازة 30 يوماً", "article_id": "L1", "quote": "إجازة مدفوعة لمدة شهر"}]}}


def run_with(pipeline, req, budget):
    async def go():
        await pipeline.res.load(pipeline.store)
        ctx = RequestCtx(deadline=Deadline(budget))
        return await pipeline.handle(req, ctx), ctx
    return _asyncio.run(go())


def test_budget_default_is_40_seconds():
    assert Settings().budget_s == 40.0


def test_rewrite_skipped_when_this_requests_timing_does_not_fit():
    # generate takes 0.4 s, so a rewrite needs >= max(floor, 1.25 * 0.4 + reserve); with the floor at 0 and
    # only ~1 s budget left the rewrite must be skipped and the essential failing claim -> safe abstention
    s = Settings(retry_min_remaining_s=0.0, reserve_s=0.5, budget_s=1.2)
    gw = FakeGateway({"understand": [UNDERSTAND_LABOUR], "generate": [Delay(0.4, TWO_CLAIMS_BAD), GOOD],
                      "verify": [SUPPORTED, SUPPORTED]})
    p = Pipeline(gw, FakeStore(), make_resources(s), PIIMasker(), s)
    resp, ctx = run_with(p, AskRequest(question="كم يوم إجازة سنوية؟"), 1.2)
    assert ctx.trace.fields["generations"] == 1                    # no rewrite started
    assert resp.type == "no_support" and resp.message == M.UNVERIFIED
    assert ctx.trace.fields["retry_min_s"] >= 1.0


def test_rewrite_happens_when_time_remains():
    p, gw, _ = make({"understand": [UNDERSTAND_LABOUR], "generate": [TWO_CLAIMS_BAD, GOOD],
                     "verify": [SUPPORTED, SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert ctx.trace.fields["generations"] == 2 and resp.type == "answer"


def test_failed_rewrite_falls_back_to_the_first_answer_safely():
    one_optional_bad = {"status": "answered", "summary": "ملخص", "sections": {"rights": [
        {"id": "c1", "text": "يحق لك إجازة سنوية مدتها 14 يوماً", "article_id": "L1",
         "quote": "إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً"},
        {"id": "c2", "text": "معلومة إضافية", "article_id": "L1", "quote": "إجازة مدفوعة لمدة شهر",
         "essential": False}]}}
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [one_optional_bad, ModelCallError("timeout")],
                  "verify": [SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert ctx.trace.fields.get("rewrite_failed") is True
    assert resp.type == "answer" and resp.answer.partial                 # optional claim dropped, marked partial
    assert [c.text for c in resp.answer.sections.rights] == ["يحق لك إجازة سنوية مدتها 14 يوماً"]


def test_failed_rewrite_with_essential_failure_abstains():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [TWO_CLAIMS_BAD, ModelCallError("timeout")],
                  "verify": [SUPPORTED]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "no_support" and resp.message == M.UNVERIFIED


def test_whole_request_is_cut_at_the_budget():
    s = Settings(budget_s=0.5)
    gw = FakeGateway({"understand": [Delay(5.0, UNDERSTAND_LABOUR)]})   # a call that ignores its timeout
    p = Pipeline(gw, FakeStore(), make_resources(s), PIIMasker(), s)
    import time as _t
    start = _t.monotonic()
    resp, ctx = run_with(p, AskRequest(question="كم يوم إجازة سنوية؟"), 0.5)
    assert _t.monotonic() - start < 1.5
    assert resp.type == "error" and resp.message == M.TIMEOUT and ctx.trace.fields["outcome"] == "timeout"


def test_summary_number_not_in_question_or_articles_is_never_shown():
    bad_summary = {**GOOD, "summary": "لديك 999 يوماً من الإجازة"}
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [bad_summary, bad_summary],
                  "verify": [SUPPORTED, SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert ctx.trace.fields["generations"] == 2
    assert resp.type == "answer" and "999" not in resp.answer.summary and resp.answer.summary == M.NEUTRAL_SUMMARY


def test_summary_may_restate_numbers_from_the_question():
    s_ok = {**GOOD, "summary": "عملت 18 سنة، والقانون يحدد مدة الإجازة كما يلي."}
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [s_ok], "verify": [SUPPORTED]})
    resp, ctx = run(p, AskRequest(question="اشتغلت 18 سنة، كم يوم إجازة إلي؟"))
    assert ctx.trace.fields["generations"] == 1 and "18" in resp.answer.summary


# ---- P0 / C / D: summary verification, checker failure policy, inline citations ----------------------------
def test_summary_without_a_supported_verdict_is_replaced():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [CLAIM_ONLY]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "answer" and resp.answer.summary == M.NEUTRAL_SUMMARY
    assert resp.answer.verification == "verified"            # the claims themselves were checked


def test_summary_judged_unsupported_is_replaced_not_the_answer():
    bad = {"verdicts": [{"id": "c1", "verdict": "supported"},
                        {"id": "summary", "verdict": "unsupported", "reason": "استنتاج غير وارد"}]}
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [bad]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "answer" and resp.answer.summary == M.NEUTRAL_SUMMARY
    assert ctx.trace.fields["generations"] == 1                # a bad summary alone does not cost a rewrite


def test_every_shown_claim_has_an_inline_citation_from_the_store():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [SUPPORTED]})
    resp, _ = run(p, AskRequest(question="اسمي خالد، كم يوم إجازة سنوية إلي؟"))
    claim = resp.answer.sections.rights[0]
    assert claim.display_text == f"{claim.text} (قانون تجريبي للعمل، المادة 61)"
    assert "خالد" in claim.display_text                         # personal data restored in the display text too
    for name in ("rights", "obligations", "steps", "conditions"):
        for c in getattr(resp.answer.sections, name):
            assert c.citation.label and c.citation.quote and c.display_text.endswith(f"({c.citation.label})")


def test_checker_down_delivers_only_code_checked_claims_marked_not_semantically_verified():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [ModelCallError("429")]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "answer" and resp.answer.verification == "not_semantically_verified"
    assert resp.coverage == "initial" and M.SEMANTIC_SKIPPED_NOTE in resp.answer.caveats
    assert resp.answer.summary == M.NEUTRAL_SUMMARY             # an unchecked summary is never shown


def test_checker_down_with_an_essential_code_failure_abstains():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [TWO_CLAIMS_BAD, TWO_CLAIMS_BAD],
                  "verify": [ModelCallError("429"), ModelCallError("429")]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "no_support" and resp.message == M.UNVERIFIED


def test_checker_request_sends_each_article_once_and_keeps_one_verdict_per_claim():
    import json as _json
    three_on_one_article = {"status": "answered", "summary": "ملخص", "sections": {"rights": [
        {"id": f"c{i}", "text": "يحق لك إجازة سنوية مدتها 14 يوماً", "article_id": "L1",
         "quote": "إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً"} for i in (1, 2, 3)]}}
    verdicts = {"verdicts": [{"id": "c1", "verdict": "supported"}, {"id": "c2", "verdict": "supported"},
                             {"id": "summary", "verdict": "supported"}]}            # c3 missing on purpose
    p, gw, _ = make({"understand": [UNDERSTAND_LABOUR], "generate": [three_on_one_article, three_on_one_article],
                     "verify": [verdicts, verdicts]})
    resp, _ = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    sent = _json.loads([m for role, m in gw.calls if role == "verify"][0][1]["content"])
    assert [a["id"] for a in sent["articles"]] == ["L1"]                      # article text sent once
    assert [c["article_id"] for c in sent["claims"]] == ["L1", "L1", "L1"]     # every claim still listed
    assert sent["summary"]["id"] == "summary" and sent["summary"]["claim_ids"] == ["c1", "c2", "c3"]
    # c3 got no verdict -> it fails on its own (essential) -> the system abstains rather than drop it
    assert resp.type == "no_support" and resp.message == M.UNVERIFIED


def test_a_claim_with_the_reserved_summary_id_is_renamed():
    from app.schemas.llm import GeneratedAnswer as _GA
    a = _GA.model_validate({"status": "answered", "summary": "", "sections": {"rights": [
        {"id": "summary", "text": "t", "article_id": "L1", "quote": "q"}]}})
    a.ensure_unique_ids()
    assert a.sections.rights[0].id != "summary"


def test_name_model_masking_runs_off_the_event_loop_and_guard_still_holds():
    class FakeNames:
        name = "fake"

        def detect_names(self, text):
            i = text.find("غيث القيسي")
            return [(i, i + len("غيث القيسي"))] if i >= 0 else []

    gw = FakeGateway({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [SUPPORTED]})
    p = Pipeline(gw, FakeStore(), make_resources(S), PIIMasker(name_model=FakeNames()), S)
    resp, ctx = run(p, AskRequest(question="حكيت مع غيث القيسي عن الإجازة، كم يوم إجازة سنوية؟"))
    assert resp.type == "answer"
    assert all("غيث" not in m["content"] for _, msgs in gw.calls for m in msgs)   # nothing reached a model


# ---- structured errors and the API contract ---------------------------------------------------------------
from app.errors import ModelOutputError  # noqa: E402
from app.schemas.api import AskResponse  # noqa: E402


def test_invalid_model_output_returns_a_structured_error_not_a_timeout():
    p, *_ = make({"understand": [UNDERSTAND_LABOUR], "generate": [ModelOutputError("invalid JSON after repair")]})
    resp, ctx = run(p, AskRequest(question="كم يوم إجازة سنوية؟"))
    assert resp.type == "error" and resp.message == M.ERROR and resp.answer is None
    assert ctx.trace.fields["outcome"] == "generation_invalid_output"
    AskResponse.model_validate(resp.model_dump(mode="json"))               # still a valid API response


def _all_response_types():
    u_clar = {**UNDERSTAND_LABOUR, "missing_facts": [{"fact": "نوع العقد", "blocking": True}],
              "clarifying_question": "هل عقدك محدد المدة؟"}
    scripts = {
        "answer": ({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD], "verify": [SUPPORTED]}, "كم يوم إجازة؟"),
        "not_semantically_verified": ({"understand": [UNDERSTAND_LABOUR], "generate": [GOOD],
                                       "verify": [ModelCallError("timeout")]}, "كم يوم إجازة؟"),
        "clarification": ({"understand": [u_clar]}, "بدي أترك الشغل"),
        "referral": ({"understand": [UNDERSTAND_LABOUR]}, "ابني موقوف بالمخفر بسبب خلاف مع صاحب العمل"),
        "no_support": ({"understand": [UNDERSTAND_LABOUR], "generate": [BAD_QUOTE, BAD_QUOTE]}, "كم يوم إجازة؟"),
        "out_of_scope": ({"understand": [{**UNDERSTAND_LABOUR, "is_legal_question": False}]}, "مرحبا كيفك"),
        "error": ({"understand": [UNDERSTAND_LABOUR], "generate": [ModelCallError("timeout")]}, "كم يوم إجازة؟"),
    }
    for name, (script, q) in scripts.items():
        p, *_ = make(script)
        yield name, run(p, AskRequest(question=q))[0]


def test_every_response_type_matches_the_documented_contract():
    seen = set()
    for name, resp in _all_response_types():
        data = resp.model_dump(mode="json")
        AskResponse.model_validate(data)                                    # JSON round-trip is valid
        seen.add(data["type"])
        assert data["request_id"] and "timings_ms" in data and "data_version" in data
        if data["type"] in ("no_support", "out_of_scope", "error"):
            assert data["message"] and data["answer"] is None
        if data["type"] == "clarification":
            assert data["clarifying_question"]
        if data["type"] == "referral":
            assert data["referral"]["reason"]
        if data["answer"]:
            a = data["answer"]
            assert a["verification"] in ("verified", "not_semantically_verified")
            assert (a["verification"] == "not_semantically_verified") == (name == "not_semantically_verified")
            for section in ("rights", "obligations", "steps", "conditions"):
                for c in a["sections"][section]:                           # never a claim without its citation
                    assert c["citation"]["label"] and c["citation"]["quote"] and c["citation"]["article_id"]
                    assert c["display_text"] == f"{c['text']} ({c['citation']['label']})"
    assert seen == {"answer", "clarification", "referral", "no_support", "out_of_scope", "error"}


def test_documented_sample_responses_are_valid():
    import json as _json
    from pathlib import Path as _P
    for f in sorted((_P(__file__).parent.parent / "docs" / "sample_responses").glob("*.json")):
        AskResponse.model_validate(_json.loads(f.read_text(encoding="utf-8")))
