"""The unchanged orchestrator running on LocalStore (scripted models, invented fixture laws)."""
import asyncio
from pathlib import Path

from app.config import Settings
from app.context import Deadline, RequestCtx
from app.db.local_store import LocalStore
from app.orchestrator import Pipeline
from app.pii.masker import PIIMasker
from app.schemas.api import AskRequest
from fakes import FakeGateway, make_resources

FIX = Path(__file__).parent / "fixtures"
S = Settings()

U_LABOUR = {"msa_rewrite": "كم مدة الإجازة السنوية للعامل؟", "legal_rewrite": "مدة الإجازة السنوية للعامل",
            "domain": "labour", "domain_confidence": 0.9}
U_OTHER = {**U_LABOUR, "domain": "other", "domain_confidence": 0.8,
           "msa_rewrite": "هل يجوز للمالك قطع المياه عن المستأجر؟", "legal_rewrite": "قطع المالك المياه عن المأجور"}
SUPPORTED = {"verdicts": [{"id": "c1", "verdict": "supported"}]}


def answer(article_id, quote, text):
    return {"status": "answered", "summary": "ملخص",
            "sections": {"rights": [{"id": "c1", "text": text, "article_id": article_id, "quote": quote}]}}


def run(scripts, question):
    store = LocalStore(FIX / "local_laws", domains_file=FIX / "local_domains_test.yaml")
    p = Pipeline(FakeGateway(scripts), store, make_resources(S), PIIMasker(), S)

    async def go():
        await p.res.load(store)
        ctx = RequestCtx(deadline=Deadline(S.budget_s))
        return await p.handle(AskRequest(question=question), ctx), ctx, p

    return asyncio.run(go())


def test_focus_domain_uses_local_pack_with_definitions():
    resp, ctx, p = run({"understand": [U_LABOUR],
                        "generate": [answer("lob-2-9001:3", "إجازة سنوية بأجر كامل مدتها أربعة عشر يوماً",
                                            "للعامل إجازة سنوية مدتها 14 يوماً")],
                        "verify": [SUPPORTED]}, "كم يوم اجازة سنوية بستاهل؟")
    assert resp.type == "answer" and resp.coverage == "focus"
    assert ctx.trace.fields["path"] == "pack"
    assert resp.answer.sections.rights[0].citation.label == "قانون العمل التجريبي رقم 1 لسنة 2001، المادة 3"
    supplied = ctx.trace.fields["supplied_ids"]
    assert "lob-2-9001:2" in supplied and "lob-2-9001:5" not in supplied  # definitions in, repealed article out


def test_other_domain_uses_local_keyword_search():
    resp, ctx, _ = run({"understand": [U_OTHER],
                        "generate": [answer("lob-2-9002:3", "يحظر على المالك قطع المياه أو الكهرباء عن المأجور",
                                            "لا يجوز للمالك قطع المياه")],
                        "verify": [SUPPORTED]}, "المالك قطع عني المي")
    assert resp.type == "answer" and resp.coverage == "initial"
    assert ctx.trace.fields["path"] == "search"
    assert ctx.trace.fields["retrieved_ids"][0] == "lob-2-9002:3"
    assert "lob-2-9003:1" not in ctx.trace.fields["retrieved_ids"]  # repealed law never retrieved
