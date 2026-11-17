"""The real gateway against a mocked HTTP transport."""
import asyncio
import json

import httpx
import pytest

from app.context import Deadline, RequestCtx
from app.errors import ModelCallError, PIILeakError
from app.gateway.client import ModelGateway, extract_json
from app.schemas.llm import VerifierOutput

CONFIG = {
    "providers": {
        "a": {"kind": "openai_chat", "base_url": "https://a.test", "api_key_env": "A_KEY"},
        "b": {"kind": "openai_chat", "base_url": "https://b.test", "api_key_env": "B_KEY"},
        "e": {"kind": "openai_embed", "base_url": "https://e.test", "api_key_env": "E_KEY"},
        "r": {"kind": "cohere_rerank", "base_url": "https://r.test", "api_key_env": "R_KEY"},
    },
    "roles": {
        "understand": {"provider": "a", "model": "m-a", "timeout_s": 5},
        "generate": {"provider": "b", "model": "m-b", "timeout_s": 5},
        "verify": {"provider": "a", "model": "m-a", "timeout_s": 5},
        "embed": {"provider": "e", "model": "emb", "dimensions": 4, "timeout_s": 3},
        "rerank": {"provider": "r", "model": "rr", "timeout_s": 3},
    },
}


def chat_reply(content):
    return {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


def make_gw(handler):
    http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return ModelGateway(CONFIG, http, env={"A_KEY": "x", "B_KEY": "y", "E_KEY": "z", "R_KEY": "w"})


def ctx(pii=None):
    return RequestCtx(deadline=Deadline(30), pii_values=pii or [])


def test_extract_json_tolerates_fences():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": 2} thanks') == {"a": 2}


def test_repair_after_invalid_json():
    replies = iter([chat_reply("not json"), chat_reply(json.dumps({"verdicts": []}))])
    gw = make_gw(lambda req: httpx.Response(200, json=next(replies)))
    out = asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "x"}], VerifierOutput, ctx()))
    assert out.verdicts == []


def test_retry_once_on_503():
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=chat_reply('{"verdicts": []}'))

    gw = make_gw(handler)
    asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "x"}], VerifierOutput, ctx()))
    assert calls["n"] == 2


def test_400_is_not_retried():
    gw = make_gw(lambda req: httpx.Response(400, text="bad"))
    with pytest.raises(ModelCallError):
        asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "x"}], VerifierOutput, ctx()))


def test_pii_guard_blocks_call_but_ignores_article_text():
    gw = make_gw(lambda req: httpx.Response(200, json=chat_reply('{"verdicts": []}')))
    with pytest.raises(PIILeakError):
        asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "اسمي خالد"}], VerifierOutput,
                                 ctx(["خالد"])))
    msg = [{"role": "user", "content": '<article id="1">قال خالد بن الوليد</article> سؤال [شخص_1]'}]
    asyncio.run(gw.chat_json("verify", msg, VerifierOutput, ctx(["خالد"])))


def test_independence_check():
    gw = make_gw(lambda req: httpx.Response(200))
    gw.validate()
    bad = json.loads(json.dumps(CONFIG))
    bad["roles"]["generate"]["provider"] = "a"
    gw2 = ModelGateway(bad, httpx.AsyncClient(), env={})
    with pytest.raises(RuntimeError):
        gw2.validate()


def test_embed_and_rerank_shapes():
    def handler(req):
        if req.url.path.endswith("/embeddings"):
            body = json.loads(req.content)
            assert body["dimensions"] == 4
            return httpx.Response(200, json={"data": [{"index": 1, "embedding": [1, 1, 1, 1]},
                                                      {"index": 0, "embedding": [0, 0, 0, 0]}]})
        return httpx.Response(200, json={"results": [{"index": 1, "relevance_score": 0.9},
                                                     {"index": 0, "relevance_score": 0.2}]})

    gw = make_gw(handler)
    vecs = asyncio.run(gw.embed(["a", "b"], ctx()))
    assert vecs[0] == [0, 0, 0, 0]
    hits = asyncio.run(gw.rerank("q", ["d0", "d1"], 2, ctx()))
    assert hits[0].index == 1


def test_extra_body_is_sent_and_temperature_null_is_omitted():
    import copy
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["generate"].update({"temperature": None, "extra_body": {"reasoning_effort": "low"}})
    seen = {}

    def handler(req):
        seen.update(json.loads(req.content))
        return httpx.Response(200, json=chat_reply(json.dumps({"verdicts": []})))

    gw = ModelGateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                      env={"A_KEY": "x", "B_KEY": "y", "E_KEY": "z", "R_KEY": "w"})
    asyncio.run(gw.chat_json("generate", [{"role": "user", "content": "x"}], VerifierOutput, ctx()))
    assert seen["reasoning_effort"] == "low" and "temperature" not in seen and seen["model"] == "m-b"


@pytest.mark.parametrize("bad", [{"model": "other"}, {"messages": []}, {"response_format": None}, "low"])
def test_extra_body_cannot_override_core_fields(bad):
    import copy
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["generate"]["extra_body"] = bad
    with pytest.raises(RuntimeError):
        ModelGateway(cfg, httpx.AsyncClient(), env={})


def test_a_hanging_call_is_cut_at_its_timeout():
    import copy
    import time as _t
    cfg = copy.deepcopy(CONFIG)
    cfg["roles"]["verify"]["timeout_s"] = 0.5

    async def handler(req):                      # server that never answers in time
        await asyncio.sleep(5)
        return httpx.Response(200, json=chat_reply("{}"))

    gw = ModelGateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(handler)), min_call_s=0.1,
                      reserve_s=0.0, env={"A_KEY": "x", "B_KEY": "y", "E_KEY": "z", "R_KEY": "w"})
    start = _t.monotonic()
    with pytest.raises(ModelCallError, match="timeout"):
        asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "x"}], VerifierOutput, ctx()))
    assert _t.monotonic() - start < 1.5


def test_no_call_starts_without_enough_budget():
    gw = make_gw(lambda req: httpx.Response(200, json=chat_reply(json.dumps({"verdicts": []}))))
    tight = RequestCtx(deadline=Deadline(1.0))   # 1.0 - reserve 1.5 < min_call 2.0
    with pytest.raises(ModelCallError):
        asyncio.run(gw.chat_json("verify", [{"role": "user", "content": "x"}], VerifierOutput, tight))
