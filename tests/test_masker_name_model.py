"""The optional name-model hook in PIIMasker, with a fake model (no download, no ML libraries)."""
import asyncio

import httpx
import pytest

from app.context import Deadline, RequestCtx
from app.errors import PIILeakError
from app.gateway.client import ModelGateway
from app.pii.masker import LABELS, PIIMasker, PIIModelError
from app.pii.name_model import NameModel


class FakeNames:
    """Flags the given words as names (every occurrence); optionally raises."""

    name = "fake"

    def __init__(self, words=(), raw_spans=None, fail=False):
        self.words, self.raw_spans, self.fail = list(words), raw_spans, fail
        self.calls = 0

    def detect_names(self, text):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model crashed")
        if self.raw_spans is not None:
            return self.raw_spans
        out = []
        for w in self.words:
            start = 0
            while (i := text.find(w, start)) != -1:
                out.append((i, i + len(w)))
                start = i + len(w)
        return out


def test_fake_satisfies_the_protocol():
    assert isinstance(FakeNames(), NameModel)


def test_model_adds_names_the_rules_miss():
    text = "تواصلت مع براء الفلاحات من الموارد البشرية"
    assert PIIMasker().mask(text).mapping == {}                    # rules alone miss it
    r = PIIMasker(name_model=FakeNames(["براء الفلاحات"])).mask(text)
    assert r.text == "تواصلت مع [شخص_1] من الموارد البشرية"
    assert r.mapping == {"[شخص_1]": "براء الفلاحات"}


def test_rule_and_model_name_spans_merge_into_one_name():
    text = "يا جماعة عمار السرحان صاحب المحل باعني تلفون خربان"
    assert PIIMasker().mask(text).text.startswith("يا جماعة [شخص_1] السرحان")    # rules: first name only
    r = PIIMasker(name_model=FakeNames(["السرحان"])).mask(text)                 # model: family name only
    assert r.text.startswith("يا جماعة [شخص_1] صاحب")
    assert r.mapping == {"[شخص_1]": "عمار السرحان"}


def test_structured_spans_always_win_over_model_names():
    text = "رقمي 0791234567 وايميلي ali.k@example.com"
    model = FakeNames(raw_spans=[(text.index("079"), text.index("079") + 6),      # overlaps the phone
                                 (text.index("ali"), text.index("ali") + 5)])     # overlaps the email
    r = PIIMasker(name_model=model).mask(text)
    assert r.text == "رقمي [هاتف_1] وايميلي [بريد_1]"
    assert set(r.mapping.values()) == {"0791234567", "ali.k@example.com"}


def test_invalid_model_spans_are_ignored():
    text = "سؤال عن الإجازة"
    space = text.index(" ")
    model = FakeNames(raw_spans=[(-5, 0), (3, 3), (100, 120), (6, 2), (space, space + 1)])  # out of range, empty,
    #                                                                     beyond the end, reversed, only a space
    assert PIIMasker(name_model=model).mask(text).text == text


def test_placeholders_numbering_and_same_value_same_placeholder():
    text = "براء قال لغيث إنه براء رح يدفع، ورقم غيث 0791234567"
    r = PIIMasker(name_model=FakeNames(["براء", "غيث"])).mask(text)
    assert r.text == "[شخص_1] قال ل[شخص_2] إنه [شخص_1] رح يدفع، ورقم [شخص_2] [هاتف_1]"
    assert all(p.startswith("[") and p.endswith("]") for p in r.mapping)
    assert {p.split("_")[0][1:] for p in r.mapping} <= set(LABELS.values())
    assert PIIMasker.restore(r.text, r.mapping) == text


def test_spans_method_matches_mask():
    m = PIIMasker(name_model=FakeNames(["براء"]))
    text = "اسمي خالد، وزميلي براء، ورقمي 0791234567"
    spans = m.spans(text)
    assert [text[s:e] for s, e, _ in spans] == ["خالد", "براء", "0791234567"]


def test_per_request_failure_fail_closed_raises():
    m = PIIMasker(name_model=FakeNames(fail=True), model_error_policy="fail_closed")
    with pytest.raises(PIIModelError):
        m.mask("اسمي خالد")


def test_per_request_failure_fallback_uses_rules_and_reports_it():
    m = PIIMasker(name_model=FakeNames(fail=True), model_error_policy="fallback")
    r = m.mask("اسمي خالد ورقمي 0791234567")
    assert r.text == PIIMasker().mask("اسمي خالد ورقمي 0791234567").text
    assert r.model_error == "RuntimeError"


def test_default_policy_is_fail_closed_and_invalid_policy_rejected():
    assert PIIMasker().model_error_policy == "fail_closed"
    with pytest.raises(ValueError):
        PIIMasker(model_error_policy="ignore")


def test_gateway_leak_guard_blocks_model_detected_names():
    text = "تواصلت مع براء الفلاحات من الموارد البشرية"
    r = PIIMasker(name_model=FakeNames(["براء الفلاحات"])).mask(text)
    cfg = {"providers": {"p": {"kind": "openai_chat", "base_url": "https://p.test", "api_key_env": "K"},
                         "q": {"kind": "openai_chat", "base_url": "https://q.test", "api_key_env": "K"}},
           "roles": {"verify": {"provider": "p", "model": "m", "timeout_s": 5}}}
    gw = ModelGateway(cfg, httpx.AsyncClient(transport=httpx.MockTransport(lambda req: httpx.Response(500))),
                      env={"K": "x"})
    ctx = RequestCtx(deadline=Deadline(30), pii_values=r.originals)
    with pytest.raises(PIILeakError):   # the raw name must never be sent
        asyncio.run(gw.chat_json("verify", [{"role": "user", "content": text}], object, ctx))
    gw._guard([r.text], ctx)            # the masked text passes the guard
