"""MVP step 2: run real questions through the whole pipeline (all 7 stages, real models) and report everything.

    python scripts/run_e2e.py

Builds the pipeline exactly like the server (app/main.py) with the locked models.yaml and settings, then sends
the 4 test questions. Q1 carries an INVENTED name and phone number to test masking end to end. If the pipeline
asks a clarifying question, the prepared reply below is sent back once (the website does the same).

Prints a summary per question and writes logs/e2e_<time>.html (the answer as the user sees it, plus the
understanding, every draft, every checker verdict, timings) and logs/e2e_<time>.json. Changes nothing.
Costs a few cents.
"""
import asyncio
import html
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import PROJECT_DIR, Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.factory import make_store  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.orchestrator import Pipeline  # noqa: E402
from app.pii.factory import build_masker  # noqa: E402
from app.resources import Resources  # noqa: E402
from app.schemas.api import AskRequest, Clarification  # noqa: E402

INVENTED_PII = ["براء الفلاحات", "براء", "الفلاحات", "0791234567"]   # must never reach a model
QUESTIONS = [  # synthetic; same scenarios as the writer comparison
    dict(id="Q1", domain="labour",
         q="اسمي براء الفلاحات ورقمي 0791234567، اشتغلت سنتين بالشركة وما أخذت ولا يوم إجازة، "
           "كم يوم إجازة سنوية إلي وشو بصير بالأيام اللي ما أخذتها؟",
         reply="لسا بشتغل بنفس الشركة وما اتفقت معهم على تأجيل الإجازة."),
    dict(id="Q2", domain="rentals",
         q="المالك بدو يطلعني من الشقة قبل ما يخلص العقد وأنا دافع الأجرة أول بأول، بيحقله؟",
         reply="العقد موقع سنة 2021 لمدة سنتين، والشقة للسكن."),
    dict(id="Q3", domain="social_security",
         q="أنا مشترك بالضمان من 18 سنة وعمري 47، بقدر أطلع على التقاعد المبكر هلأ؟",
         reply="أنا ذكر، وشغلي محاسب بمكتب."),
    dict(id="Q4", domain="consumer_protection",
         q="اشتريت غسالة جديدة وطلع فيها عيب بعد أسبوع والمحل رافض يرجعها أو يصلحها، شو حقي؟",
         reply="اشتريتها من محل أجهزة كهربائية ومعي الفاتورة."),
]
REVIEW = [
    "Legal accuracy: every claim and the SUMMARY match the cited articles (the summary is not machine-checked).",
    "Interpretation: the answer addresses this user's situation and facts, not a nearby topic.",
    "Completeness: main rule + conditions/exceptions that change the outcome + next step.",
    "Citations: each claim cites the article that really supports it.",
    "Essential flags: claims the answer depends on are marked essential (see the drafts section).",
    "Safety: abstentions/referrals are appropriate; nothing misleading is shown.",
]


class Recorder:
    """Wraps gateway.chat_json to keep every role call: what was sent, what came back, how long it took."""

    def __init__(self, gw: ModelGateway):
        self.gw, self.calls, self._orig = gw, [], gw.chat_json
        gw.chat_json = self.chat_json

    async def chat_json(self, role, messages, schema, ctx):
        t0 = time.monotonic()
        rec = {"role": role, "messages": messages, "output": None, "error": None}
        try:
            out = await self._orig(role, messages, schema, ctx)
            rec["output"] = out.model_dump()
            return out
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            raise
        finally:
            rec["seconds"] = round(time.monotonic() - t0, 2)
            self.calls.append(rec)


async def ask(pipeline, settings, req):
    ctx = RequestCtx(deadline=Deadline(settings.budget_s))
    t0 = time.monotonic()
    resp = await pipeline.handle(req, ctx)
    return resp, ctx, time.monotonic() - t0


def round_info(resp, ctx, secs, calls):
    return {"seconds": round(secs, 1), "response": resp.model_dump(mode="json"),
            "trace": ctx.trace.to_row(ctx.request_id, ctx.deadline.elapsed_ms()), "calls": calls}


_ARTICLE = re.compile(r"<article\b[^>]*>.*?</article>", re.DOTALL)


def pii_leaks(calls, values=INVENTED_PII) -> list[str]:
    """Invented values found as WHOLE words in what was sent, ignoring law text (<article> blocks), like the
    gateway guard. (A plain substring test flagged "براء" inside the law words "خبراء" and "براءة".)"""
    sent = " ".join(_ARTICLE.sub(" ", m.get("content", "")) for c in calls for m in c["messages"])
    return [v for v in values if re.search(rf"(?<![\w\u0600-\u06FF]){re.escape(v)}(?![\w\u0600-\u06FF])", sent)]


# ------------------------------------------------------------------ report
def e(s) -> str:
    return html.escape(str(s))


def render_round(label, rd) -> str:
    r, t = rd["response"], rd["trace"]
    out = [f"<h3>{e(label)}: type <b>{e(r['type'])}</b> · {rd['seconds']} s</h3>"]
    if r.get("message"):
        out.append(f"<p class=msg>{e(r['message'])}</p>")
    if r.get("clarifying_question"):
        out.append(f"<p class=msg>سؤال توضيحي: {e(r['clarifying_question'])}</p>")
    a = r.get("answer")
    if a:
        out.append(f"<div class=ans><p><b>الخلاصة:</b> {e(a['summary'])}</p>")
        names = {"rights": "الحقوق", "obligations": "الالتزامات", "steps": "الخطوات", "conditions": "الشروط"}
        for k, title in names.items():
            items = a["sections"].get(k) or []
            if items:
                out.append(f"<h4>{title}</h4><ol>" + "".join(
                    f"<li>{e(c.get('display_text') or c['text'])}"
                    f"<div class=quote>«{e(c['citation']['quote'])}»</div></li>" for c in items) + "</ol>")
        if a.get("caveats"):
            out.append("<h4>تنبيهات</h4><ul>" + "".join(f"<li>{e(x)}</li>" for x in a["caveats"]) + "</ul>")
        out.append(f"<p class=ltr>verification: <b>{a.get('verification')}</b> · partial: {a.get('partial')} · "
                   f"amendment warning: {a.get('amendment_warning')}</p></div>")
    fields = {k: v for k, v in t.items() if k not in ("model_calls", "supplied_ids", "retrieved_ids",
                                                      "cited_ids", "request_id")}
    out.append(f"<details class=ltr><summary>trace (timings, decision)</summary><pre>"
               f"{e(json.dumps(fields, ensure_ascii=False, indent=1))}</pre></details>")
    for c in rd["calls"]:
        body = c["error"] or json.dumps(c["output"], ensure_ascii=False, indent=1)
        out.append(f"<details><summary class=ltr>{e(c['role'])} call · {c['seconds']} s"
                   f"{' · ERROR' if c['error'] else ''}</summary><pre>{e(body)}</pre></details>")
    return "".join(out)


def render(results) -> str:
    css = """body{font-family:Tahoma,Arial,sans-serif;margin:24px;line-height:1.7;color:#222}
    .ans{border:1px solid #ccc;border-radius:8px;padding:12px;background:#fafafa} .msg{background:#fff4e5;padding:8px}
    .cite{font-size:13px;color:#555} .quote{font-size:13px;color:#335;background:#eef;padding:4px}
    .ltr{direction:ltr;text-align:left} pre{direction:ltr;text-align:left;white-space:pre-wrap;font-size:12px}
    .bad{color:#b00020}"""
    out = [f"<!DOCTYPE html><html lang=ar dir=rtl><meta charset=utf-8><title>E2E run</title><style>{css}</style>"
           "<body><div class=ltr><h1>First end-to-end run</h1><h2>Review checklist</h2><ol>"
           + "".join(f"<li>{e(x)}</li>" for x in REVIEW) + "</ol></div>"]
    for r in results:
        leak = "<span class=bad>LEAK: " + e(", ".join(r["pii_leaks"])) + "</span>" if r["pii_leaks"] \
            else "no invented personal data reached any model"
        out.append(f"<hr><h2>{e(r['id'])} · {e(r['domain'])}</h2><p>{e(r['question'])}</p>"
                   f"<p class=ltr>PII check: {leak}</p>")
        for i, rd in enumerate(r["rounds"]):
            label = "Round 1" if i == 0 else f"Round 2 (after clarification reply: {r['reply']})"
            out.append(render_round(label, rd))
    out.append("</body></html>")
    return "".join(out)


# ------------------------------------------------------------------ main
async def main():
    s = Settings.from_env()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=50)) as http:
        gw = ModelGateway.from_file(s.models_file, http, reserve_s=s.reserve_s, min_call_s=s.min_call_s)
        gw.validate()
        need = {"DEEPSEEK_API_KEY", "MOONSHOT_API_KEY"}
        if missing := need & set(gw.missing_keys()):
            sys.exit(f"Set these keys in this terminal first: {', '.join(sorted(missing))}")
        store = make_store(s, http)
        res = Resources.from_settings(s)
        await res.load(store)
        masker, pii_status = build_masker(s)
        pipeline = Pipeline(gw, store, res, masker, s)
        rec = Recorder(gw)
        print(f"budget {s.budget_s:g} s · models: "
              + ", ".join(f"{r}={gw.roles[r].model}" for r in ("understand", "generate", "verify"))
              + f" · PII: {pii_status.label()}\n")

        results = []
        for item in QUESTIONS:
            rounds = []
            start = len(rec.calls)
            resp, ctx, secs = await ask(pipeline, s, AskRequest(question=item["q"]))
            rounds.append(round_info(resp, ctx, secs, rec.calls[start:]))
            if resp.type == "clarification" and resp.clarifying_question:
                start = len(rec.calls)
                req = AskRequest(question=item["q"], clarification=Clarification(
                    original_question=item["q"], clarifying_question=resp.clarifying_question, answer=item["reply"]))
                resp, ctx, secs = await ask(pipeline, s, req)
                rounds.append(round_info(resp, ctx, secs, rec.calls[start:]))
            all_calls = [c for rd in rounds for c in rd["calls"]]
            leaks = pii_leaks(all_calls)
            results.append({**item, "question": item["q"], "rounds": rounds, "pii_leaks": leaks})

            for i, rd in enumerate(rounds):
                t, r = rd["trace"], rd["response"]
                tm = t["timings_ms"]
                stages = " ".join(f"{k}={v / 1000:.1f}" for k, v in tm.items() if k != "total")
                ver = t.get("verification") or {}
                a = r.get("answer") or {}
                n = sum(len(v) for v in (a.get("sections") or {}).values()) if a else 0
                print(f"{item['id']}{'' if i == 0 else ' (after clarification)'} [{item['domain']}] "
                      f"type={r['type']}  total={rd['seconds']} s  gate={t.get('gate')}  path={t.get('path')}")
                print(f"     stages(s): {stages}")
                print(f"     generations={t.get('generations')}  verification={ver}  shown claims={n}  "
                      f"partial={a.get('partial') if a else '-'}  verification={a.get('verification') if a else '-'}  "
                      f"summary={t.get('summary_verdict')}  outcome={t.get('outcome')}"
                      + (f"  retry_min={t.get('retry_min_s')}" if t.get("retry_min_s") else "")
                      + ("  REWRITE FAILED" if t.get("rewrite_failed") else ""))
                for c in t["model_calls"]:
                    print(f"       {c['role']:10s} {c['model']:16s} {c['latency_ms'] / 1000:5.1f} s  {c['status']}"
                          f"  tokens in/out={c['input_tokens']}/{c['output_tokens']}")
            print(f"     PII: {'LEAK ' + ', '.join(leaks) if leaks else 'ok (nothing invented reached a model)'}\n")

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out = PROJECT_DIR / "logs"
    out.mkdir(exist_ok=True)
    (out / f"e2e_{stamp}.html").write_text(render(results), encoding="utf-8")
    (out / f"e2e_{stamp}.json").write_text(json.dumps(results, ensure_ascii=False, indent=1, default=str),
                                           encoding="utf-8")
    print(f"Report: {out / f'e2e_{stamp}.html'}")


if __name__ == "__main__":
    asyncio.run(main())
