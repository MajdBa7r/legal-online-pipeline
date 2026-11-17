"""Compare two answer writers (generate role) on the same questions and the same legal context.

    python scripts/compare_writers.py                 # 25 s limit per answer
    python scripts/compare_writers.py --timeout 90    # quality-only run: give slow writers time to finish

What is held equal for both writers:
* the 4 questions below (synthetic, no personal data) and their understanding (written here, not by a model);
* the retrieved articles: retrieval runs ONCE per question with the production Retriever, and both writers
  receive exactly the same articles in the same order. For the 4 focus domains production sends the WHOLE
  law (domain pack, 14k-142k characters), so times here are realistic, unlike scripts/try_generate_settings.py
  which used 6 articles;
* the prompt (app/prompts/generate.md via the production build_messages), max_tokens from models.yaml,
  JSON mode, thinking disabled, and a 25 s timeout (comparison only; production stays at 14 s);
* temperature: Kimi k2.6 fixes it at 0.6 when thinking is off, so DeepSeek is also set to 0.6.

Objective checks per answer (the production code checks, no model judge):
* every claim's quote is an exact span of the cited article (or only matches after normalization / too short);
* the cited article was among the supplied articles; numbers in the claim appear in the article;
* article numbers written in the text ("المادة 61") exist among the supplied articles.
These do NOT establish legal correctness: a correct quote can support a wrong conclusion. Read the answers.

Output: a summary in the terminal and a side-by-side HTML report in logs/ (git-ignored). Answers are shown
as "Answer A" / "Answer B" in random order per question; the key is hidden at the bottom of the report.
Changes nothing in the project. Costs a few cents.
"""
import argparse
import asyncio
import copy
import html
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import yaml  # noqa: E402

from app.config import PROJECT_DIR, Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.factory import make_store  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.resources import Resources  # noqa: E402
from app.schemas.llm import SECTION_NAMES, GeneratedAnswer, Understanding  # noqa: E402
from app.stages.generate import build_messages  # noqa: E402
from app.stages.retrieve import Retriever  # noqa: E402
from app.stages.verify import code_checks  # noqa: E402

TIMEOUT_S = 25          # comparison only; models.yaml keeps its own value
TEMPERATURE = 0.6       # what Kimi k2.6 uses with thinking off (fixed by Moonshot)
OFF = {"thinking": {"type": "disabled"}}
WRITERS = {  # name -> (provider, model, temperature or None to omit, extra_body)
    "deepseek-flash": ("deepseek", "deepseek-flash", TEMPERATURE, OFF),
    "kimi-k2.6": ("moonshot", "kimi-k2.6", None, OFF),
}

QUESTIONS = [  # synthetic, written for this comparison; no personal data
    dict(domain="labour",
         q="اشتغلت سنتين بالشركة وما أخذت ولا يوم إجازة، كم يوم إجازة سنوية إلي وشو بصير بالأيام اللي ما أخذتها؟",
         msa="عملت سنتين في الشركة ولم آخذ أي إجازة، فكم يوما من الإجازة السنوية أستحق وما مصير الأيام غير المستخدمة؟",
         legal="مدة الإجازة السنوية للعامل وحكم الإجازات السنوية غير المستعملة",
         issues=["الإجازة السنوية", "الإجازات غير المستعملة"]),
    dict(domain="rentals",
         q="المالك بدو يطلعني من الشقة قبل ما يخلص العقد وأنا دافع الأجرة أول بأول، بيحقله؟",
         msa="يريد المالك إخراجي من الشقة قبل انتهاء عقد الإيجار مع أنني أدفع الأجرة بانتظام، فهل يحق له ذلك؟",
         legal="حالات جواز إخلاء المأجور قبل انتهاء مدة عقد الإيجار",
         issues=["إخلاء المأجور", "انتهاء عقد الإيجار"]),
    dict(domain="social_security",
         q="أنا مشترك بالضمان من 18 سنة وعمري 47، بقدر أطلع على التقاعد المبكر هلأ؟",
         msa="أنا مشترك في الضمان الاجتماعي منذ 18 سنة وعمري 47 عاما، فهل يحق لي التقاعد المبكر الآن؟",
         legal="شروط استحقاق راتب التقاعد المبكر من حيث السن ومدة الاشتراك",
         issues=["التقاعد المبكر", "مدة الاشتراك", "السن"]),
    dict(domain="consumer_protection",
         q="اشتريت غسالة جديدة وطلع فيها عيب بعد أسبوع والمحل رافض يرجعها أو يصلحها، شو حقي؟",
         msa="اشتريت غسالة جديدة وظهر فيها عيب بعد أسبوع ويرفض المتجر إرجاعها أو إصلاحها، فما حقوقي؟",
         legal="حق المستهلك في إرجاع السلعة المعيبة أو استبدالها أو إصلاحها",
         issues=["السلعة المعيبة", "الإرجاع والاستبدال", "التزامات المزود"]),
]

CHECKLIST = [
    ("Legal accuracy", "Every statement matches what the cited articles actually say; no invented rules, "
                       "no wrong numbers, durations or conditions; nothing contradicts the law."),
    ("Interpretation", "The answer understood the dialect question and the user's real situation "
                       "(e.g. 2 years of unused leave, landlord before contract end), not a nearby topic."),
    ("Completeness", "Covers the main right, the conditions/exceptions that change the outcome, and the "
                     "practical next step; says clearly when the articles do not settle the question."),
    ("Citation relevance", "Each claim cites the article that really supports it (not just a related one), "
                           "and the quote is the part that matters."),
    ("Clarity in Arabic", "Clear Modern Standard Arabic a non-lawyer can follow; no legal jargon left "
                          "unexplained; no awkward or machine-like phrasing."),
    ("Caution", "Flags uncertainty, missing facts and when to see a lawyer; does not overstate certainty."),
]

_ART_REF = re.compile(r"(?:الماد[ةه]|مادة|المواد)\s*\(?\s*([0-9٠-٩]+)")
_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")


def lead_number(s: str) -> str:
    m = re.match(r"\s*([0-9٠-٩]+)", str(s).translate(_DIGITS))
    return m.group(1) if m else str(s).strip()


def objective_checks(ans: GeneratedAnswer, articles, settings) -> dict:
    supplied = {a.id: a for a in articles}
    supplied_nums = {lead_number(a.article_no) for a in articles}
    claims = []
    for section, c in ans.claims():
        chk = code_checks(c, supplied, settings)
        art = supplied.get(c.article_id)
        refs = [r.translate(_DIGITS) for r in _ART_REF.findall(c.text)]
        bad_refs = [r for r in refs if r not in supplied_nums]
        claims.append({
            "section": section, "id": c.id, "text": c.text, "quote": c.quote, "article_id": c.article_id,
            "article": f"{art.law_title} - المادة {art.article_no}" if art else "(not supplied)",
            "failures": chk.failures, "normalized_only": chk.quote_normalized_only,
            "unknown_article_refs": bad_refs,
        })
    free_text = " ".join([ans.summary, *ans.caveats])
    free_refs = [r.translate(_DIGITS) for r in _ART_REF.findall(free_text)]
    n = len(claims)
    return {
        "claims": n,
        "exact_quotes": sum("quote_mismatch" not in c["failures"] and "quote_too_short" not in c["failures"]
                            and "not_supplied" not in c["failures"] for c in claims),
        "quote_normalized_only": sum(c["normalized_only"] for c in claims),
        "quote_mismatch": sum("quote_mismatch" in c["failures"] for c in claims),
        "quote_too_short": sum("quote_too_short" in c["failures"] for c in claims),
        "not_supplied": sum("not_supplied" in c["failures"] for c in claims),
        "number_mismatch": sum("number_mismatch" in c["failures"] for c in claims),
        "not_in_force": sum("not_in_force" in c["failures"] for c in claims),
        "unknown_article_refs": sum(len(c["unknown_article_refs"]) for c in claims)
        + sum(r not in supplied_nums for r in free_refs),
        "claims_failing_code_checks": sum(bool(c["failures"]) or bool(c["unknown_article_refs"]) for c in claims),
        "claim_details": claims,
    }


async def write(gw: ModelGateway, messages) -> tuple[GeneratedAnswer | None, float, str | None]:
    t0 = time.monotonic()
    try:
        ans = await gw.chat_json("generate", messages, GeneratedAnswer,
                                 RequestCtx(deadline=Deadline(TIMEOUT_S + 30)))
        ans.ensure_unique_ids()
        return ans, time.monotonic() - t0, None
    except Exception as e:
        return None, time.monotonic() - t0, f"{type(e).__name__}: {str(e)[:300]}"


def writer_gateway(base: dict, http, provider, model, temperature, extra) -> ModelGateway:
    cfg = copy.deepcopy(base)
    r = cfg["roles"]["generate"]
    r.update({"provider": provider, "model": model, "temperature": temperature, "timeout_s": TIMEOUT_S,
              "fallback": None})
    r.pop("extra_body", None)
    if extra:
        r["extra_body"] = extra
    return ModelGateway(cfg, http, min_call_s=1.0)


# ------------------------------------------------------------------ report
def e(s) -> str:
    return html.escape(str(s))


def render_answer(label, run) -> str:
    if run["answer"] is None:
        return f"<div class=ans><h3>{label}</h3><p class=bad>FAILED after {run['seconds']:.1f} s: " \
               f"{e(run['error'])}</p></div>"
    a, oc = run["answer"], run["checks"]
    parts = [f"<div class=ans><h3>{label}</h3>",
             f"<p class=meta>status: {e(a['status'])} · {run['seconds']:.1f} s · {oc['claims']} claims · "
             f"exact quotes {oc['exact_quotes']}/{oc['claims']} · failing code checks "
             f"{oc['claims_failing_code_checks']} · unknown article numbers {oc['unknown_article_refs']}</p>",
             f"<p><b>الخلاصة:</b> {e(a['summary'])}</p>"]
    by_section = {}
    for c in oc["claim_details"]:
        by_section.setdefault(c["section"], []).append(c)
    names = {"rights": "الحقوق", "obligations": "الالتزامات", "steps": "الخطوات", "conditions": "الشروط"}
    for s in SECTION_NAMES:
        if s not in by_section:
            continue
        parts.append(f"<h4>{names[s]}</h4><ol>")
        for c in by_section[s]:
            flags = ", ".join(c["failures"] + [f"unknown article {r}" for r in c["unknown_article_refs"]])
            badge = f"<span class=bad>✗ {e(flags)}</span>" if flags else "<span class=good>✓ code checks</span>"
            parts.append(f"<li>{e(c['text'])}<div class=cite>{e(c['article'])} · {badge}</div>"
                         f"<div class=quote>«{e(c['quote'])}»</div></li>")
        parts.append("</ol>")
    if a["caveats"]:
        parts.append("<h4>تنبيهات</h4><ul>" + "".join(f"<li>{e(x)}</li>" for x in a["caveats"]) + "</ul>")
    parts.append("</div>")
    return "".join(parts)


def render_report(results, seed) -> str:
    css = """body{font-family:Tahoma,Arial,sans-serif;margin:24px;line-height:1.7;color:#222}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:16px} .ans{border:1px solid #ccc;border-radius:8px;
    padding:12px;background:#fafafa} .meta{font-size:13px;color:#555;direction:ltr;text-align:left}
    .cite{font-size:13px;color:#555} .quote{font-size:13px;color:#335;background:#eef;padding:4px;margin:2px 0}
    .bad{color:#b00020} .good{color:#1b7f3a} .ltr{direction:ltr;text-align:left}
    table{border-collapse:collapse} td,th{border:1px solid #bbb;padding:4px 8px} details{margin:8px 0}"""
    out = [f"<!DOCTYPE html><html lang=ar dir=rtl><meta charset=utf-8><title>Writer comparison</title>"
           f"<style>{css}</style><body>",
           "<div class=ltr><h1>Writer comparison (blind)</h1><p>Same questions, same retrieved articles, same "
           f"prompt; thinking off; {TIMEOUT_S} s timeout; temperature {TEMPERATURE}. Code checks show quote and "
           "number support only; they do not establish legal correctness.</p>",
           "<h2>Manual checklist: score each answer 1-5</h2><ol>"
           + "".join(f"<li><b>{e(n)}</b>: {e(d)}</li>" for n, d in CHECKLIST)
           + "</ol><p>Fill in the scores before opening the key at the bottom.</p></div>"]
    for i, r in enumerate(results, 1):
        out.append(f"<hr><h2 class=ltr>Q{i} · {e(r['domain'])}</h2><p><b>السؤال:</b> {e(r['question'])}</p>")
        arts = "".join(f"<details><summary>{e(a['header'])}</summary><p>{e(a['text'])}</p></details>"
                       for a in r["articles"])
        out.append(f"<details><summary class=ltr>Retrieved articles ({len(r['articles'])}, path: "
                   f"{e(r['path'])}) - identical for both answers</summary>{arts}</details>")
        out.append("<div class=grid>" + "".join(render_answer(f"Answer {lbl}", r["runs"][name])
                                                 for lbl, name in r["order"]) + "</div>")
        rows = "".join(f"<tr><td>{e(n)}</td><td></td><td></td></tr>" for n, _ in CHECKLIST)
        out.append(f"<table class=ltr><tr><th>Criterion</th><th>A (1-5)</th><th>B (1-5)</th></tr>{rows}</table>")
    key = "".join(f"<li>Q{i}: " + ", ".join(f"{lbl} = {e(name)}" for lbl, name in r["order"]) + "</li>"
                  for i, r in enumerate(results, 1))
    out.append(f"<hr><details class=ltr><summary>Key: which model wrote A and B (open after scoring)"
               f"</summary><ol>{key}</ol><p>seed {seed}</p></details></body></html>")
    return "".join(out)


# ------------------------------------------------------------------ main
async def main():
    global TIMEOUT_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=TIMEOUT_S, help="seconds per answer (comparison only)")
    TIMEOUT_S = ap.parse_args().timeout
    print(f"timeout per answer: {TIMEOUT_S:g} s (comparison only; models.yaml unchanged)")
    s = Settings.from_env()
    with open(s.models_file, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    seed = random.SystemRandom().randrange(10**6)
    rng = random.Random(seed)
    results = []
    async with httpx.AsyncClient() as http:
        store = make_store(s, http)
        res = Resources.from_settings(s)
        await res.load(store)
        retriever = Retriever(ModelGateway(base, http), store, res, s)
        writers = {name: writer_gateway(base, http, *spec) for name, spec in WRITERS.items()}

        for i, item in enumerate(QUESTIONS, 1):
            u = Understanding(msa_rewrite=item["msa"], legal_rewrite=item["legal"], domain=item["domain"],
                              domain_confidence=0.9, legal_issues=item["issues"])
            ctx = RequestCtx(deadline=Deadline(60))
            articles, path = await retriever.get_articles(item["q"], u, item["domain"], True, ctx)
            messages = build_messages(item["q"], u, articles, res)       # identical for both writers
            chars = sum(len(a.text) for a in articles)
            listed = ", ".join(a.article_no for a in articles) if len(articles) <= 12 else "whole law (domain pack)"
            print(f"\nQ{i} [{item['domain']}] {len(articles)} articles via {path} ({chars:,} chars): {listed}")
            runs = {}
            for name, gw in writers.items():
                ans, secs, err = await write(gw, messages)
                run = {"seconds": secs, "error": err, "answer": ans.model_dump() if ans else None,
                       "checks": objective_checks(ans, articles, s) if ans else None}
                runs[name] = run
                if ans:
                    oc = run["checks"]
                    print(f"  {name:16s} {secs:5.1f} s  status={ans.status:12s} claims={oc['claims']:2d}  "
                          f"exact quotes={oc['exact_quotes']}/{oc['claims']}  "
                          f"failing code checks={oc['claims_failing_code_checks']}  "
                          f"unknown article numbers={oc['unknown_article_refs']}  "
                          f"number mismatches={oc['number_mismatch']}")
                else:
                    print(f"  {name:16s} {secs:5.1f} s  FAILED {err}")
            order = list(WRITERS)
            rng.shuffle(order)
            results.append({"domain": item["domain"], "question": item["q"], "path": path,
                            "articles": [{"id": a.id, "header": a.header(), "text": a.text} for a in articles],
                            "runs": runs, "order": list(zip("AB", order))})

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    out_dir = PROJECT_DIR / "logs"
    out_dir.mkdir(exist_ok=True)
    html_path = out_dir / f"compare_writers_{stamp}.html"
    html_path.write_text(render_report(results, seed), encoding="utf-8")
    (out_dir / f"compare_writers_{stamp}.json").write_text(
        json.dumps({"seed": seed, "settings": {"timeout_s": TIMEOUT_S, "temperature": TEMPERATURE,
                                               "writers": WRITERS}, "results": results},
                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    print("\nTotals (both writers saw identical questions, articles and prompt):")
    for name in WRITERS:
        rs = [r["runs"][name] for r in results]
        ok = [r for r in rs if r["answer"]]
        n = sum(r["checks"]["claims"] for r in ok)
        print(f"  {name:16s} answered {len(ok)}/{len(rs)}  "
              f"median time {sorted(r['seconds'] for r in rs)[len(rs) // 2]:.1f} s  "
              f"claims {n}  exact quotes {sum(r['checks']['exact_quotes'] for r in ok)}/{n}  "
              f"failing code checks {sum(r['checks']['claims_failing_code_checks'] for r in ok)}  "
              f"unknown article numbers {sum(r['checks']['unknown_article_refs'] for r in ok)}")
    print(f"\nReport: {html_path}")
    print("Open it in a browser, score each answer with the checklist BEFORE opening the key at the bottom.")
    print("Code checks only show that quotes and numbers come from the supplied articles; they do not show")
    print("that the legal conclusion is right. The manual checklist is the real comparison.")


if __name__ == "__main__":
    asyncio.run(main())
