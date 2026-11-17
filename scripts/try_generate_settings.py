"""Time realistic WRITE (generate) and CHECK (verify) calls with a few models, to choose roles in models.yaml.

    python scripts/try_generate_settings.py

Uses the real generate prompt, real labour-law articles from data/dev (LocalStore, keyword search) and a
synthetic question with no personal data. Each setting is called once with a 40 s limit, so the real time is
visible even when it is above the pipeline's limits (generate 14 s, verify 6 s). The verify calls check the
claims written by the first successful generate call. Costs a few cents. Changes nothing.
"""
import asyncio
import copy
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import yaml  # noqa: E402

from app.config import Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.factory import make_store  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.resources import Prompts  # noqa: E402
from app.schemas.llm import GeneratedAnswer, Understanding  # noqa: E402
from app.schemas.llm import VerifierOutput  # noqa: E402
from app.stages.generate import build_messages  # noqa: E402
from app.stages.verify import _semantic_messages  # noqa: E402

QUESTION = "اشتغلت سنتين بالشركة وما أخذت ولا يوم إجازة، كم يوم إجازة سنوية إلي وشو بصير بالأيام اللي ما أخذتها؟"
UNDERSTANDING = Understanding(
    msa_rewrite="عملت سنتين في الشركة ولم آخذ أي إجازة، فكم يوما من الإجازة السنوية أستحق وما مصير الأيام غير المستخدمة؟",
    legal_rewrite="مدة الإجازة السنوية للعامل وحكم الإجازات غير المستعملة",
    domain="labour", domain_confidence=0.9, legal_issues=["الإجازة السنوية", "الإجازات غير المستعملة"])

OFF = {"thinking": {"type": "disabled"}}   # DeepSeek V4 and Kimi K2.6 think by default; this turns it off
WRITERS = [  # (label, provider, model, extra_body)
    ("deepseek-flash, thinking disabled", "deepseek", "deepseek-flash", OFF),
    ("deepseek-v4-pro, thinking disabled", "deepseek", "deepseek-v4-pro", OFF),
]
CHECKERS = [
    ("kimi-k2.6, thinking disabled", "moonshot", "kimi-k2.6", OFF),
    ("deepseek-flash (reference only)", "deepseek", "deepseek-flash", OFF),
]


async def main():
    s = Settings.from_env()
    with open(s.models_file, encoding="utf-8") as f:
        base = yaml.safe_load(f)
    prompts = Prompts.from_dir(s.prompts_dir)

    async with httpx.AsyncClient() as http:
        store = make_store(s, http)
        hits = await store.hybrid_search(UNDERSTANDING.msa_rewrite, None, 6, domain="labour")
        arts = await store.get_articles([h.article_id for h in hits])
        articles = [arts[h.article_id] for h in hits if h.article_id in arts]
        print(f"{len(articles)} labour-law articles: " + ", ".join(f"art. {a.article_no}" for a in articles))

        class Res:  # build_messages only needs res.prompts
            pass
        res = Res()
        res.prompts = prompts
        messages = build_messages(QUESTION, UNDERSTANDING, articles, res)
        print(f"prompt size: {sum(len(m['content']) for m in messages)} characters\n")

        async def timed(role, label, provider, model, extra, msgs, schema, limit):
            cfg = copy.deepcopy(base)
            r = cfg["roles"][role]
            r.update({"provider": provider, "model": model, "timeout_s": 40, "fallback": None})
            if provider == "moonshot":
                r["temperature"] = None            # Kimi models fix their own temperature
            r.pop("extra_body", None)
            if extra:
                r["extra_body"] = extra
            gw = ModelGateway(cfg, http)
            t0 = time.monotonic()
            try:
                out = await gw.chat_json(role, msgs, schema, RequestCtx(deadline=Deadline(90)))
                secs = time.monotonic() - t0
                fits = f"fits {limit} s" if secs <= limit else f"TOO SLOW for {limit} s"
                return out, f"{label:34s} ok      {secs:5.1f} s  ({fits})"
            except Exception as e:
                return None, f"{label:34s} FAILED  {time.monotonic() - t0:5.1f} s  {str(e)[:160]}"

        print("WRITE (generate), limit 14 s")
        answer = None
        for label, provider, model, extra in WRITERS:
            out, line = await timed("generate", label, provider, model, extra, messages, GeneratedAnswer, 14)
            if out is not None:
                line += f"; {len(out.claims())} claims, status={out.status}"
                answer = answer or out
            print("  " + line)

        if answer is None or not answer.claims():
            print("\nno answer with claims to check; skipping verify timings")
            return
        by_id = {a.id: a for a in articles}
        items = [(c, by_id[c.article_id]) for _, c in answer.claims() if c.article_id in by_id]
        vmsgs = _semantic_messages(items, res)
        print(f"\nCHECK (verify) {len(items)} claims, limit 6 s")
        for label, provider, model, extra in CHECKERS:
            out, line = await timed("verify", label, provider, model, extra, vmsgs, VerifierOutput, 6)
            if out is not None:
                line += f"; {len(out.verdicts)} verdicts"
            print("  " + line)


if __name__ == "__main__":
    asyncio.run(main())
