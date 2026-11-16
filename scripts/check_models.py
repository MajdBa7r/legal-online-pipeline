"""Day 1 check: call every model role once and print status and latency.

    python scripts/check_models.py

Needs the API keys in the environment (Replit Secrets). Sends no personal data.
"""
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402
from app.schemas.llm import VerifierOutput  # noqa: E402

PROBE = [{"role": "system", "content": 'Reply with JSON only: {"verdicts": []}'},
         {"role": "user", "content": "اختبار"}]


async def main():
    s = Settings.from_env()
    async with httpx.AsyncClient() as http:
        gw = ModelGateway.from_file(s.models_file, http)
        gw.validate()
        if missing := gw.missing_keys():
            print("missing keys:", ", ".join(missing))
        for role in ("understand", "generate", "verify"):
            ctx = RequestCtx(deadline=Deadline(60))
            t0 = time.monotonic()
            try:
                await gw.chat_json(role, PROBE, VerifierOutput, ctx)
                print(f"{role:10s} ok     {int((time.monotonic() - t0) * 1000):5d} ms  model={gw.roles[role].model}")
            except Exception as e:
                print(f"{role:10s} FAILED {e}")
        ctx = RequestCtx(deadline=Deadline(60))
        t0 = time.monotonic()
        try:
            vecs = await gw.embed(["ما مدة الإجازة السنوية للعامل؟"], ctx)
            print(f"embed      ok     {int((time.monotonic() - t0) * 1000):5d} ms  dim={len(vecs[0])}")
        except Exception as e:
            print(f"embed      FAILED {e}")
        t0 = time.monotonic()
        try:
            hits = await gw.rerank("الإجازة السنوية", ["مدة الإجازة السنوية", "عقوبة السير"], 2, ctx)
            print(f"rerank     ok     {int((time.monotonic() - t0) * 1000):5d} ms  top={hits[0].index}")
        except Exception as e:
            print(f"rerank     FAILED {e}")


if __name__ == "__main__":
    asyncio.run(main())
