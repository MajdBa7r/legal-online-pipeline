"""FastAPI app: /api/ask, /api/article/{id}, /api/health. Runs on Replit."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.config import Settings
from app.context import Deadline, RequestCtx
from app.db.store import SupabaseStore
from app.gateway.client import ModelGateway
from app.orchestrator import Pipeline
from app.pii.masker import PIIMasker
from app.resources import Resources
from app.schemas.api import ArticleOut, AskRequest, AskResponse, DataVersionOut, HealthOut

settings = Settings.from_env()
logging.basicConfig(level=settings.log_level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI):
    http = httpx.AsyncClient(limits=httpx.Limits(max_connections=50, max_keepalive_connections=20))
    gw = ModelGateway.from_file(settings.models_file, http, reserve_s=settings.reserve_s,
                                min_call_s=settings.min_call_s)
    gw.validate()
    if missing := gw.missing_keys():
        log.warning("missing API keys: %s", ", ".join(missing))
    store = SupabaseStore(settings.supabase_url, settings.supabase_service_key, http)
    res = Resources.from_settings(settings)
    try:
        await res.load(store)
    except Exception as e:  # start anyway; /api/health reports degraded
        log.error("initial resource load failed: %s", e)
    app.state.pipeline = Pipeline(gw, store, res, PIIMasker(), settings)
    app.state.store, app.state.gw, app.state.res = store, gw, res
    app.state.bg_tasks = set()
    yield
    await http.aclose()


app = FastAPI(title="Jordanian Legal AI Assistant API", lifespan=lifespan)


def _keep(app_: FastAPI, coro) -> None:
    task = asyncio.create_task(coro)
    app_.state.bg_tasks.add(task)
    task.add_done_callback(app_.state.bg_tasks.discard)


def _trace_row(ctx: RequestCtx, req: AskRequest, resp: AskResponse, request: Request) -> dict:
    row = ctx.trace.to_row(ctx.request_id, ctx.deadline.elapsed_ms())
    row["response_type"] = resp.type
    token = request.headers.get("x-eval-token")
    if settings.eval_token and token == settings.eval_token:
        # test-set questions only: they contain no personal data
        row["eval_question"] = req.question
        row["eval_response"] = resp.model_dump()
    return row


@app.post("/api/ask", response_model=AskResponse)
async def ask(req: AskRequest, request: Request):
    pipeline: Pipeline = request.app.state.pipeline
    wants_stream = "text/event-stream" in request.headers.get("accept", "")

    if not wants_stream:
        ctx = RequestCtx(deadline=Deadline(settings.budget_s))
        resp = await pipeline.handle(req, ctx)
        _keep(request.app, request.app.state.store.insert_trace(_trace_row(ctx, req, resp, request)))
        return resp

    queue: asyncio.Queue[tuple[str, dict]] = asyncio.Queue()

    async def emit(stage: str) -> None:
        await queue.put(("stage", {"stage": stage}))

    ctx = RequestCtx(deadline=Deadline(settings.budget_s), emit=emit)

    async def run():
        resp = await pipeline.handle(req, ctx)
        await queue.put(("result", resp.model_dump()))
        _keep(request.app, request.app.state.store.insert_trace(_trace_row(ctx, req, resp, request)))

    async def stream():
        task = asyncio.create_task(run())
        while True:
            event, data = await queue.get()
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            if event == "result":
                break
        await task

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/article/{article_id}", response_model=ArticleOut)
async def article(article_id: str, request: Request):
    arts = await request.app.state.store.get_articles([article_id])
    a = arts.get(article_id)
    if a is None:
        raise HTTPException(status_code=404, detail="article not found")
    return ArticleOut(id=a.id, law_title=a.law_title, type=a.law_type, number=a.law_number or None,
                      year=a.law_year or None, article_no=a.article_no, text=a.text,
                      status="in_force" if a.in_force() else "repealed", last_amendment=a.last_amendment,
                      gazette=a.gazette, source_url=a.source_url)


@app.get("/api/health", response_model=HealthOut)
async def health(request: Request):
    db_ok = await request.app.state.store.ping()
    res: Resources = request.app.state.res
    models = dict(request.app.state.gw.status)
    dv = res.data_version
    loaded = bool(res.packs) and any(res.packs.values())
    ok = db_ok and loaded and not any(v == "error" for v in models.values())
    return HealthOut(status="ok" if ok else "degraded", database=db_ok, models=models,
                     data_version=DataVersionOut(version=dv.version if dv else None,
                                                 scraped_at=dv.scraped_at if dv else None))


# The Web and Accuracy team's built front end goes in ./web (not part of this package).
_web = Path(__file__).resolve().parent.parent / "web"
if _web.is_dir():
    app.mount("/", StaticFiles(directory=_web, html=True), name="web")
