"""The one module every model call goes through.

Responsibilities: provider clients, per-call timeouts from the request deadline,
one retry on transient errors, JSON parsing + one repair attempt, optional
fallback model, PII guard, provider-independence check, tracing, health status.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import httpx
import yaml
from pydantic import BaseModel, ValidationError

from app.context import RequestCtx
from app.errors import BudgetExceeded, ModelCallError, ModelOutputError, PIILeakError
from app.pii.masker import PIIMasker
from app.tracing import ModelCall

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)

_ARTICLE_BLOCK = re.compile(r"<article\b.*?</article>", re.DOTALL)
RETRYABLE_STATUS ={408, 409, 425, 429, 500, 502, 503, 504}
REPAIR_PROMPT = (
    "Your previous reply was not valid JSON for the required schema. Error: {error}\n"
    "{where}"
    "The usual cause is a double quote character inside a text value: never use the ASCII double quote inside text;"
    " use « » instead. Reply again with ONLY the corrected JSON object, no prose and no code fences."
)


def escape_inner_quotes(raw: str) -> str:
    """Deterministic first repair: escape ASCII double quotes that sit INSIDE a string value.

    Inside a string, a quote counts as closing only if the next non-space character is one of : , } ]
    (or the end); any other quote is escaped. Content is kept exactly (escaping, not replacing), so a claim whose
    quote contained a stray quote still has to match the article text in the code checks."""
    out, in_str, i, n = [], False, 0, len(raw)
    while i < n:
        ch = raw[i]
        if in_str and ch == "\\":
            out.append(raw[i:i + 2])
            i += 2
            continue
        if ch == '"':
            if not in_str:
                in_str = True
            else:
                j = i + 1
                while j < n and raw[j] in " \t\r\n":
                    j += 1
                if j >= n or raw[j] in ":,}]":
                    in_str = False
                else:
                    out.append('\\"')
                    i += 1
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _where(raw: str, e: Exception) -> str:
    pos = getattr(e, "pos", None)
    if pos is None:
        return ""
    return f"The JSON breaks near character {pos}: ...{raw[max(0, pos - 60):pos + 20]}...\n"


@dataclass
class RerankHit:
    index: int
    score: float


@dataclass
class RoleConfig:
    role: str
    provider: str
    model: str
    timeout_s: float
    temperature: float | None = None
    max_tokens: int | None = None
    json_mode: bool = True
    dimensions: int | None = None
    extra_body: dict[str, Any] | None = None   # provider-specific fields, e.g. {"reasoning_effort": "low"}
    fallback: "RoleConfig | None" = None


def extract_json(raw: str) -> Any:
    """Parse a JSON object from a model reply, tolerating code fences and stray prose."""
    text = raw.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError as first_error:
        if (merged := _merge_split_objects(text)) is not None:
            return merged
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                pass
        raise first_error


def _merge_split_objects(text: str) -> dict | None:
    """A reply split into several complete JSON objects ({"summary": ...} then {"sections": ...}).
    Merged only when every part is an object and no key appears twice; otherwise None (not guessed)."""
    dec, i, parts = json.JSONDecoder(), 0, []
    while i < len(text):
        while i < len(text) and text[i] in " \t\r\n,":
            i += 1
        if i >= len(text):
            break
        try:
            obj, i = dec.raw_decode(text, i)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None
        parts.append(obj)
    if len(parts) < 2:
        return None
    merged: dict = {}
    for p in parts:
        if set(p) & set(merged):
            return None
        merged.update(p)
    return merged


class ModelGateway:
    def __init__(self, config: dict, http: httpx.AsyncClient, *, reserve_s: float = 1.5,
                 min_call_s: float = 2.0, env: dict | None = None):
        self.http = http
        self.reserve_s = reserve_s
        self.min_call_s = min_call_s
        self.env = env if env is not None else os.environ
        self.providers: dict[str, dict] = config["providers"]
        self.roles: dict[str, RoleConfig] = {
            name: self._role_cfg(name, raw) for name, raw in config["roles"].items()
        }
        self.status: dict[str, str] = {name: "unknown" for name in self.roles}

    @classmethod
    def from_file(cls, path: str | Path, http: httpx.AsyncClient, **kw) -> "ModelGateway":
        with open(path, encoding="utf-8") as f:
            return cls(yaml.safe_load(f), http, **kw)

    def _role_cfg(self, name: str, raw: dict) -> RoleConfig:
        fb = raw.get("fallback")
        fallback = None
        if fb:
            merged = {**raw, **fb, "fallback": None}
            fallback = self._role_cfg(name, merged)
        return RoleConfig(
            role=name,
            provider=raw["provider"],
            model=raw["model"],
            timeout_s=float(raw.get("timeout_s", 10)),
            temperature=raw.get("temperature"),
            max_tokens=raw.get("max_tokens"),
            json_mode=bool(raw.get("json_mode", True)),
            dimensions=raw.get("dimensions"),
            extra_body=self._extra_body(name, raw.get("extra_body")),
            fallback=fallback,
        )

    @staticmethod
    def _extra_body(role: str, extra: Any) -> dict[str, Any] | None:
        if extra is None:
            return None
        if not isinstance(extra, dict):
            raise RuntimeError(f"models.yaml: role {role}: extra_body must be a mapping")
        reserved = {"model", "messages", "temperature", "max_tokens", "response_format", "input"} & set(extra)
        if reserved:
            raise RuntimeError(f"models.yaml: role {role}: extra_body may not set {sorted(reserved)}")
        return dict(extra)

    # ---- startup checks ----------------------------------------------------
    def validate(self) -> None:
        for name in ("understand", "generate", "verify", "embed", "rerank"):
            if name not in self.roles:
                raise RuntimeError(f"models.yaml: missing role '{name}'")
        gen, ver = self.roles["generate"], self.roles["verify"]
        gen_providers = {gen.provider} | ({gen.fallback.provider} if gen.fallback else set())
        ver_providers = {ver.provider} | ({ver.fallback.provider} if ver.fallback else set())
        if gen_providers & ver_providers:
            raise RuntimeError("generate and verify must use different providers (independent check)")
        for cfg in self.roles.values():
            if cfg.provider not in self.providers:
                raise RuntimeError(f"role {cfg.role}: unknown provider {cfg.provider}")

    def missing_keys(self) -> list[str]:
        return [p["api_key_env"] for p in self.providers.values() if not self.env.get(p["api_key_env"])]

    # ---- public API --------------------------------------------------------
    async def chat_json(self, role: str, messages: list[dict], schema: type[T], ctx: RequestCtx) -> T:
        self._guard([m["content"] for m in messages], ctx)
        cfg: RoleConfig | None = self.roles[role]
        last_error: Exception | None = None
        while cfg is not None:
            try:
                result = await self._chat_json_once(cfg, messages, schema, ctx)
                self.status[role] = "ok"
                return result
            except (ModelCallError, ModelOutputError) as e:
                last_error = e
                log.warning("role=%s model=%s failed: %s", role, cfg.model, e)
                if isinstance(e, BudgetExceeded):
                    break
                cfg = cfg.fallback
        self.status[role] = "error"
        assert last_error is not None
        raise last_error

    async def embed(self, texts: list[str], ctx: RequestCtx) -> list[list[float]]:
        self._guard(texts, ctx)
        cfg = self.roles["embed"]
        prov = self.providers[cfg.provider]
        body: dict[str, Any] = {"model": cfg.model, "input": texts}
        if cfg.dimensions:
            body["dimensions"] = cfg.dimensions
        data = await self._post(cfg, f"{prov['base_url']}/embeddings", body, ctx)
        self.status["embed"] = "ok"
        rows = sorted(data["data"], key=lambda r: r["index"])
        return [r["embedding"] for r in rows]

    async def rerank(self, query: str, docs: list[str], top_n: int, ctx: RequestCtx) -> list[RerankHit]:
        self._guard([query], ctx)
        cfg = self.roles["rerank"]
        prov = self.providers[cfg.provider]
        body = {"model": cfg.model, "query": query, "documents": docs, "top_n": min(top_n, len(docs))}
        data = await self._post(cfg, f"{prov['base_url']}/rerank", body, ctx)
        self.status["rerank"] = "ok"
        return [RerankHit(index=r["index"], score=r["relevance_score"]) for r in data["results"]]

    # ---- internals ---------------------------------------------------------
    def _guard(self, texts: list[str], ctx: RequestCtx) -> None:
        # Law text inside <article> blocks is ours, not the user's: a law that mentions a
        # common name must not block the call.
        user_parts = [_ARTICLE_BLOCK.sub(" ", t) for t in texts if isinstance(t, str)]
        leaked = PIIMasker.leaked(user_parts, ctx.pii_values)
        if leaked:
            raise PIILeakError(f"{len(leaked)} personal value(s) found in outgoing text; call blocked")

    def _timeout(self, cfg: RoleConfig, ctx: RequestCtx) -> float:
        left = ctx.deadline.remaining() - self.reserve_s
        if left < self.min_call_s:
            raise BudgetExceeded(f"{cfg.role}: only {left:.1f}s left")
        return min(cfg.timeout_s, left)

    async def _chat_json_once(self, cfg: RoleConfig, messages: list[dict], schema: type[T], ctx: RequestCtx) -> T:
        raw = await self._chat(cfg, messages, ctx)
        try:
            return schema.model_validate(extract_json(raw))
        except (ValidationError, ValueError) as e:
            ctx.trace.add_call(ModelCall(cfg.role, cfg.model, 0, "invalid_json"))
            # model output holds masked text only (the guard checked the input); log a short snippet to diagnose
            log.warning("role=%s invalid JSON (%s) near: %s", cfg.role, str(e)[:120], _where(raw, e)[:200])
            try:                                   # 1) deterministic local repair, no extra model call
                fixed = schema.model_validate(extract_json(escape_inner_quotes(raw)))
                ctx.trace.add_call(ModelCall(cfg.role, cfg.model, 0, "repaired_locally"))
                return fixed
            except (ValidationError, ValueError):
                pass
            repair = messages + [                  # 2) one model repair, told where it broke
                {"role": "assistant", "content": raw},
                {"role": "user", "content": REPAIR_PROMPT.format(error=str(e)[:400], where=_where(raw, e))},
            ]
            raw2 = await self._chat(cfg, repair, ctx)
            try:
                return schema.model_validate(extract_json(raw2))
            except (ValidationError, ValueError) as e2:
                raise ModelOutputError(f"{cfg.role}: invalid JSON after repair: {str(e2)[:300]}") from e2

    async def _chat(self, cfg: RoleConfig, messages: list[dict], ctx: RequestCtx) -> str:
        prov = self.providers[cfg.provider]
        body: dict[str, Any] = {"model": cfg.model, "messages": messages}
        if cfg.temperature is not None:
            body["temperature"] = cfg.temperature
        if cfg.max_tokens:
            body["max_tokens"] = cfg.max_tokens
        if cfg.json_mode:
            body["response_format"] = {"type": "json_object"}
        if cfg.extra_body:
            body.update(cfg.extra_body)
        data = await self._post(cfg, f"{prov['base_url']}/chat/completions", body, ctx)
        try:
            content = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as e:
            raise ModelCallError(f"{cfg.role}: unexpected response shape") from e
        if not content:
            raise ModelCallError(f"{cfg.role}: empty content")
        return content

    async def _post(self, cfg: RoleConfig, url: str, body: dict, ctx: RequestCtx) -> dict:
        prov = self.providers[cfg.provider]
        key = self.env.get(prov["api_key_env"], "")
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        attempt = 0
        while True:
            attempt += 1
            timeout = self._timeout(cfg, ctx)
            t0 = time.monotonic()
            status = "ok" if attempt == 1 else "retry_ok"
            try:
                # httpx timeouts apply per phase (connect / each read); wait_for caps the whole call
                resp = await asyncio.wait_for(self.http.post(url, json=body, headers=headers, timeout=timeout),
                                              timeout=timeout)
                latency = int((time.monotonic() - t0) * 1000)
                if resp.status_code >= 400:
                    retryable = resp.status_code in RETRYABLE_STATUS
                    ctx.trace.add_call(ModelCall(cfg.role, cfg.model, latency, f"http_{resp.status_code}"))
                    if retryable and attempt == 1 and await self._can_retry(cfg, ctx):
                        continue
                    raise ModelCallError(f"{cfg.role}: HTTP {resp.status_code}: {resp.text[:200]}")
                data = resp.json()
                usage = data.get("usage") or {}
                ctx.trace.add_call(ModelCall(
                    cfg.role, cfg.model, latency, status,
                    usage.get("prompt_tokens") or usage.get("input_tokens"),
                    usage.get("completion_tokens") or usage.get("output_tokens"),
                ))
                return data
            except (httpx.TimeoutException, asyncio.TimeoutError) as e:
                ctx.trace.add_call(ModelCall(cfg.role, cfg.model, int((time.monotonic() - t0) * 1000), "timeout"))
                raise ModelCallError(f"{cfg.role}: timeout after {timeout:.1f}s") from e
            except httpx.TransportError as e:
                ctx.trace.add_call(ModelCall(cfg.role, cfg.model, int((time.monotonic() - t0) * 1000), "error"))
                if attempt == 1 and await self._can_retry(cfg, ctx):
                    continue
                raise ModelCallError(f"{cfg.role}: {type(e).__name__}") from e

    async def _can_retry(self, cfg: RoleConfig, ctx: RequestCtx) -> bool:
        if ctx.deadline.remaining() - self.reserve_s < self.min_call_s + 0.6:
            return False
        await asyncio.sleep(random.uniform(0.3, 0.6))
        return True
