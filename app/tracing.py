"""One `traces` row per request: timings, models, article IDs, outcome. Never personal data."""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelCall:
    role: str
    model: str
    latency_ms: int
    status: str            # ok | retry_ok | error | timeout | invalid_json | repaired
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class Trace:
    timings_ms: dict[str, int] = field(default_factory=dict)
    calls: list[ModelCall] = field(default_factory=list)
    fields: dict[str, Any] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str):
        t0 = time.monotonic()
        try:
            yield
        finally:
            self.timings_ms[name] = self.timings_ms.get(name, 0) + int((time.monotonic() - t0) * 1000)

    def set(self, **kwargs: Any) -> None:
        self.fields.update(kwargs)

    def add_call(self, call: ModelCall) -> None:
        self.calls.append(call)

    def to_row(self, request_id: str, total_ms: int) -> dict[str, Any]:
        tokens_in = sum(c.input_tokens or 0 for c in self.calls)
        tokens_out = sum(c.output_tokens or 0 for c in self.calls)
        return {
            "request_id": request_id,
            "timings_ms": {**self.timings_ms, "total": total_ms},
            "model_calls": [c.__dict__ for c in self.calls],
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            **self.fields,
        }
