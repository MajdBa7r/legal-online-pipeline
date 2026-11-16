"""Per-request state: deadline, trace, PII originals, progress events."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Awaitable, Callable

from app.tracing import Trace

EmitFn = Callable[[str], Awaitable[None]]


async def _no_emit(_: str) -> None:
    return None


class Deadline:
    def __init__(self, seconds: float):
        self.start = time.monotonic()
        self.end = self.start + seconds

    def remaining(self) -> float:
        return self.end - time.monotonic()

    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self.start) * 1000)


@dataclass
class RequestCtx:
    deadline: Deadline
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    trace: Trace = field(default_factory=Trace)
    pii_values: list[str] = field(default_factory=list)  # originals the gateway must never send
    emit: EmitFn = _no_emit
