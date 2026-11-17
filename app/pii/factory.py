"""Build the PII masker from settings and apply the failure policy.

PII_NAME_MODEL=none          rules only (default; identical to the original masker)
PII_NAME_MODEL=camelbert-da  rules + name model (needs requirements-ml.txt and the downloaded model)
PII_FAMILY_NAME_EXPANSION=on experimental, off by default (Phase 3D decides)

If the configured model cannot be loaded at startup:
  APP_ENV=development  -> rules only, with a prominent warning at startup and on every request
  APP_ENV=production   -> the server starts, but /api/ask answers HTTP 503 (fail closed)
If the model fails on a single request:
  development -> that request is masked with rules only (warning logged)
  production  -> that request fails and nothing is sent to any model (PIIModelError)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable

from app.config import Settings
from app.pii.masker import PIIMasker
from app.pii.name_model import SUPPORTED_NAME_MODELS, NameModel, load_name_model

log = logging.getLogger(__name__)
APP_ENVS = ("development", "production")
BANNER = "=" * 78


@dataclass(frozen=True)
class PIIStatus:
    app_env: str
    configured: str            # PII_NAME_MODEL as configured
    active: str                # what is actually masking: "rules" or "rules+<model>"
    serving: bool              # False -> /api/ask must answer 503
    degraded: bool             # True -> configured model missing, rules only (development)
    error: str | None = None
    family_expansion: bool = False   # experimental option, off by default

    def label(self) -> str:
        if self.configured == "none":
            return "none"
        if not self.serving:
            return f"{self.configured}: unavailable, requests blocked"
        if self.degraded:
            return f"{self.configured}: unavailable, rules only"
        return self.configured


def build_masker(settings: Settings,
                 loader: Callable[..., NameModel] = load_name_model) -> tuple[PIIMasker, PIIStatus]:
    env = settings.app_env.strip().lower()
    kind = settings.pii_name_model.strip().lower()
    if env not in APP_ENVS:
        raise ValueError(f"APP_ENV must be one of {APP_ENVS}, got {settings.app_env!r}")
    if kind not in SUPPORTED_NAME_MODELS:
        raise ValueError(f"PII_NAME_MODEL must be one of {SUPPORTED_NAME_MODELS}, got {settings.pii_name_model!r}")
    policy = "fail_closed" if env == "production" else "fallback"
    family = bool(settings.pii_family_name_expansion)
    if family:
        log.warning("PII family-name expansion is ON (experimental; not accepted for production until Phase 3D)")

    if kind == "none":
        return (PIIMasker(family_name_expansion=family),
                PIIStatus(env, kind, "rules", serving=True, degraded=False, family_expansion=family))

    try:
        model = loader(kind, model_id=settings.pii_name_model_id, min_score=settings.pii_name_model_min_score,
                       revision=settings.pii_name_model_revision or None)
    except Exception as e:  # missing package, download failure, unimplemented, out of memory ...
        error = f"{type(e).__name__}: {e}"
        if env == "production":
            log.error("%s\nPII NAME MODEL %r FAILED TO LOAD (%s).\nProduction policy: /api/ask is BLOCKED "
                      "(HTTP 503) until the model loads.\n%s", BANNER, kind, error, BANNER)
            return (PIIMasker(family_name_expansion=family),
                    PIIStatus(env, kind, "rules", serving=False, degraded=True, error=error, family_expansion=family))
        log.warning("%s\nPII NAME MODEL %r FAILED TO LOAD (%s).\nDevelopment policy: masking with RULES ONLY. "
                    "Names outside the rules will NOT be masked.\n%s", BANNER, kind, error, BANNER)
        return (PIIMasker(family_name_expansion=family),
                PIIStatus(env, kind, "rules", serving=True, degraded=True, error=error, family_expansion=family))

    masker = PIIMasker(name_model=model, model_error_policy=policy, family_name_expansion=family)
    log.info("PII masking: rules + %s (%s policy on per-request errors, min_score=%.2f)", kind, policy,
             settings.pii_name_model_min_score)
    return masker, PIIStatus(env, kind, f"rules+{kind}", serving=True, degraded=False, family_expansion=family)
