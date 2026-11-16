"""Settings read from environment variables (Replit Secrets in production)."""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent


def _cast(value: str, typ):
    if typ is bool:
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if typ is int:
        return int(value)
    if typ is float:
        return float(value)
    return value


@dataclass(frozen=True)
class Settings:
    # Storage backend: "local" (scraped JSON files, development) or "supabase" (Data team's database)
    store_backend: str = "local"
    local_data_dir: str = str(PROJECT_DIR / "data" / "dev")
    local_embeddings_dir: str = str(PROJECT_DIR / "data" / "dev" / "embeddings")
    local_traces_path: str = str(PROJECT_DIR / "logs" / "traces.jsonl")
    chunk_max_chars: int = 1200

    # Supabase (service key lives only on the backend)
    supabase_url: str = ""
    supabase_service_key: str = ""

    # Time budget (seconds)
    budget_s: float = 30.0
    reserve_s: float = 1.5          # kept back from every model timeout
    min_call_s: float = 2.0         # no model call starts with less than this left
    retry_min_remaining_s: float = 12.0  # a rewrite needs at least this much left

    # Retrieval (stage 4b) - tune on the test set
    match_count: int = 30           # hybrid_search results per wording
    candidates: int = 40            # fused articles sent to the reranker
    rerank_top_n: int = 20
    expansion_max: int = 5          # definitions / parent-law articles added
    rerank_query: str = "legal_rewrite"  # or "original"

    # Domain packs (stage 4a)
    pack_max_chars: int = 180_000   # above this, search inside the domain instead
    pack_fallback_top_n: int = 40

    # Verification
    min_quote_chars: int = 8

    # Files
    models_file: str = str(APP_DIR / "gateway" / "models.yaml")
    rules_file: str = str(APP_DIR / "rules" / "gate_rules.yaml")
    prompts_dir: str = str(APP_DIR / "prompts")

    # Ops
    eval_token: str = ""            # X-Eval-Token value that allows storing test-question text in traces
    demo_cache: bool = False        # cache full responses (demo warm-up)
    resource_refresh_s: int = 300   # how often to check data_versions for a new load
    log_level: str = "INFO"

    @classmethod
    def from_env(cls) -> "Settings":
        kwargs = {}
        for f in fields(cls):
            raw = os.environ.get(f.name.upper())
            if raw is not None and raw != "":
                kwargs[f.name] = _cast(raw, type(f.default))
        return cls(**kwargs)
