"""Choose the storage backend from settings. The pipeline only ever sees the `Store` interface."""
from __future__ import annotations

import httpx

from app.config import Settings
from app.db.store import Store, SupabaseStore


def make_store(settings: Settings, http: httpx.AsyncClient) -> Store:
    backend = settings.store_backend.strip().lower()
    if backend == "local":
        from app.db.local_store import LocalStore

        return LocalStore(settings.local_data_dir, traces_path=settings.local_traces_path,
                          embeddings_dir=settings.local_embeddings_dir,
                          chunk_max_chars=settings.chunk_max_chars)
    if backend == "supabase":
        return SupabaseStore(settings.supabase_url, settings.supabase_service_key, http)
    raise ValueError(f"STORE_BACKEND must be 'local' or 'supabase', got {settings.store_backend!r}")
