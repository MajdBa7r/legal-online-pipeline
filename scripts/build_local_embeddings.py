"""Embed every local chunk once and save the vectors for hybrid search during development.

    python scripts/build_local_embeddings.py           # needs OPENAI_API_KEY (see models.yaml, role "embed")
    python scripts/build_local_embeddings.py --dry-run # count chunks and characters, call nothing

Only law text is sent (chunks start with the "[law – type – article]" header, as the Data team plans).
Output: data/dev/embeddings/<model>-<dim>.jsonl (first line = metadata). The scraped JSON files are not touched.
"""
import argparse
import asyncio
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402

from app.config import Settings  # noqa: E402
from app.context import Deadline, RequestCtx  # noqa: E402
from app.db.local_store import LocalStore  # noqa: E402
from app.gateway.client import ModelGateway  # noqa: E402


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    s = Settings.from_env()
    store = LocalStore(s.local_data_dir, chunk_max_chars=s.chunk_max_chars)
    chunks = store.chunks
    total_chars = sum(len(c.text) for c in chunks)
    print(f"{len(chunks)} chunks, {total_chars} characters")
    if args.dry_run:
        return

    async with httpx.AsyncClient() as http:
        gw = ModelGateway.from_file(s.models_file, http)
        cfg = gw.roles["embed"]
        cfg.timeout_s = 60  # offline batch job, not bound by the request budget
        out_dir = Path(s.local_embeddings_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cfg.model}-{cfg.dimensions}.jsonl"
        tmp_path = out_path.with_suffix(".tmp")
        t0 = time.monotonic()
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"model": cfg.model, "dim": cfg.dimensions, "chunk_max_chars": s.chunk_max_chars,
                                "chunks": len(chunks), "created_at": datetime.now(timezone.utc).isoformat()}) + "\n")
            for i in range(0, len(chunks), args.batch):
                batch = chunks[i : i + args.batch]
                vecs = await gw.embed([c.text for c in batch], RequestCtx(deadline=Deadline(120)))
                for c, v in zip(batch, vecs):
                    f.write(json.dumps({"chunk_id": c.id, "vector": v}) + "\n")
                print(f"  {min(i + args.batch, len(chunks))}/{len(chunks)}")
        tmp_path.replace(out_path)
        print(f"saved {out_path} in {time.monotonic() - t0:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
