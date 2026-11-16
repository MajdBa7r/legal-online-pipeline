"""Retrieval smoke evaluation shared by scripts/eval_retrieval.py and the real-data tests.

It sends each query through the same path the pipeline uses for one wording
(build_fts_query -> store.hybrid_search), collapses chunk hits to articles in
rank order, and scores hit@k and MRR against the expected article IDs.

This is a developer smoke set, not the Accuracy team's test set. It checks that
the keyword baseline finds the obvious articles; the official numbers come from
the expert-reviewed test set later.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from app.text.arabic import build_fts_query

KS = (1, 3, 5, 10)


@dataclass
class QueryResult:
    id: str
    domain: str
    style: str
    query: str
    expected: list[str]
    ranked: list[str]

    @property
    def first_rank(self) -> int | None:
        for i, aid in enumerate(self.ranked, start=1):
            if aid in self.expected:
                return i
        return None


def load_queries(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.startswith("//")]


async def run_queries(store, queries: list[dict], *, match_count: int = 30, use_domain_filter: bool = False,
                      embed_fn=None) -> list[QueryResult]:
    out = []
    for q in queries:
        embedding = await embed_fn(q["query"]) if embed_fn else None
        hits = await store.hybrid_search(build_fts_query(q["query"]), embedding, match_count,
                                         q["domain"] if use_domain_filter else None)
        ranked = list(dict.fromkeys(h.article_id for h in hits))
        out.append(QueryResult(q["id"], q["domain"], q["style"], q["query"], q["expected"], ranked))
    return out


def summarize(results: list[QueryResult]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QueryResult]] = defaultdict(list)
    for r in results:
        groups["all"].append(r)
        groups[f"style={r.style}"].append(r)
        groups[f"domain={r.domain}"].append(r)
    table = {}
    for name, rs in groups.items():
        row = {"n": float(len(rs))}
        for k in KS:
            row[f"hit@{k}"] = sum(1 for r in rs if r.first_rank and r.first_rank <= k) / len(rs)
        row["mrr"] = sum(1 / r.first_rank for r in rs if r.first_rank) / len(rs)
        table[name] = row
    return table
