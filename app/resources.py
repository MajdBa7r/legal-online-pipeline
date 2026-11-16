"""Everything loaded once at startup and kept in memory: domain packs, glossary,
prompts, few-shot examples, gate rules. Reloaded when data_versions changes.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from app.config import Settings
from app.db.store import Store
from app.schemas.domain import FOCUS_DOMAINS, Article, DataVersion, GlossaryEntry
from app.text.arabic import normalize

log = logging.getLogger(__name__)


@dataclass
class GateRules:
    referral_keywords: list[str] = field(default_factory=list)
    referral_risk_levels: list[str] = field(default_factory=lambda: ["high"])
    clarify_on_blocking_missing_fact: bool = True
    min_domain_confidence: float = 0.6
    keywords_norm: list[str] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: str | Path) -> "GateRules":
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        rules = cls(
            referral_keywords=raw.get("referral_keywords", []),
            referral_risk_levels=raw.get("referral_risk_levels", ["high"]),
            clarify_on_blocking_missing_fact=raw.get("clarify_on_blocking_missing_fact", True),
            min_domain_confidence=float(raw.get("min_domain_confidence", 0.6)),
        )
        rules.keywords_norm = [normalize(k) for k in rules.referral_keywords]
        return rules


@dataclass
class Prompts:
    understand: str
    generate: str
    verify: str
    examples: list[dict]

    @classmethod
    def from_dir(cls, path: str | Path) -> "Prompts":
        p = Path(path)
        examples = []
        ex_file = p / "examples.jsonl"
        if ex_file.exists():
            for line in ex_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("//"):
                    examples.append(json.loads(line))
        return cls(
            understand=(p / "understand.md").read_text(encoding="utf-8"),
            generate=(p / "generate.md").read_text(encoding="utf-8"),
            verify=(p / "verify.md").read_text(encoding="utf-8"),
            examples=examples,
        )


@dataclass
class Resources:
    settings: Settings
    prompts: Prompts
    rules: GateRules
    packs: dict[str, list[Article]] = field(default_factory=dict)
    pack_chars: dict[str, int] = field(default_factory=dict)
    glossary: list[GlossaryEntry] = field(default_factory=list)
    data_version: DataVersion | None = None
    loaded_at: float = 0.0
    last_check: float = 0.0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @classmethod
    def from_settings(cls, settings: Settings) -> "Resources":
        return cls(settings=settings, prompts=Prompts.from_dir(settings.prompts_dir),
                   rules=GateRules.from_file(settings.rules_file))

    def glossary_hits(self, text: str, limit: int = 15) -> list[GlossaryEntry]:
        norm = normalize(text)
        hits = [g for g in self.glossary if g.phrase_norm and g.phrase_norm in norm]
        hits.sort(key=lambda g: -len(g.phrase_norm))  # longest phrase first
        return hits[:limit]

    def pack_fits(self, domain: str) -> bool:
        return 0 < self.pack_chars.get(domain, 0) <= self.settings.pack_max_chars

    async def load(self, store: Store) -> None:
        async with self._lock:
            version = await store.latest_data_version()
            pack_ids = await store.load_domain_packs()
            glossary = await store.load_glossary()
            packs: dict[str, list[Article]] = {}
            all_ids = [i for d in FOCUS_DOMAINS for i in pack_ids.get(d, [])]
            arts = await store.get_articles(all_ids) if all_ids else {}
            for d in FOCUS_DOMAINS:
                packs[d] = [arts[i] for i in pack_ids.get(d, []) if i in arts and arts[i].in_force()]
            defs = await store.definitions_for(
                list({a.legislation_id for arts_ in packs.values() for a in arts_}))
            extra = await store.get_articles([i for i in defs if i not in arts]) if defs else {}
            arts.update(extra)
            for d, lst in packs.items():
                laws = {a.legislation_id for a in lst}
                have = {a.id for a in lst}
                for i in defs:
                    a = arts.get(i)
                    if a and a.legislation_id in laws and a.id not in have and a.in_force():
                        lst.insert(0, a)  # definitions first
                        have.add(a.id)
            self.packs = packs
            self.pack_chars = {d: sum(len(a.text) for a in lst) for d, lst in packs.items()}
            self.glossary = glossary
            self.data_version = version
            self.loaded_at = self.last_check = time.time()
            log.info("resources loaded: version=%s packs=%s glossary=%d",
                     version.version if version else None,
                     {d: (len(v), self.pack_chars[d]) for d, v in packs.items()}, len(glossary))

    async def refresh_if_stale(self, store: Store) -> None:
        now = time.time()
        if now - self.last_check < self.settings.resource_refresh_s or self._lock.locked():
            return
        self.last_check = now
        try:
            latest = await store.latest_data_version()
            if latest and (not self.data_version or latest.version != self.data_version.version):
                await self.load(store)
        except Exception as e:  # never fail a request because of a refresh
            log.warning("resource refresh failed: %s", e)
