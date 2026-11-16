"""Internal objects passed between stages (not part of the public API)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

FOCUS_DOMAINS = ("labour", "rentals", "consumer_protection", "social_security")

LAW_TYPE_AR = {"law": "قانون", "regulation": "نظام", "instructions": "تعليمات"}


@dataclass
class Article:
    id: str
    article_no: str
    text: str
    repealed: bool = False
    legislation_id: str = ""
    law_title: str = ""
    law_type: str = "law"            # law | regulation | instructions
    law_number: str = ""
    law_year: str = ""
    law_status: str = "in_force"     # in_force | repealed
    last_amendment: str | None = None
    source_url: str | None = None
    gazette: str | None = None
    domain: str | None = None
    coverage_level: str | None = None
    # Provenance and status detail (optional; the Supabase store may leave them unset)
    law_status_raw: str | None = None   # status exactly as the source displays it
    law_amended: bool = False           # law-level "amended" marker from the source
    repeal_evidence: str | None = None  # why an article is treated as repealed, if it is
    chapter: str | None = None
    article_source_url: str | None = None

    def in_force(self) -> bool:
        return not self.repealed and self.law_status == "in_force"

    def label(self) -> str:
        """Short citation label shown on the website, e.g. 'قانون العمل، المادة 23'."""
        return f"{self.law_title}، المادة {self.article_no}"

    def header(self) -> str:
        """Same header format the Data team prepends to chunks before embedding."""
        return f"[{self.law_title} – {LAW_TYPE_AR.get(self.law_type, self.law_type)} – المادة {self.article_no}]"

    def amendment_year(self) -> int | None:
        if not self.last_amendment:
            return None
        m = re.search(r"(19|20)\d{2}", str(self.last_amendment))
        return int(m.group(0)) if m else None

    def metadata_numbers(self) -> set[float]:
        """Numbers a claim may mention without them appearing in the article text."""
        out: set[float] = set()
        for v in (self.article_no, self.law_number, self.law_year, self.last_amendment):
            for m in re.findall(r"\d+", str(v or "")):
                out.add(float(m))
        return out


@dataclass
class GlossaryEntry:
    phrase: str
    legal_term: str
    domain: str | None
    phrase_norm: str = ""


@dataclass
class SearchHit:
    article_id: str
    score: float
    chunk_id: str | None = None


@dataclass
class DataVersion:
    version: str
    scraped_at: str | None = None
    embedding_model: str | None = None
    embedding_dim: int | None = None
    extra: dict = field(default_factory=dict)
