"""Local development store: the scraper's JSON files instead of Supabase.

It implements the same `Store` interface as `SupabaseStore` and returns the same
types (`Article`, `SearchHit`, `GlossaryEntry`, `DataVersion`), so the pipeline
cannot tell the difference. Swapping to Supabase later is a config change.

Rules this module follows:
* The scraped JSON files are opened read-only and never modified. All mapping,
  normalization and chunking happen in memory, here.
* Status is never assumed. Law-level status comes from the source's displayed
  status; an unknown value is treated as NOT in force (fail closed) and reported.
  Article-level repeal is only set when the article text itself says so.
* Search mirrors sql/hybrid_search_contract.sql: keyword ranking plus an optional
  vector ranking, fused with Reciprocal Rank Fusion, in-force articles only,
  optional domain filter, chunk-level hits with their article ID.
  The keyword side is BM25 over light-stemmed tokens, a local stand-in for
  PostgreSQL full-text search: numbers measured here are a baseline, to be
  measured again on the Data team's database.
"""
from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.schemas.domain import Article, DataVersion, GlossaryEntry, SearchHit
from app.text.arabic import clean_display, collapse_ws, normalize, search_tokens

log = logging.getLogger(__name__)

# Displayed status (after normalize()) -> (internal status, amended?)
STATUS_MAP: dict[str, tuple[str, bool]] = {
    normalize("ساري"): ("in_force", False),
    normalize("ساري معدل"): ("in_force", True),
    normalize("غير ساري"): ("repealed", False),
    normalize("غير ساري معدل"): ("repealed", True),
}
# Bureau LegislationType codes seen in law_id ("lob-<type>-<id>"). Only laws (2) are scraped so far.
LOB_TYPES = {"2": "law"}

DEFINITIONS_MARKERS = (normalize("المعاني المخصصة لها"), normalize("يكون للكلمات والعبارات"))
_REPEALED_ONLY = re.compile(
    r"^(?:الماده\s*\(?\s*\d+\s*\)?\s*[-–:]\s*)?[\(\[]?\s*(ملغاه|ملغيه|ملغي|الغيت|تلغي)\s*[\)\]]?\s*\.?$"
)
_SELF_HEADING = re.compile(r"^\s*(?:ال)?ماد[ةه]\s*\(?\s*(\d+)\s*\)?\s*[-–:]")
_EMBEDDED_HEADING = re.compile(r"(?:^|\n)\s*(?:ال)?ماد[ةه]\s*\(?\s*(\d+)\s*\)?\s*(مكرر[ةه]?)?\s*[:\-–]")
_INVISIBLE = re.compile(r"[​-‏‪-‮⁦-⁩﻿]")
_CHAPTER_IN_TEXT = re.compile(r"^\s*(?:الفصل|الباب)\s+\S+")
_REPLACEMENT_CHAR = re.compile(r"[\u0621-\u064A]\?|\?[\u0621-\u064A]|\ufffd")
_ZERO_AS_STOP = re.compile(r"[\u0621-\u064A\)]\s+0(?:\s|$)")
_SCHEDULE_MARKERS = ("جدول رقم", "الجدول رقم", "قائمة الأمراض", "قائمة الامراض", "الجداول الملحقة")
_OR = re.compile(r"(?<![\w؀-ۿ])or(?![\w؀-ۿ])")

BM25_K1 = 1.5
BM25_B = 0.75
RRF_K = 50          # same default as the SQL contract
MAX_RESULTS = 50    # the SQL contract caps match_count at 50


@dataclass
class Chunk:
    id: str
    article_id: str
    law_id: str
    index: int
    text: str                       # header + chunk text (what gets embedded)
    tf: Counter = field(default_factory=Counter)
    length: int = 0


@dataclass
class LawRecord:
    law_id: str
    title: str
    title_raw: str
    status_raw: str | None
    status: str
    amended: bool
    domain: str | None
    file: str
    file_sha256: str
    scraped_at: str | None
    article_ids: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def split_into_chunks(text: str, max_chars: int = 1200) -> list[str]:
    """Split an article by paragraph (line), merging short lines, splitting very long ones.

    Long articles are split "by paragraph into chunks that keep their article ID" (blueprint 3.2).
    """
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    pieces: list[str] = []
    for ln in lines:
        if len(ln) <= max_chars:
            pieces.append(ln)
            continue
        # very long line (often a flattened table): split at sentence ends, then at spaces
        sentences = re.split(r"(?<=[.؛:])\s+", ln)
        buf = ""
        for s in sentences:
            while len(s) > max_chars:
                cut = s.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                if buf:
                    pieces.append(buf)
                    buf = ""
                pieces.append(s[:cut].strip())
                s = s[cut:].strip()
            if len(buf) + len(s) + 1 > max_chars and buf:
                pieces.append(buf)
                buf = s
            else:
                buf = f"{buf} {s}".strip()
        if buf:
            pieces.append(buf)
    chunks: list[str] = []
    buf = ""
    for p in pieces:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n{p}" if buf else p
    if buf:
        chunks.append(buf)
    return chunks or [text.strip()]


class LocalStore:
    def __init__(self, data_dir: str | Path, *, domains_file: str | Path | None = None,
                 traces_path: str | Path | None = None, embeddings_dir: str | Path | None = None,
                 glossary_file: str | Path | None = None, chunk_max_chars: int = 1200):
        self.data_dir = Path(data_dir)
        self.traces_path = Path(traces_path) if traces_path else None
        self.chunk_max_chars = chunk_max_chars
        self.domains = self._load_domains(domains_file)
        self.glossary_file = Path(glossary_file) if glossary_file else self.data_dir / "glossary.csv"
        self.laws: dict[str, LawRecord] = {}
        self.articles: dict[str, Article] = {}
        self.chunks: list[Chunk] = []
        self.chunks_by_id: dict[str, Chunk] = {}
        self.df: Counter = Counter()
        self.avgdl = 0.0
        self.vectors: dict[str, list[float]] = {}
        self.vector_meta: dict[str, Any] = {}
        self.warnings: list[str] = []
        self._load_files()
        self._build_index()
        if embeddings_dir:
            self._load_vectors(Path(embeddings_dir))

    # ------------------------------------------------------------------ load
    @staticmethod
    def _load_domains(path: str | Path | None) -> dict[str, dict]:
        p = Path(path) if path else Path(__file__).with_name("local_domains.yaml")
        if not p.exists():
            return {}
        with open(p, encoding="utf-8") as f:
            return (yaml.safe_load(f) or {}).get("laws", {}) or {}

    def _load_files(self) -> None:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"local data folder not found: {self.data_dir}")
        files = sorted(self.data_dir.glob("*.json"))
        if not files:
            raise FileNotFoundError(f"no scraped JSON files in {self.data_dir}")
        for path in files:
            with open(path, "rb") as f:          # read-only, bytes: the file is never written
                raw = json.loads(f.read().decode("utf-8"))
            if "law_id" not in raw or "articles" not in raw:
                self.warnings.append(f"{path.name}: not a scraped law file, skipped")
                continue
            self._add_law(raw, path)

    def _add_law(self, raw: dict, path: Path) -> None:
        law_id = str(raw["law_id"])
        status_raw = raw.get("status_as_displayed_ar")
        status, amended = STATUS_MAP.get(normalize(status_raw or ""), ("unknown", False))
        if status == "unknown":
            self.warnings.append(f"{law_id}: unknown status {status_raw!r}; treated as NOT in force")
        type_code = law_id.split("-")[1] if law_id.count("-") >= 2 else ""
        law_type = LOB_TYPES.get(type_code, f"lob_type_{type_code}")
        amend_years = [r.get("target_release_year") for r in raw.get("legislation_references") or []
                       if r.get("relationship") == "amended_by" and r.get("target_release_year")]
        last_amendment = str(max(amend_years)) if amend_years else None
        gazette = raw.get("gazette") or {}
        gazette_txt = (f"العدد {gazette.get('issue_number')}، ص {gazette.get('page_number')}"
                       if gazette.get("issue_number") else None)
        domain = (self.domains.get(law_id) or {}).get("domain")
        title = clean_display(raw.get("title_ar") or law_id)
        rec = LawRecord(law_id=law_id, title=title, title_raw=raw.get("title_ar") or "", status_raw=status_raw,
                        status=status, amended=amended, domain=domain, file=path.name,
                        file_sha256=_sha256(path), scraped_at=raw.get("scraped_at"), raw=raw)

        for a in sorted(raw["articles"], key=lambda x: x.get("position") or 0):
            number = str(a.get("number") or a.get("position"))
            art_id = f"{law_id}:{number}"
            if art_id in self.articles:
                self.warnings.append(f"{law_id}: duplicate article number {number}; later one skipped")
                continue
            text = a.get("text_ar") or ""
            body = _SELF_HEADING.sub("", normalize(text), count=1).strip()
            repeal_evidence = None
            if _REPEALED_ONLY.match(body):
                repeal_evidence = "article text reads only as repealed (ملغاة)"
            self.articles[art_id] = Article(
                id=art_id, article_no=number, text=text, repealed=repeal_evidence is not None,
                legislation_id=law_id, law_title=title, law_type=law_type,
                law_number=str(raw.get("law_number") or ""), law_year=str(raw.get("release_year") or ""),
                law_status=status, last_amendment=last_amendment,
                source_url=raw.get("official_source_url") or raw.get("source_url"),
                gazette=gazette_txt, domain=domain, coverage_level="focus" if domain else "initial",
                law_status_raw=status_raw, law_amended=amended, repeal_evidence=repeal_evidence,
                chapter=a.get("chapter_ar"), article_source_url=a.get("source_url"),
            )
            rec.article_ids.append(art_id)
        self.laws[law_id] = rec

    def _build_index(self) -> None:
        for art in self.articles.values():
            for i, piece in enumerate(split_into_chunks(art.text, self.chunk_max_chars)):
                text = f"{art.header()}\n{piece}"
                tokens = search_tokens(text)
                ch = Chunk(id=f"{art.id}#{i}", article_id=art.id, law_id=art.legislation_id, index=i,
                           text=text, tf=Counter(tokens), length=len(tokens))
                self.chunks.append(ch)
                self.chunks_by_id[ch.id] = ch
                self.df.update(set(tokens))
        self.avgdl = (sum(c.length for c in self.chunks) / len(self.chunks)) if self.chunks else 0.0

    def _load_vectors(self, emb_dir: Path) -> None:
        files = sorted(emb_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime) if emb_dir.is_dir() else []
        if not files:
            return
        path = files[-1]
        with open(path, encoding="utf-8") as f:
            meta = json.loads(f.readline())
            for line in f:
                row = json.loads(line)
                if row["chunk_id"] in self.chunks_by_id:
                    self.vectors[row["chunk_id"]] = row["vector"]
        self.vector_meta = {**meta, "file": path.name}
        missing = len(self.chunks) - len(self.vectors)
        if missing:
            self.warnings.append(f"embeddings file {path.name} covers {len(self.vectors)}/{len(self.chunks)} "
                                 f"chunks; rebuild it with scripts/build_local_embeddings.py")

    # ---------------------------------------------------------------- search
    def _eligible(self, chunk: Chunk, domain: str | None) -> bool:
        art = self.articles[chunk.article_id]
        return art.in_force() and (domain is None or art.domain == domain)

    def _bm25_ranking(self, query_text: str, domain: str | None) -> list[str]:
        terms = list(dict.fromkeys(search_tokens(_OR.sub(" ", query_text))))
        if not terms or not self.chunks:
            return []
        n = len(self.chunks)
        scores: dict[str, float] = {}
        for t in terms:
            df = self.df.get(t, 0)
            if df == 0:
                continue
            idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
            for ch in self.chunks:
                f = ch.tf.get(t)
                if not f:
                    continue
                denom = f + BM25_K1 * (1 - BM25_B + BM25_B * ch.length / (self.avgdl or 1))
                scores[ch.id] = scores.get(ch.id, 0.0) + idf * f * (BM25_K1 + 1) / denom
        ranked = sorted(scores, key=lambda cid: -scores[cid])
        return [cid for cid in ranked if self._eligible(self.chunks_by_id[cid], domain)]

    def _vector_ranking(self, embedding: list[float], domain: str | None) -> list[str]:
        if not self.vectors:
            return []
        qn = math.sqrt(sum(x * x for x in embedding)) or 1.0
        sims = []
        for cid, vec in self.vectors.items():
            if len(vec) != len(embedding) or not self._eligible(self.chunks_by_id[cid], domain):
                continue
            dot = sum(a * b for a, b in zip(embedding, vec))
            vn = math.sqrt(sum(x * x for x in vec)) or 1.0
            sims.append((dot / (qn * vn), cid))
        sims.sort(reverse=True)
        return [cid for _, cid in sims]

    async def hybrid_search(self, query_text: str, embedding: list[float] | None, match_count: int,
                            domain: str | None = None, timeout: float = 3.0,
                            full_text_weight: float = 1.0, semantic_weight: float = 1.0,
                            rrf_k: int = RRF_K) -> list[SearchHit]:
        limit = min(match_count, MAX_RESULTS)
        ft = self._bm25_ranking(query_text, domain)[: limit * 2]
        sem = self._vector_ranking(embedding, domain)[: limit * 2] if embedding is not None else []
        if embedding is not None and not self.vectors:
            log.debug("embedding given but no local chunk vectors loaded: keyword-only search")
        score: dict[str, float] = {}
        for rank, cid in enumerate(ft, start=1):
            score[cid] = score.get(cid, 0.0) + full_text_weight / (rrf_k + rank)
        for rank, cid in enumerate(sem, start=1):
            score[cid] = score.get(cid, 0.0) + semantic_weight / (rrf_k + rank)
        best = sorted(score, key=lambda cid: -score[cid])[:limit]
        return [SearchHit(article_id=self.chunks_by_id[cid].article_id, score=score[cid], chunk_id=cid)
                for cid in best]

    # -------------------------------------------------------------- articles
    async def get_articles(self, ids: list[str]) -> dict[str, Article]:
        return {str(i): self.articles[str(i)] for i in ids if str(i) in self.articles}

    async def load_domain_packs(self) -> dict[str, list[str]]:
        packs: dict[str, list[str]] = {}
        for rec in self.laws.values():
            if rec.domain:
                packs.setdefault(rec.domain, []).extend(rec.article_ids)
        return packs

    def definitions_article(self, law_id: str) -> str | None:
        rec = self.laws.get(law_id)
        if not rec:
            return None
        for art_id in rec.article_ids[:6]:
            head = normalize(self.articles[art_id].text[:400])
            if any(m in head for m in DEFINITIONS_MARKERS):
                return art_id
        return None

    async def definitions_for(self, legislation_ids: list[str]) -> list[str]:
        # Local data has no parent-law links (only laws are scraped); definitions come from the law itself.
        out = []
        for lid in dict.fromkeys(legislation_ids):
            d = self.definitions_article(lid)
            if d:
                out.append(d)
        return out

    async def load_glossary(self) -> list[GlossaryEntry]:
        if not self.glossary_file.exists():
            return []
        out = []
        with open(self.glossary_file, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                phrase = (row.get("dialect_phrase") or "").strip()
                if phrase:
                    out.append(GlossaryEntry(phrase=phrase, legal_term=(row.get("legal_term") or "").strip(),
                                             domain=(row.get("domain") or "").strip() or None,
                                             phrase_norm=normalize(phrase)))
        return out

    async def latest_data_version(self) -> DataVersion | None:
        if not self.laws:
            return None
        digest = hashlib.sha256("".join(f"{r.file}:{r.file_sha256}" for r in
                                        sorted(self.laws.values(), key=lambda r: r.file)).encode()).hexdigest()
        scraped = max((r.scraped_at or "" for r in self.laws.values()), default="")
        return DataVersion(version=f"local-{digest[:12]}", scraped_at=scraped[:10] or None,
                           embedding_model=self.vector_meta.get("model"),
                           embedding_dim=self.vector_meta.get("dim"),
                           extra={"source": "local", "laws": len(self.laws), "articles": len(self.articles),
                                  "chunks": len(self.chunks), "vectors": len(self.vectors)})

    async def insert_trace(self, row: dict[str, Any]) -> None:
        if not self.traces_path:
            return
        try:
            self.traces_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.traces_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        except OSError as e:
            log.warning("local trace write failed: %s", e)

    async def ping(self) -> bool:
        return bool(self.articles)

    # --------------------------------------------------------------- report
    def quality_report(self) -> dict[str, dict[str, Any]]:
        """Facts about each loaded law, including known scraper issues. Nothing is fixed here."""
        report: dict[str, dict[str, Any]] = {}
        for rec in self.laws.values():
            arts = [self.articles[i] for i in rec.article_ids]
            raw = rec.raw
            numbers = [a.get("number") for a in raw["articles"]]
            merged, self_heading, urls, invisible = [], 0, [], 0
            for a in arts:
                for m in _EMBEDDED_HEADING.finditer(a.text):
                    if m.start() < 3 and m.group(1) == a.article_no and not m.group(2):
                        self_heading += 1
                    else:
                        merged.append(f"{a.article_no} contains 'المادة {m.group(1)}"
                                      f"{' ' + m.group(2) if m.group(2) else ''}' at char {m.start()}")
                if "http" in a.text:
                    urls.append(a.article_no)
                if _INVISIBLE.search(a.text):
                    invisible += 1
            longest = sorted(arts, key=lambda a: -len(a.text))[:3]
            chapter_in_text = [a.article_no for a in arts if _CHAPTER_IN_TEXT.match(a.text)]
            odd_chars = [a.article_no for a in arts if _REPLACEMENT_CHAR.search(a.text)]
            zero_stops = [a.article_no for a in arts if _ZERO_AS_STOP.search(a.text)]
            schedules = [a.article_no for a in arts if len(a.text) > 3000 and any(
                ln.strip().startswith(_SCHEDULE_MARKERS) for ln in a.text.split("\n")[1:])]
            report[rec.law_id] = {
                "title": rec.title,
                "domain": rec.domain,
                "file": rec.file,
                "status_raw": rec.status_raw,
                "status": rec.status,
                "law_amended": rec.amended,
                "last_amendment": arts[0].last_amendment if arts else None,
                "articles_loaded": len(arts),
                "articles_displayed": raw.get("article_count_as_displayed"),
                "articles_in_force": sum(1 for a in arts if a.in_force()),
                "articles_repealed_by_text": [a.article_no for a in arts if a.repealed],
                "article_level_status_in_source": False,
                "empty_articles": [a.article_no for a in arts if not a.text.strip()],
                "duplicate_numbers": [n for n, c in Counter(numbers).items() if c > 1],
                "non_numeric_numbers": [n for n in numbers if not str(n).isdigit()],
                "longest_articles": [(a.article_no, len(a.text)) for a in longest],
                "merged_article_headings": merged,
                "self_heading_articles": self_heading,
                "articles_with_urls": urls,
                "articles_with_invisible_chars": invisible,
                "articles_without_chapter": sum(1 for a in arts if not a.chapter),
                "chapter_heading_inside_text": chapter_in_text,
                "possible_appended_schedules": schedules,
                "question_mark_or_replacement_char": odd_chars,
                "zero_used_as_full_stop": zero_stops,
                "chunks": sum(1 for c in self.chunks if c.law_id == rec.law_id),
                "definitions_article": self.definitions_article(rec.law_id),
                "total_chars": sum(len(a.text) for a in arts),
            }
        return report
