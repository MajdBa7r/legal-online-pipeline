# Backend handoff: Data Team (MVP)

Validation results and known limitations: `docs/MVP_STATUS.md`.

What the online pipeline needs from the legal data, how it uses it today (local files), and what must be agreed
before it reads from your database. Items marked **[AGREE]** are open decisions; nothing here assumes how you
have built your side.

## 1. Two data sources behind one interface

The pipeline never reads files or tables directly. It calls a `Store` with eight operations
(`app/db/store.py`):

| Operation | Used for |
|---|---|
| `get_articles(ids)` | fetch articles (text + law metadata) by id |
| `load_domain_packs()` | the article ids of each focus domain, in order |
| `hybrid_search(query_text, embedding, match_count, domain)` | find articles for other questions |
| `definitions_for(law_ids)` | the definitions article of a law (added to the context) |
| `load_glossary()` | dialect phrase → legal term hints |
| `latest_data_version()` | detect new data and reload |
| `insert_trace(row)` | one log row per request (no question text, no personal data) |
| `ping()` | health check |

Two implementations, chosen with `STORE_BACKEND`:
- `local` (**today**): `LocalStore` reads the scraper's JSON files from `data/dev/` (read-only).
- `supabase` (**after integration**): `SupabaseStore` reads Postgres/pgvector tables through Supabase.
  Its table and column names live in one file, `app/db/schema_map.py`; names marked ASSUMED there are our guesses.

## 2. What the backend needs for every article

| Field | Why | Today (scraper JSON) |
|---|---|---|
| article id | stable reference in citations and `/api/article/{id}` | built: `<law_id>:<article number>`, e.g. `lob-2-3392:61` |
| law id | group articles by law | `law_id`, e.g. `lob-2-3392` |
| article number | shown in every citation: «المادة 61» | `articles[].number` (fallback `position`) |
| article text | sent to the AI and quoted exactly | `articles[].text_ar` |
| law title | shown in every citation | `title_ar` |
| law type | law / regulation / instructions | from the id: `lob-2-…` = law |
| law number, year | metadata, allowed numbers in claims | `law_number`, `release_year` |
| law status | only laws in force are used | `status_as_displayed_ar` (table below) |
| article repealed | repealed articles are never used | text that reads only «ملغاة» |
| last amendment year | warning when the user's case predates it | max `target_release_year` of `legislation_references` with `relationship = amended_by` |
| domain | which focus area the law belongs to | `app/db/local_domains.yaml` (not in the scraper output) |
| gazette, source URL | shown in the full-article view | `gazette.issue_number/page_number`, `official_source_url` |
| order of articles | domain packs keep the law's order | `articles[].position` |

Status mapping used today (anything else = **not in force**, fail closed):

| `status_as_displayed_ar` | status | amended |
|---|---|---|
| ساري | in_force | no |
| ساري معدل | in_force | yes |
| غير ساري | repealed | no |
| غير ساري معدل | repealed | yes |

## 3. How articles are retrieved

- **Four focus domains** (labour, rentals, social_security, consumer_protection), one law each today
  (`local_domains.yaml`). For a question in one of them, the backend sends the **whole law** (all in-force
  articles, in order, plus its definitions article) to the AI, if it is at most 180,000 characters. Above that, it
  searches inside the domain instead. Current sizes: 13,510 to 142,468 characters.
- **Any other question**: hybrid search with up to three wordings (the question, a Standard Arabic rewrite, a
  legal-terms rewrite): keyword ranking (BM25) plus vector similarity, merged with Reciprocal Rank Fusion (k=50,
  at most 50 results). Only in-force articles of in-force laws are returned.
- **Chunks**: articles are split into pieces of at most 1,200 characters; each piece is indexed with a header
  (law title, type, article number). A search hit points back to its article.
- **Embeddings**: `text-embedding-3-large`, 1,024 dimensions, cosine similarity. The MVP runs without an OpenAI key,
  so search is keyword-only for now; vectors can be added later without code changes.
- **Definitions article**: detected by the phrases «يكون للكلمات والعبارات» / «المعاني المخصصة لها» in the first
  articles of a law.

## 4. Adding or updating legal data

**Today (local files):**
1. Put the scraper's JSON file for the law in `data/dev/` (one file per law, unchanged; the backend never edits it).
2. If it belongs to a focus domain, add its `law_id` to `app/db/local_domains.yaml`.
3. Restart the backend. The data version is a hash of the files, so it changes automatically.
4. Run `python scripts/inspect_corpus.py` to see the quality report (merged headings, appended schedules, broken
   characters, invisible characters).

**After integration (database):** load the new data, then insert a new row in `data_versions`. The backend checks
for a new version every 5 minutes and reloads packs and glossary without a restart. **[AGREE]** the load procedure
and who inserts the version row.

## 5. The database contract (proposal; you own the final schema)

Files to review together:
- `app/db/schema_map.py`: every table and column name the backend uses (`legislation`, `articles`, `chunks`,
  `domain_packs`, `glossary`, `relations`, `data_versions`, `traces`). Rename there, nothing else changes.
- `sql/hybrid_search_contract.sql`: the search function the backend calls (`hybrid_search(query_text,
  query_embedding vector(1024) default null, match_count, full_text_weight, semantic_weight, rrf_k,
  filter_domain)` → `article_id, chunk_id, score`). Requirements: works with a null embedding (keyword-only),
  filters by domain, returns only in-force articles of in-force laws, receives an already normalized, OR-joined
  query text.
- `sql/traces_table.sql`: the log table the backend writes (one row per request; no question text, answers or
  personal data, except test questions explicitly marked with a secret header).

## 6. Items to agree during integration

1. **[AGREE] Article ids.** Citations expose `article_id` to the website (`/api/article/{id}`), and traces store
   them. Today they are strings like `lob-2-3392:61`; the database proposal uses numbers. We need ids that stay
   the same when a law is re-scraped or updated.
2. **[AGREE] Domains.** Who assigns a law to a domain (today a hand-written YAML), and the domain values
   (`labour`, `rentals`, `social_security`, `consumer_protection`, `other`).
3. **[AGREE] Status and repeal.** The status values and how a repealed article is marked (today inferred from
   «ملغاة» text).
4. **[AGREE] Article text cleaning.** The backend quotes articles exactly and checks numbers in them. Known defects
   in the current files: missing spaces (Labour Law art. 61: «إذاانقضت», «وانقضتالسنتان»), headings merged into
   text, schedules appended to the last article, broken characters. The backend works around these, but
   corrections belong in the data. Any cleaning must keep the official wording.
5. **[AGREE] Amendments.** How the last amendment year is stored (today from `legislation_references`).
6. **[AGREE] Embeddings.** Model and dimension (proposal: `text-embedding-3-large`, 1,024), chunk size, index type
   (HNSW with cosine), and who generates them.
7. **[AGREE] Glossary.** Columns `dialect_phrase`, `legal_term`, `domain` (today an optional CSV).
8. **[AGREE] Data versions.** Format of the version id and the scrape date.
9. **[AGREE] Access.** The backend needs a Supabase URL and a service key (read on content tables, insert on
   `traces`); where the key is stored (Replit Secrets).

## 7. What to send us for a first database test

One law (for example the Labour Law) loaded in your schema, the `hybrid_search` function, and one `data_versions`
row. We run the same questions against `STORE_BACKEND=local` and `STORE_BACKEND=supabase` and compare the
articles and citations returned.
