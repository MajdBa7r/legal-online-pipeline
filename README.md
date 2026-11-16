# Jordanian Legal AI Assistant: online pipeline

Backend for `POST /api/ask`, `GET /api/article/{id}` and `GET /api/health`, owned by the Core System team.
It follows the "Online Pipeline Architecture" doc. This package has no data (that's the Data team's
Supabase) and no website (the Web and Accuracy team's front end goes in `./web/`, which FastAPI serves
if it exists).

```
question -> 1 mask PII -> 2 understand -> 3 gate -> 4a domain pack | 4b search + rerank
         -> 5 generate -> 6 verify (code + model) -> 7 decide (rewrite once) -> restore PII -> response
```

## Run it

```bash
pip install -r requirements.txt
cp .env.example .env            # locally; on Replit put these in Secrets
pytest                          # 32 tests, no keys or network needed
python scripts/check_models.py  # Day 1: every model role answers, with latency
python scripts/check_supabase.py  # Day 1: DB latency, data version, pack sizes, hybrid_search
uvicorn app.main:app --reload --port 8000
python scripts/smoke_test.py "كم يوم إجازة سنوية بستحق؟" --show
```

Streaming progress for the website: send `Accept: text/event-stream` to `/api/ask`. You get `stage` events
(`understanding`, `searching`, `writing`, `checking`) and then one `result` event with the final JSON.
Sample responses for the website team are in `docs/sample_responses/`.

## Layout and owners

| Path | What | Owner |
| --- | --- | --- |
| `app/main.py` | Routes, SSE, startup, health | A |
| `app/orchestrator.py` | The 7 stages, deadline, rewrite loop, response building | A + B |
| `app/pii/masker.py`, `common_names.txt` | Stage 1 masking and restore | A |
| `app/gateway/` | Model gateway and `models.yaml` (roles to models) | A |
| `app/stages/understand.py`, `gate.py` | Stages 2 and 3 | A |
| `app/stages/retrieve.py` | Stages 4a and 4b (packs, hybrid search, RRF, rerank, expansion) | B |
| `app/stages/generate.py`, `verify.py`, `decide.py` | Stages 5, 6 and 7 | B |
| `app/verify_checks/` | Quote, number (incl. Arabic number words) checks | B |
| `app/prompts/` | Prompts and few-shot examples | A (understand), B (generate, verify) |
| `app/schemas/api.py` | Public API contract with the website | Both, agreed with the Web team |
| `app/schemas/llm.py` | JSON schema per model role | Both |
| `app/db/store.py`, `schema_map.py` | Supabase access; all table and column names | B |
| `app/rules/gate_rules.yaml` | Referral keywords, thresholds | Legal members |
| `app/messages.py` | Fixed Arabic texts shown to users | Legal members + Web team |
| `sql/` | Proposed `hybrid_search` contract and `traces` table for the Data team | B |

## Before it works on real data (Day 1 checklist)

- [ ] **Model ids**: replace the placeholder ids in `app/gateway/models.yaml` with the exact ids from each
      provider, then run `scripts/check_models.py`. If a provider rejects `temperature` or `response_format`,
      set it to `null` / `json_mode: false` for that role.
- [ ] **Schema names**: align `app/db/schema_map.py` with the Data team's final tables (entries marked ASSUMED).
- [ ] **hybrid_search**: agree `sql/hybrid_search_contract.sql` with the Data team, especially `filter_domain`,
      the NULL-embedding (full-text-only) mode, and that the text-search config matches `chunks.fts`.
- [ ] **Normalization**: `app/text/arabic.py::normalize` must match the Data team's normalized text exactly.
- [ ] **Traces**: Data team creates the table from `sql/traces_table.sql` (or tells us their columns).
- [ ] **Few-shot examples**: replace the 2 placeholders in `app/prompts/examples.jsonl` with the 15-20 reviewed ones.
- [ ] **Gate rules**: legal members review `app/rules/gate_rules.yaml` and the national ID rule in `masker.py`.
- [ ] **Pack sizes**: `scripts/check_supabase.py` prints chars per pack; set `PACK_MAX_CHARS` so a pack
      fits the generation budget, otherwise that domain uses domain-filtered search.
- [ ] **Latency**: measure Replit to Supabase and each model; adjust `timeout_s` per role and `BUDGET_S`.

## Rules the code enforces

- No answer without evidence: every claim has one article ID and a quote that is an exact span of the
  supplied article (whitespace-collapsed), numbers in the claim appear in the article, the article is in force,
  and a second model from a **different provider** agrees it is supported (`ModelGateway.validate()` refuses
  to start otherwise).
- At most two generations per question; a rewrite starts only if `RETRY_MIN_REMAINING_S` seconds are left.
- 30-second budget: every model timeout is `min(role timeout, time left - reserve)`.
- Nothing personal is sent or logged: the gateway blocks any call whose user text still contains a masked
  value; traces store IDs, timings and counts only.

## Tuning knobs (env vars, see `app/config.py`)

`MATCH_COUNT`, `CANDIDATES`, `RERANK_TOP_N`, `RERANK_QUERY` (`legal_rewrite` or `original`),
`EXPANSION_MAX`, `PACK_MAX_CHARS`, `MIN_QUOTE_CHARS`, `RETRY_MIN_REMAINING_S`, `BUDGET_S`.
`app/stages/decide.py::SEMANTIC_SKIPPED_POLICY` decides what happens when the verify model is down
(open decision with the legal members).

## Test-set runs

Start the server with `EVAL_TOKEN=<secret>` and send requests with header `X-Eval-Token: <secret>`;
only those traces store the question and response text (test questions hold no personal data).
`DEMO_CACHE=true` caches full answers for repeated questions (demo warm-up with `scripts/smoke_test.py --file`).
