# Product Scope: Jordanian Legal AI Assistant (MVP)

Status date: 2026-11-17. Based on the code in this repository (backend) and `docs/MVP_STATUS.md`.
Status labels used: **Implemented and tested**, **Implemented, not fully validated**, **Planned / not
implemented**, **Web Team**, **Data Team**, **To agree** (decision needed during integration),
**Needs confirmation** (cannot be confirmed from this repository).

---

## Part 1. In plain language

### What it is

A website where anyone in Jordan can ask a question about their legal rights in everyday language, including
Jordanian dialect («المالك بدو يطلعني من الشقة قبل ما يخلص العقد، بيحقله؟»), and get a short, structured
answer in Arabic that is based only on the text of Jordanian laws. Every legal statement in the answer shows the
law and article it comes from, with the exact sentence of the law it relies on.

### The problem it solves

People often do not know which law applies to their situation, cannot read long legal texts, and cannot easily
reach a lawyer for simple questions. General AI chatbots answer confidently but may invent rules or articles. This
assistant answers only from official law text, shows its sources, checks its own statements, and says so when it
cannot give a reliable answer.

### Who it is for

- Workers, tenants, landlords, consumers and insured people with everyday legal questions.
- People who write in Jordanian dialect or Standard Arabic.
- It is **not** a replacement for a lawyer, and it does not handle court cases or criminal matters.

### What it covers today

Four laws, one per area ("focus domains"), taken from the official Legislation and Opinion Bureau website
(lob.gov.jo):

| Area | Law | Articles |
|---|---|---|
| Work (labour) | قانون العمل رقم 8 لسنة 1996 | 142 |
| Renting (landlords and tenants) | قانون المالكين والمستأجرين رقم 11 لسنة 1994 | 22 |
| Social security | قانون الضمان الاجتماعي رقم 1 لسنة 2014 | 110 |
| Consumer protection | قانون حماية المستهلك رقم 7 لسنة 2017 | 27 |

Questions about other areas are accepted, but the system can only search these same four laws: it either finds
a relevant article in them or answers "no clear legal text found". This path has not been tested end to end.

### What the user experiences

1. Reads a short privacy notice (do not type your name, phone number or national ID) and types a question.
2. Waits about 15-35 seconds while the answer is prepared (progress can be shown).
3. Gets one of these results:
   - **An answer**: a short opening sentence, then the rules grouped as rights, obligations, steps and conditions.
     Each rule ends with its source, for example «(قانون العمل رقم 8 لسنة 1996، المادة 61)», and the exact
     sentence of the law can be shown under it. Notes list facts the answer depends on.
   - **An answer marked "not fully checked"**: shown with a visible warning when the automatic meaning check could
     not run; every rule still comes from a real article with an exact quote.
   - **A clarifying question**, such as «في أي سنة بدأ عقد الإيجار؟», when one missing fact changes the answer.
     The user replies once and gets the answer.
   - **A referral to a lawyer**, for court cases, arrests or high-risk situations.
   - **"We could not verify a complete answer"**, when part of the answer the user needs could not be confirmed
     against the law. The system prefers saying nothing to saying something misleading.
   - **"Not a legal question"** or **"No clear legal text found"**.
   - **An error message** ("took too long" or a general error) with the option to try again.

### Privacy

Before the question is sent to any AI service, names, phone numbers, national ID numbers, bank account numbers
(IBAN) and email addresses are replaced with placeholders, and any call that would still contain one of them is
blocked. The user's own details are put back only in the answer shown to them. Detection is not perfect, mainly
for names, which is why the privacy notice asks users not to type personal details.

### What it does not do (MVP)

No user accounts, no saved conversations, no follow-up chat beyond one clarifying question, no laws beyond the four
above, no documents or file uploads, no English answers, no answers about court procedures or criminal matters,
and no legal advice on a specific dispute.

---

## Part 2. Technical description

### Product capabilities and status

| Capability | Where | Status |
|---|---|---|
| Accept questions in dialect or Standard Arabic (2-2,000 chars) | `app/schemas/api.py` `AskRequest` | Implemented and tested |
| Understand intent: domain, confidence, Standard Arabic and legal rewrites, issues, missing facts, risk | `app/stages/understand.py`, `app/prompts/understand.md` (DeepSeek `deepseek-flash`) | Implemented and tested (with test doubles); validated on 4 real questions |
| Decision gate: out of scope, referral, clarification, continue | `app/stages/gate.py`, `app/rules/gate_rules.yaml` | Implemented and tested; keyword list is a starter list to be reviewed by the legal members |
| Retrieve law text: whole law for the 4 focus domains | `app/stages/retrieve.py` (`pack`), `app/resources.py` | Implemented, tested, validated on 4 real questions |
| Retrieve law text for other questions: 3 wordings, keyword search, RRF | `app/stages/retrieve.py` (`search`), `app/db/local_store.py` | Implemented and tested; keyword-only; not validated end to end; only the same 4 laws are searchable |
| Embeddings (semantic search) and reranking | `app/gateway` roles `embed` (OpenAI), `rerank` (Cohere) | Code present, **not active** (no keys, no embedding files) |
| Generate a structured answer with claims, quotes, essential flags | `app/stages/generate.py`, `app/prompts/generate.md` (DeepSeek `deepseek-flash`) | Implemented and tested; validated on 4 real questions |
| Citations: every claim cites a real article; inline display text | `app/stages/verify.py` `code_checks`, `app/orchestrator.py` `_answer_out` | Implemented and tested |
| Verification: code checks + meaning check of claims and summary | `app/stages/verify.py`, `app/prompts/verify.md` (Moonshot `kimi-k2.6`) | Implemented and tested; validated on 4 real questions (see limitations) |
| Safe outcomes: essential-claim abstention, partial answers, neutral summary, verification status | `app/stages/decide.py` | Implemented and tested |
| Clarification round (one, stateless) | `app/orchestrator.py`, `compose_question` | Implemented and tested |
| PII masking: rules for phone, national ID, IBAN, email, names | `app/pii/masker.py` | Implemented and tested |
| PII names with CAMeLBERT-da | `app/pii/name_model.py`, `PII_NAME_MODEL=camelbert-da` | Implemented, not fully validated (development set only); code default is `none`, deployment setting **To agree / Needs confirmation** |
| Leak guard before every external AI call | `app/gateway/client.py` `_guard` | Implemented and tested |
| 40-second hard limit per question | `app/orchestrator.py` | Implemented and tested |
| JSON API + optional progress stream | `app/main.py` | JSON implemented and tested; stream implemented, **not tested** |
| Full article view | `GET /api/article/{id}` | Implemented |
| Request logs without question text | `insert_trace` → `logs/traces.jsonl` | Implemented |
| Website UI, privacy notice display, status badges | Web Team | **Web Team** (not in this repository) |
| Legal data collection and future database | Data Team | **Data Team** |
| Deployment (Replit) | - | **Planned / not implemented** |
| Authentication, user accounts, sessions, history, per-user rate limits, feedback | - | **Planned / not implemented** (**To agree**) |

### Supported question types

| Type | Behaviour |
|---|---|
| Rights and obligations within the 4 focus domains | Main use case: whole-law context, cited answer |
| Questions with a missing deciding fact | One clarifying question, then the answer |
| High-risk or court/criminal questions (risk "high" or keywords such as توقيف، مخفر، جناية) | Referral to a lawyer |
| Legal questions outside the 4 domains | Keyword search over the same 4 laws; an answer only if one of them applies, otherwise "no clear legal text found" (not validated end to end) |
| Non-legal questions | "Not a legal question" |

### How results are presented (API contract: `docs/HANDOFF_WEB_TEAM.md`)

- `type` decides the screen: `answer`, `clarification`, `referral`, `no_support`, `out_of_scope`, `error`.
- Each claim: `display_text` (claim + «(law title، المادة n)»), `citation.quote` (exact law sentence),
  `citation.article_id` (opens the full article). Labels are built from the stored article, never from AI text.
- `answer.verification`: `verified` or `not_semantically_verified` (must be shown visibly).
- `answer.partial`, `answer.amendment_warning`, `answer.caveats`; `summary` is a neutral sentence when it could
  not be verified.

### Known limitations (from `docs/MVP_STATUS.md`)

1. Abstentions: 2 of 4 test questions ended with "could not verify a complete answer" (an essential claim
   dropped a condition of the article; a rewrite rarely fits in 40 s).
2. The personalised summary was replaced by the neutral sentence in 4 of 4 test answers.
3. The checker is occasionally over-strict or gives a wrong reason; it never let an unsupported claim pass in review.
4. Clarifying questions are frequent (3 of 4 test questions).
5. 17-33 s per question.
6. PII (development set, rules + CAMeLBERT): 83% of names fully masked, 7.9% of sentences leak a value; 100% for
   phones, national IDs, IBANs, emails. Misses names with an attached preposition; occasional over-masking.
   Not validated on a held-out set or on the deployment server (+720 MB memory).
7. Only 4 laws; keyword-only search outside them; no embeddings.
8. Open API: no authentication or rate limiting.

### Outside the MVP scope

Laws beyond the four; other legislation types (regulations, instructions) in the data; court judgments; English;
documents and uploads; multi-turn conversation; user accounts; payment; lawyer marketplace; mobile apps.

### Future development (not started)

Claim-level repair instead of abstaining; better summaries; fewer clarifying questions; database integration with
the Data Team (Supabase, embeddings, more laws); held-out PII evaluation and deployment measurement;
authentication, rate limiting, feedback; deployment and monitoring.
