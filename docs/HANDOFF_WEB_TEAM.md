# Backend handoff: Web Team (MVP)

Status: integration-ready MVP; validation results and known limitations: `docs/MVP_STATUS.md`.
This document is the contract; it does not depend on backend internals.
Items marked **[AGREE]** need a decision together during integration.

## 1. Endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/ask` | Ask a legal question (JSON reply, or a stream with `Accept: text/event-stream`) |
| GET | `/api/article/{article_id}` | Full text of a cited article (for a "read the full article" view) |
| GET | `/api/health` | Monitoring: service, database, models, data version, PII status |

Base URL: the deployed backend (Replit) **[AGREE]**. JSON is UTF-8; all user-facing texts are Arabic (RTL).

## 2. Request: `POST /api/ask`

```json
{"question": "المالك بدو يطلعني من الشقة قبل ما يخلص العقد، بيحقله؟"}
```

| field | required | rules |
|---|---|---|
| `question` | yes | 2-2000 characters, any Arabic (dialect or Standard) |
| `clarification` | only in the second round | see section 5 |

Invalid input (missing, too short, too long) returns **HTTP 422** with FastAPI's standard error body.

## 3. Response: `AskResponse` (HTTP 200 for every pipeline outcome)

```json
{
  "request_id": "6ea3f20b04f14de0",
  "type": "answer",                       // answer | clarification | referral | no_support | out_of_scope | error
  "domain": "labour",                     // labour | rentals | social_security | consumer_protection | other | null
  "coverage": "focus",                    // focus (one of the 4 main areas) | initial | null
  "answer": { ... } ,                     // AnswerOut, only for type answer (and optionally referral)
  "clarifying_question": null,            // only for type clarification
  "referral": null,                       // {"reason": "..."} only for type referral
  "message": null,                        // text to show for no_support / out_of_scope / error
  "data_version": {"version": "local-...", "scraped_at": "2026-10-13"},
  "timings_ms": {"understand": 1800, "generate": 7000, "verify": 6000, "total": 15500}
}
```

### What to show for each `type`

| `type` | Show | Notes |
|---|---|---|
| `answer` | `answer` (section 4) | the normal case |
| `clarification` | `clarifying_question` + a reply box | send the reply as in section 5 |
| `referral` | `referral.reason`; also `answer` if present | situations that need a lawyer |
| `no_support` | `message` | no clear legal text found, **or** the system abstained because a complete answer could not be verified |
| `out_of_scope` | `message` | not a question about Jordanian law |
| `error` | `message` | includes "took too long" and generic errors; offer "try again" |

`message` texts are final Arabic texts from the backend; show them as they are.

## 4. The answer: `AnswerOut`

```json
{
  "summary": "...",
  "sections": {
    "rights":      [ClaimOut],
    "obligations": [ClaimOut],
    "steps":       [ClaimOut],
    "conditions":  [ClaimOut]
  },
  "caveats": ["..."],
  "partial": false,
  "amendment_warning": false,
  "verification": "verified"
}
```

`ClaimOut`, one legal statement with its source:

```json
{
  "text": "لكل عامل الحق بإجازة سنوية بأجر كامل مدتها أربعة عشر يوماً عن كل سنة خدمة.",
  "display_text": "لكل عامل الحق بإجازة سنوية بأجر كامل مدتها أربعة عشر يوماً عن كل سنة خدمة. (قانون العمل رقم 8 لسنة 1996، المادة 61)",
  "citation": {
    "article_id": "lob-2-3392:61",
    "quote": "لكل عامل الحق باجازة سنوية بأجر كامل لمدة أربعة عشر يوما عن كل سنة خدمة",
    "label": "قانون العمل رقم 8 لسنة 1996، المادة 61"
  }
}
```

### Citation format (required display)

- Show **`display_text`** for every claim: it already ends with the inline citation `(<law title>، المادة <n>)`.
  Do not show `text` alone.
- `citation.quote` is the exact sentence of the law the claim relies on: show it under the claim or on tap.
- `citation.article_id` opens the full article: `GET /api/article/{article_id}`.
- Backend guarantees: every claim cites a real article of our law data; `label` is built from that article
  (never written by the AI); `quote` is an exact span of that article. A claim that fails these checks is never
  returned.
- Sections may be empty; hide empty sections. Suggested headings: الحقوق، الالتزامات، الخطوات، الشروط.

### Verification status (required display)

| `answer.verification` | Meaning | UI |
|---|---|---|
| `verified` | every shown claim passed the code checks **and** the AI meaning check | normal answer |
| `not_semantically_verified` | the meaning check could not run (time-out or provider problem); shown claims passed the code checks only (real article, exact quote, numbers) | **a visible badge or banner**, e.g. «لم يكتمل التحقق من معنى الإجابة» |

Never style a `not_semantically_verified` answer like a verified one. The summary in that state is a neutral
sentence by design.

### Other answer fields

- `summary`: one to three sentences relating the rules to the user's situation, or a neutral sentence when it
  could not be verified. It contains no legal rule beyond the claims.
- `caveats`: notes (facts the answer depends on, partial or verification notes). Show as a list.
- `partial: true`: some parts were left out because they could not be verified (a caveat already says so).
- `amendment_warning: true`: the law was amended after the user's situation (a caveat already says so).

## 5. Clarification round

When `type == "clarification"`, ask `clarifying_question`, then send:

```json
{"question": "<original question>",
 "clarification": {"original_question": "<original question>",
                   "clarifying_question": "<clarifying_question from the response>",
                   "answer": "<user's reply>"}}
```

At most one clarification per question. The server keeps no state: everything needed is in this request.

## 6. Streaming (optional)

Send `Accept: text/event-stream` to `POST /api/ask` to receive progress events, then the result:

```
event: stage
data: {"stage": "understanding"}      // then "searching", "writing", "checking"

event: result
data: { ...the same AskResponse JSON... }
```

## 7. Other responses

- **HTTP 503** `{"detail": "...", "pii_name_model": "..."}`: the service refuses questions because personal-data
  protection is not ready (production safety). Show "الخدمة غير متاحة مؤقتاً".
- **HTTP 422**: invalid request body.
- `GET /api/article/{id}` → `{"id","law_title","type","number","year","article_no","text","status",
  "last_amendment","gazette","source_url"}`; **HTTP 404** if unknown.
- `GET /api/health` → `{"status": "ok|degraded|unavailable", "database", "models", "data_version",
  "pii_name_model"}`.

## 8. Timing

Hard limit 40 s per request; typical 10-30 s. Use the stream or a progress indicator; set the client timeout
above 40 s (e.g. 45 s).

## 9. Privacy notice (required)

Show above the question box, always:

> لا تكتب اسمك أو رقم هاتفك أو رقمك الوطني أو أي معلومات شخصية في سؤالك. نحاول إخفاء هذه المعلومات تلقائياً
> قبل معالجة السؤال، لكن الإخفاء الآلي قد لا يلتقط كل شيء.

(Source: `app/messages.py`, `PRIVACY_NOTICE`.) The backend masks names, phone numbers, national IDs, IBANs and
emails before any AI call, restores them only in the text shown back to the user, and blocks any call that would
send a detected value. Detection is not perfect (see limitations), hence the notice.

## 10. Implemented vs missing

| Topic | Status |
|---|---|
| Authentication / user accounts | **Not implemented.** The API is open. **[AGREE]** whether the MVP needs a key or login |
| Sessions / conversation history | **Not implemented.** Each request is independent; only the one clarification round is supported (stateless) |
| Rate limiting per user | **Not implemented.** Every question costs AI calls. **[AGREE]** a limit before public exposure |
| CORS | Implemented: set `CORS_ORIGINS=https://<website origin>` on the backend if the site is on another origin. **[AGREE]** the origin |
| Hosting the site | Either on its own origin (needs CORS) or built into `./web` and served by the backend (same origin) **[AGREE]** |
| Feedback / thumbs up-down | Not implemented |

## 11. Samples

`docs/sample_responses/*.json` (all validated by the test suite): `answer_focus_domain`,
`answer_not_semantically_verified`, `clarification`, `referral`, `no_support`, `no_support_unverified`,
`out_of_scope`, `error_timeout`. The `_note` field in each sample is documentation only.
