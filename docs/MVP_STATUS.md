# MVP status and known limitations (2026-11-17)

Backend version: integration-ready MVP. Models: DeepSeek Flash (understand, write), Kimi K2.6 (check).
Request budget: 40 s (hard). PII: rules + CAMeLBERT-da + gateway leak guard.

## Final validation: the 4 end-to-end questions (real models, run 2026-11-17 15:13)

| Question | Outcome | Time | Why |
|---|---|---|---|
| Labour: two years without annual leave | **answer, verified** (8/8 claims) | 17 s | all claims supported; the summary drew a conclusion not in the claims (on deferred leave) and was replaced by the neutral sentence |
| Rentals: eviction before the contract ends | **answer, verified, partial** (6/7 claims) | 33 s* | one non-essential claim removed (the checker treated «يلتزم» vs the article's «يلزم» as a difference: over-strict); summary replaced (it concluded "no eviction ground" itself) |
| Social security: early retirement at 47 | **abstained** (`no_support`) | 22 s | an essential claim (the 216-contribution rule) was judged partial; the checker's stated reason is wrong (the claim does say "before the law took effect"), but the claim also presents a pension-calculation paragraph as a separate route: borderline. No time for a rewrite. Summary also calculated "216 contributions", which the rules forbid |
| Consumer: faulty washing machine | **abstained** (`no_support`) | 19 s | an essential claim dropped the article's condition «للمدة التي تتناسب وطبيعتها»: a correct catch. No time for a rewrite |

\* includes a JSON repair call; split JSON replies are now merged without a model call.

All four: no crash, response structure valid, no personal data reached a model, nothing unverified presented as
verified. Questions 1-3 first asked a clarifying question (2 s).

Checker review (all 12 non-"supported" verdicts read): **8 correct**, **2 over-strict** (both on non-essential
claims, which were only removed), **1 with a wrong reason** (the essential Q3 claim, borderline), 1 plausible.

## Known limitations

1. **Abstention rate.** Writing often drops a condition or rephrases an article; the checker then marks an
   essential claim partial and, since a rewrite rarely fits in 40 s (it needs ~20-26 s), the system abstains.
   Safe, but users will see «لم نتمكن من التحقق من إجابة كاملة» for a share of questions (2 of 4 here).
2. **Summary is almost always neutral.** The personalised summary was judged to go beyond the claims in 4/4
   answers and replaced. Users see the cited claims with a neutral opening sentence.
3. **Checker strictness varies.** Occasionally over-strict on wording (non-essential claims removed) or gives a
   wrong reason. It never made an unsupported claim pass in this review.
4. **Clarifying questions are frequent** (3 of 4): one extra round trip before the answer.
5. **Speed.** 17-33 s per question; Kimi checks take 8-15 s. If the checker times out, the answer is returned as
   `not_semantically_verified` (code-checked claims only, neutral summary).
6. **PII.** Development-set results with CAMeLBERT: 83% of names fully masked, sentences leaking any value 7.9%
   (rules only: 61%, 18.4%); phones, national IDs, IBANs, emails 100%. Misses names with an attached
   preposition («لبراء»); masks a few ordinary words («ايميلي»). Not yet validated on a held-out set or measured
   on the deployment server (+720 MB memory, 0.5 s per 2,000 characters on a laptop). The privacy notice in the
   UI is required.
7. **Not implemented:** authentication, per-user rate limiting, sessions/history, feedback. The API is open.
8. **Data.** Local scraped files for 4 laws; the Data Team database integration depends on the items in
   `HANDOFF_DATA_TEAM.md` section 6. Search outside the 4 domains is keyword-only (no embeddings key).

## After the MVP (not started)

Rewrite within budget or claim-level repair instead of abstaining; summary prompt; fewer clarifying questions;
held-out PII evaluation and deployment measurement; authentication and rate limiting; database integration.
