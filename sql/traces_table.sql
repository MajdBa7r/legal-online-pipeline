-- traces: one row per request, written by the backend after the response is sent.
-- PROPOSAL for the Data team (they own the schema). Never holds question text, answers or personal data,
-- except eval_* columns, which are filled only for test-set requests carrying the secret X-Eval-Token header.

create table if not exists traces (
  id              bigint generated always as identity primary key,
  created_at      timestamptz not null default now(),
  request_id      text not null,
  response_type   text,            -- answer | clarification | referral | no_support | out_of_scope | error
  outcome         text,            -- decision of stage 7, or gate result
  domain          text,
  model_domain    text,
  gate            text,
  risk            text,
  path            text,            -- pack | pack_search | search
  generations     int,
  partial         boolean,
  semantic_skipped boolean,
  understand_fallback boolean,
  embed_fallback  boolean,
  rerank_fallback boolean,
  clarification_round boolean,
  pii_count       int,             -- how many values were masked, never the values
  total_ms        int,
  timings_ms      jsonb,
  model_calls     jsonb,           -- role, model, latency, status, tokens
  tokens_in       int,
  tokens_out      int,
  verification    jsonb,           -- counts by check
  retrieved_ids   jsonb,
  supplied_ids    jsonb,
  expanded_ids    jsonb,
  cited_ids       jsonb,
  error           text,
  eval_question   text,
  eval_response   jsonb
);

alter table traces enable row level security;  -- only the service key (backend) writes or reads
