-- What the online pipeline expects from the Data team's hybrid_search function.
-- PROPOSAL to agree in the first two hours; the Data team owns the final version.
-- Based on Supabase's hybrid search pattern (full-text + vector, merged with Reciprocal Rank Fusion),
-- with three additions the pipeline relies on:
--   1. filter_domain     -> search inside one domain when a domain pack is too large to send whole
--   2. query_embedding may be NULL -> full-text-only mode when the embedding call fails
--   3. only in-force articles of in-force legislation are returned
--
-- Notes for the Data team:
--   * The app sends query_text already normalized and OR-joined ("word1 or word2 or ..."),
--     because websearch_to_tsquery ANDs plain words and long dialect questions then match nothing.
--   * The text search config here ('arabic') must be the same one used in the chunks.fts generated column.
--     If you fall back to 'simple', change both.
--   * <=> is cosine distance; the HNSW index must use vector_cosine_ops to be used.
--     (OpenAI embeddings are normalized, so inner product <#> with vector_ip_ops also works; pick one, use it in both places.)
--   * Column names follow the blueprint; align app/db/schema_map.py with whatever you finalise.

create or replace function hybrid_search(
  query_text text,
  query_embedding vector(1024) default null,
  match_count int default 30,
  full_text_weight float default 1,
  semantic_weight float default 1,
  rrf_k int default 50,
  filter_domain text default null
)
returns table (article_id bigint, chunk_id bigint, score float)
language sql stable
as $$
with eligible as (
  select c.id, c.article_id, c.fts, c.embedding
  from chunks c
  join articles a     on a.id = c.article_id
  join legislation l  on l.id = a.legislation_id
  where not a.repealed
    and l.status = 'in_force'
    and (filter_domain is null or l.domain = filter_domain)
),
full_text as (
  select e.id,
         row_number() over (order by ts_rank_cd(e.fts, websearch_to_tsquery('arabic', query_text)) desc) as rank_ix
  from eligible e
  where e.fts @@ websearch_to_tsquery('arabic', query_text)
  order by rank_ix
  limit least(match_count, 50) * 2
),
semantic as (
  select e.id,
         row_number() over (order by e.embedding <=> query_embedding) as rank_ix
  from eligible e
  where query_embedding is not null
  order by rank_ix
  limit least(match_count, 50) * 2
)
select c.article_id,
       c.id as chunk_id,
       coalesce(1.0 / (rrf_k + ft.rank_ix), 0.0) * full_text_weight
     + coalesce(1.0 / (rrf_k + s.rank_ix), 0.0) * semantic_weight as score
from full_text ft
full outer join semantic s on ft.id = s.id
join chunks c on c.id = coalesce(ft.id, s.id)
order by score desc
limit least(match_count, 50);
$$;

-- If the eligible-CTE join keeps the planner from using the HNSW index on a large corpus,
-- move the filters after the vector ORDER BY (fetch more, e.g. match_count * 4, then filter).
-- Measure with EXPLAIN ANALYZE on Day 2.
