"""Every Supabase table and column name the pipeline uses, in one place.

The Data team owns the schema. When they finalise it, align the names here
and nothing else has to change. Values below follow the blueprint (section 4);
the ones marked ASSUMED are our guesses and must be confirmed.
"""

LEGISLATION = {
    "table": "legislation",
    "id": "id",
    "title": "title",
    "type": "type",                     # law | regulation | instructions
    "number": "number",
    "year": "year",
    "status": "status",                 # in_force | repealed  (ASSUMED values)
    "last_amendment": "last_amendment",
    "gazette": "gazette",               # ASSUMED single column (could be several)
    "domain": "domain",
    "coverage_level": "coverage_level",
    "source_url": "source_url",
}

ARTICLES = {
    "table": "articles",
    "id": "id",
    "legislation_id": "legislation_id",
    "article_no": "article_number",     # ASSUMED name
    "text": "original_text",            # ASSUMED name
    "repealed": "repealed",             # ASSUMED name
}

DOMAIN_PACKS = {
    "table": "domain_packs",
    "domain": "domain",
    "article_id": "article_id",
    "position": "position",             # ASSUMED name for "in order"
}

GLOSSARY = {
    "table": "glossary",
    "phrase": "dialect_phrase",         # ASSUMED name
    "legal_term": "legal_term",
    "domain": "domain",
}

RELATIONS = {
    "table": "relations",
    "type": "relation_type",            # ASSUMED: 'definitions' | 'parent_law'
    "legislation_id": "legislation_id", # ASSUMED: the law the relation belongs to
    "target_article_id": "target_article_id",          # ASSUMED: definitions article
    "target_legislation_id": "target_legislation_id",  # ASSUMED: parent law
}

DATA_VERSIONS = {
    "table": "data_versions",
    "version": "version",
    "scraped_at": "scrape_date",        # ASSUMED name
    "embedding_model": "embedding_model",
    "created_at": "created_at",         # ASSUMED, used for ordering
}

TRACES = {"table": "traces"}            # we write the row as JSON; see sql/traces_table.sql

HYBRID_SEARCH = {
    "function": "hybrid_search",
    # parameter names, see sql/hybrid_search_contract.sql
    "query_text": "query_text",
    "query_embedding": "query_embedding",
    "match_count": "match_count",
    "full_text_weight": "full_text_weight",
    "semantic_weight": "semantic_weight",
    "rrf_k": "rrf_k",
    "filter_domain": "filter_domain",
    # result columns
    "out_article_id": "article_id",
    "out_chunk_id": "chunk_id",
    "out_score": "score",
}
