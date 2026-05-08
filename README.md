# Personalization Agent Service

Two-stage content recommendation: BM25 narrows the catalog to a candidate set,
then a Claude agent reranks them with full per-user context and emits
auditable per-recommendation reasoning.

```
   user profile  ──►  Stage 1: BM25       ──►  k=25 candidates
   {recent_topics,                              filtered by stage + persona
    persona,
    target_stage}     Stage 2: Claude     ──►  ordered top-K
                       reranker (tool-use)     each rec carries
                                               relevance_score + per-rec reason
                                               filtered_out has reason_code from enum
```

## Eval design

For each user, two strategies are scored against the same ground truth:

1. **BM25 only** — top 5 by BM25 score within the user's filter.
2. **BM25 + rerank** — BM25 top 25 reranked by the agent to top 5.

Metrics on every user (default n=50):
- Recall@5
- nDCG@5
- Hit@5 (any-match-in-top-5)
- MRR

Plus an LLM-as-judge spot-check on a sample (default n=8): a second Claude
tier scores each returned item 0-3 against the user profile. Both strategies
are scored so the judge metric is a fair head-to-head.

The eval also aggregates `filtered_out.reason_code` counts across users, so
you can see whether the reranker is dropping items mostly for
`stage_mismatch` vs `weak_overlap` vs `duplicate_angle`.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Run

```bash
# 1) Generate the catalog (300 items + 50 user profiles, regenerable)
python -m project3_personalization_agent.data.generate_content

# 2) Eval (full 50 users; LLM judge on first 8)
python -m project3_personalization_agent.eval --n 50 --judge-n 8

# 3) API
uvicorn project3_personalization_agent.api:app --port 8001

# 4) Try a recommendation request
curl -s localhost:8001/v1/recommend \
  -H 'content-type: application/json' \
  -d '{
    "user_id": "u_demo",
    "recent_topics": ["sms-marketing","winback-campaigns","ai-personalization"],
    "persona": "lifecycle_marketer",
    "target_stage": "consideration",
    "candidate_k": 25,
    "final_k": 5
  }' | jq
```

## Static checks (no API key needed)

```bash
python verify.py
```

## Layout

```
personalization-agent-service/
├── shared/
│   ├── llm_client.py            Anthropic client (tool-use, caching, retry)
│   └── observability.py         JSONL spans
├── project3_personalization_agent/
│   ├── data/
│   │   ├── generate_content.py     300 items + 50 user profiles
│   │   ├── content_items.json
│   │   └── user_profiles.json
│   ├── retrieval.py             BM25 + reciprocal-rank-fusion hook for hybrid
│   ├── agent.py                 RerankerAgent + PersonalizationService
│   ├── api.py                   FastAPI surface
│   └── eval.py                  BM25-only vs BM25+rerank head-to-head + LLM judge
├── verify.py
├── requirements.txt
└── .gitignore
```

## Notes

- BM25 only, no dense retrieval. `Retriever.search_hybrid` accepts a callable
  for a dense retriever and fuses with reciprocal rank fusion if you want to
  add one — no other call sites need to change.
- The catalog is 300 items in-memory. For production scale, swap the index
  for Lucene / Tantivy / pgvector.
- The synthetic relevance rule (`topic ∩ recent_topics ∧ stage ∧ persona`)
  is more deterministic than real user behavior, so absolute Recall@5 numbers
  are useful as comparisons between strategies, not as predictive of
  production.
