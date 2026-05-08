# Personalization Agent

Two-stage content recommendation:

```
   user profile (recent_topics, persona, target_stage)
        │
        ▼
   BM25 candidate retrieval     k=25, filtered by stage + persona
        │
        ▼
   Claude reranker agent         tool-use → ordered top-K
                                 each rec carries relevance_score + per-rec reason
                                 dropped items have a reason_code from a fixed enum
```

## Files

```
data/
  generate_content.py      300 synthetic content items + 50 user profiles
  content_items.json
  user_profiles.json
retrieval.py               BM25 + reciprocal-rank-fusion hook for hybrid retrieval
agent.py                   RerankerAgent + PersonalizationService
api.py                     FastAPI surface
eval.py                    BM25-only vs BM25+rerank head-to-head + LLM judge
```

## Eval design

For each user, two strategies are scored against the same ground truth:

1. **BM25 only** — top 5 by BM25 score within the user's filter.
2. **BM25 + rerank** — BM25 top 25 reranked to top 5 by the agent.

Metrics on every user (n=50):
- Recall@5
- nDCG@5
- Hit@5 (any-match-in-top-5)
- MRR

Plus an LLM-as-judge spot-check on a sample (default n=8): a second Claude
tier scores each returned item 0-3 against the user profile. Both strategies
are scored so the judge metric is a fair comparison.

The eval also aggregates `filtered_out.reason_code` counts across users — a
cheap way to see whether the reranker is dropping items mostly for
`stage_mismatch` vs `weak_overlap` vs `duplicate_angle`. Useful for spotting
systematic biases in the candidate set.

## Run

```bash
pip install anthropic fastapi uvicorn pydantic
export ANTHROPIC_API_KEY=sk-ant-...

# generate the catalog (one-off)
python -m project3_personalization_agent.data.generate_content

# eval (full 50 users; LLM judge on first 8)
python -m project3_personalization_agent.eval --n 50 --judge-n 8

# API
uvicorn project3_personalization_agent.api:app --port 8001

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

## Notes

- BM25 only, no dense retrieval. `Retriever.search_hybrid` accepts a callable
  for a dense retriever and fuses with reciprocal rank fusion if you want to
  add one — no other call sites need to change.
- The catalog is 300 items in-memory. For production scale, swap the index
  for Lucene / Tantivy / pgvector.
- The synthetic relevance rule (`topic ∩ recent_topics ∧ stage ∧ persona`) is
  more deterministic than real user behavior, so the absolute Recall@5
  numbers are useful as comparisons between strategies, not as predictive of
  production.
