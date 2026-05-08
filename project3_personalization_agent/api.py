"""FastAPI surface for the personalization service."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from shared import LLMClient

from .agent import PersonalizationService
from .retrieval import Retriever, load_items


_DATA_DIR = Path(__file__).parent / "data"

app = FastAPI(title="Personalization Agent", version="0.1.0")

# Index loads at import time. The LLM-backed service is built on first use.
_items = load_items(_DATA_DIR / "content_items.json")
_retriever = Retriever(_items)
_service: PersonalizationService | None = None


def _get_service() -> PersonalizationService:
    global _service
    if _service is None:
        _service = PersonalizationService(retriever=_retriever, client=LLMClient())
    return _service


class RecommendRequest(BaseModel):
    user_id: str
    recent_topics: list[str]
    persona: str
    target_stage: str
    candidate_k: int = 25
    final_k: int = 5


@app.get("/v1/health")
def health() -> dict[str, Any]:
    return {"ok": True, "items_indexed": len(_items)}


@app.post("/v1/recommend")
def recommend(req: RecommendRequest) -> dict[str, Any]:
    try:
        out = _get_service().recommend(
            user_profile=req.dict(),
            candidate_k=req.candidate_k,
            final_k=req.final_k,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "user_id": out.user_id,
        "bm25_top_ids": out.bm25_top_ids,
        "recommendations": [
            {
                "item_id": r.item_id,
                "relevance_score": r.relevance_score,
                "why_this_user": r.why_this_user,
            }
            for r in out.recommendations
        ],
        "filtered_out": out.filtered_out,
    }
