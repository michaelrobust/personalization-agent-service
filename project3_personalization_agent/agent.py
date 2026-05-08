"""Stage-2 reranker. Takes BM25 candidates + user profile, emits ordered top-K."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from shared import EventLog, LLMClient

from .retrieval import Candidate, ContentDoc


RERANK_TOOL = {
    "name": "emit_personalized_recommendations",
    "description": "Final ordered top-K recommendations. Index 0 is the strongest.",
    "input_schema": {
        "type": "object",
        "properties": {
            "recommendations": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "relevance_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "why_this_user": {"type": "string"},
                    },
                    "required": ["item_id", "relevance_score", "why_this_user"],
                },
            },
            "filtered_out": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "reason_code": {
                            "type": "string",
                            "enum": [
                                "topic_mismatch",
                                "stage_mismatch",
                                "persona_mismatch",
                                "weak_overlap",
                                "duplicate_angle",
                                "other",
                            ],
                        },
                        "reason": {"type": "string"},
                    },
                    "required": ["item_id", "reason_code", "reason"],
                },
            },
        },
        "required": ["recommendations", "filtered_out"],
    },
}


SYSTEM = """You are a content personalization reranker. You receive:
  - A user's profile: recent_topics, persona, target_stage.
  - A candidate set of content items pre-filtered by a BM25 stage.

Re-rank so the most relevant items come first. Drop items that are stage- \
correct but topically shallow. Use reason_code from the schema enum so \
filtered_out is analyzable downstream.

Calibrate relevance_score:
  >= 0.85  multi-topic + stage + persona all align tightly
  0.6-0.85 strong topical hit but only one of stage/persona
  < 0.6    weak match that probably should have been filtered

Always call emit_personalized_recommendations exactly once."""


@dataclass
class Recommendation:
    item_id: str
    relevance_score: float
    why_this_user: str


@dataclass
class RerankResult:
    recommendations: list[Recommendation]
    filtered_out: list[dict[str, str]]


class RerankerAgent:
    def __init__(self, client: LLMClient, log: EventLog | None = None):
        self.client = client
        self.log = log or EventLog("logs/reranker.jsonl")

    def run(
        self,
        user_profile: dict[str, Any],
        candidates: list[Candidate],
        items_by_id: dict[str, ContentDoc],
        top_k: int = 5,
    ) -> RerankResult:
        with self.log.span(
            "reranker",
            user_id=user_profile.get("user_id"),
            n_candidates=len(candidates),
        ) as ctx:
            cand_blob = []
            for c in candidates:
                it = items_by_id[c.item_id]
                cand_blob.append(
                    {
                        "item_id": it.item_id,
                        "title": it.title,
                        "topics": it.topics,
                        "funnel_stage": it.funnel_stage,
                        "persona_fit": it.persona_fit,
                        "bm25_score": round(c.score, 3),
                    }
                )
            user_msg = (
                "User profile:\n"
                f"  user_id        : {user_profile.get('user_id')}\n"
                f"  recent_topics  : {user_profile['recent_topics']}\n"
                f"  persona        : {user_profile['persona']}\n"
                f"  target_stage   : {user_profile['target_stage']}\n\n"
                f"Candidate set ({len(cand_blob)} items, BM25-prefiltered):\n"
                f"{cand_blob}\n\n"
                f"Return the top {top_k} as ordered recommendations. "
                "Drop weak matches into filtered_out with a reason_code."
            )
            resp = self.client.call(
                system=SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
                tools=[RERANK_TOOL],
                tool_choice={"type": "tool", "name": RERANK_TOOL["name"]},
                cache_system=True,
                temperature=0.2,
                max_tokens=2048,
            )
            data = resp.first_tool_input()
            if not data:
                raise RuntimeError("Reranker returned no tool call.")
            recs = [Recommendation(**r) for r in data["recommendations"][:top_k]]
            ctx["n_returned"] = len(recs)
            return RerankResult(
                recommendations=recs,
                filtered_out=list(data.get("filtered_out", [])),
            )


@dataclass
class PersonalizationOutput:
    user_id: str
    bm25_top_ids: list[str]
    recommendations: list[Recommendation]
    filtered_out: list[dict[str, str]]


class PersonalizationService:
    def __init__(
        self,
        retriever,
        client: LLMClient | None = None,
        log_path: str = "logs/personalization.jsonl",
    ):
        self.retriever = retriever
        self.client = client or LLMClient()
        self.log = EventLog(log_path)
        self.reranker = RerankerAgent(self.client, self.log)

    def recommend(
        self,
        user_profile: dict[str, Any],
        candidate_k: int = 25,
        final_k: int = 5,
    ) -> PersonalizationOutput:
        with self.log.span("personalize", user_id=user_profile.get("user_id")):
            query = " ".join(user_profile["recent_topics"])
            candidates = self.retriever.search(
                query=query,
                k=candidate_k,
                target_stage=user_profile.get("target_stage"),
                persona=user_profile.get("persona"),
            )
            rerank = self.reranker.run(
                user_profile=user_profile,
                candidates=candidates,
                items_by_id=self.retriever.items_by_id,
                top_k=final_k,
            )
            return PersonalizationOutput(
                user_id=user_profile["user_id"],
                bm25_top_ids=[c.item_id for c in candidates[:final_k]],
                recommendations=rerank.recommendations,
                filtered_out=rerank.filtered_out,
            )
