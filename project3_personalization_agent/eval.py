"""BM25-only vs BM25+rerank head-to-head, plus LLM-as-judge spot-check."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from shared import LLMClient

from .agent import PersonalizationService
from .retrieval import Retriever, load_items, load_users


_DATA_DIR = Path(__file__).parent / "data"
_LOG_DIR = Path(__file__).parent / "logs"


# ---- Ranking metrics --------------------------------------------------------


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = predicted_ids[:k]
    hits = sum(1 for x in top if x in relevant_ids)
    return hits / min(len(relevant_ids), k)


def ndcg_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    dcg = 0.0
    for i, item in enumerate(predicted_ids[:k]):
        if item in relevant_ids:
            dcg += 1.0 / math.log2(i + 2)
    n_rel = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(n_rel))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> int:
    return 1 if any(x in relevant_ids for x in predicted_ids[:k]) else 0


def reciprocal_rank(predicted_ids: list[str], relevant_ids: set[str]) -> float:
    for i, item in enumerate(predicted_ids):
        if item in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ---- LLM judge --------------------------------------------------------------


JUDGE_TOOL = {
    "name": "emit_recommendation_quality",
    "description": "Score each recommended item 0-3 against the user profile.",
    "input_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "item_id": {"type": "string"},
                        "relevance": {"type": "integer", "minimum": 0, "maximum": 3},
                        "notes": {"type": "string"},
                    },
                    "required": ["item_id", "relevance", "notes"],
                },
            }
        },
        "required": ["scores"],
    },
}


JUDGE_SYSTEM = """You are a strict QA reviewer scoring how well each \
recommended content item fits the given user profile. Score on a 0-3 scale:

  0 = irrelevant
  1 = weakly related
  2 = relevant, decent fit
  3 = excellent, would-click

Score every item supplied. Always call emit_recommendation_quality once."""


def judge_recommendations(
    judge: LLMClient,
    user_profile: dict[str, Any],
    items_by_id: dict[str, Any],
    item_ids: list[str],
) -> list[dict[str, Any]]:
    items_blob = []
    for iid in item_ids:
        it = items_by_id[iid]
        items_blob.append(
            {
                "item_id": it.item_id,
                "title": it.title,
                "topics": it.topics,
                "funnel_stage": it.funnel_stage,
                "persona_fit": it.persona_fit,
            }
        )
    user_msg = (
        f"User profile:\n"
        f"  recent_topics : {user_profile['recent_topics']}\n"
        f"  persona       : {user_profile['persona']}\n"
        f"  target_stage  : {user_profile['target_stage']}\n\n"
        f"Items to score:\n{items_blob}"
    )
    resp = judge.call(
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
        tools=[JUDGE_TOOL],
        tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
        cache_system=True,
        temperature=0.1,
    )
    data = resp.first_tool_input()
    if not data:
        return []
    return list(data["scores"])


# ---- Eval driver ------------------------------------------------------------


@dataclass
class StrategyMetrics:
    name: str
    recall_at_5: float
    ndcg_at_5: float
    hit_at_5: float
    mrr: float
    avg_judge_score: float | None = None


@dataclass
class UserEvalRow:
    user_id: str
    bm25_top_ids: list[str]
    agent_top_ids: list[str]
    bm25: StrategyMetrics
    agent: StrategyMetrics
    filtered_out_reason_codes: dict[str, int]


def _strategy_metrics(
    name: str, ids: list[str], relevant: set[str]
) -> StrategyMetrics:
    return StrategyMetrics(
        name=name,
        recall_at_5=recall_at_k(ids, relevant, 5),
        ndcg_at_5=ndcg_at_k(ids, relevant, 5),
        hit_at_5=hit_at_k(ids, relevant, 5),
        mrr=reciprocal_rank(ids, relevant),
    )


def _avg(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def run_eval(
    n_users: int = 50,
    judge_n_users: int = 8,
    gen_model: str = "claude-sonnet-4-6",
    judge_model: str = "claude-haiku-4-5-20251001",
) -> dict[str, Any]:
    items = load_items(_DATA_DIR / "content_items.json")
    users = load_users(_DATA_DIR / "user_profiles.json")[:n_users]
    items_by_id = {it.item_id: it for it in items}

    retriever = Retriever(items)
    service = PersonalizationService(
        retriever=retriever,
        client=LLMClient(model=gen_model),
        log_path=str(_LOG_DIR / "eval.jsonl"),
    )
    judge = LLMClient(model=judge_model)

    rows: list[UserEvalRow] = []
    bm25_judge_scores: list[float] = []
    agent_judge_scores: list[float] = []

    for idx, u in enumerate(users):
        print(f"  [{idx+1}/{len(users)}] user={u['user_id']}", flush=True)
        relevant = set(u["relevant_item_ids"])

        out = service.recommend(user_profile=u, candidate_k=25, final_k=5)
        bm25_ids = out.bm25_top_ids
        agent_ids = [r.item_id for r in out.recommendations]

        bm25_metrics = _strategy_metrics("bm25", bm25_ids, relevant)
        agent_metrics = _strategy_metrics("agent", agent_ids, relevant)

        if idx < judge_n_users:
            bm25_scored = judge_recommendations(judge, u, items_by_id, bm25_ids)
            agent_scored = judge_recommendations(judge, u, items_by_id, agent_ids)
            if bm25_scored:
                avg = sum(s["relevance"] for s in bm25_scored) / len(bm25_scored)
                bm25_metrics.avg_judge_score = avg
                bm25_judge_scores.append(avg)
            if agent_scored:
                avg = sum(s["relevance"] for s in agent_scored) / len(agent_scored)
                agent_metrics.avg_judge_score = avg
                agent_judge_scores.append(avg)

        reason_counts = Counter(f.get("reason_code", "other") for f in out.filtered_out)
        rows.append(
            UserEvalRow(
                user_id=u["user_id"],
                bm25_top_ids=bm25_ids,
                agent_top_ids=agent_ids,
                bm25=bm25_metrics,
                agent=agent_metrics,
                filtered_out_reason_codes=dict(reason_counts),
            )
        )

    bm25_recall = _avg([r.bm25.recall_at_5 for r in rows])
    agent_recall = _avg([r.agent.recall_at_5 for r in rows])
    bm25_ndcg = _avg([r.bm25.ndcg_at_5 for r in rows])
    agent_ndcg = _avg([r.agent.ndcg_at_5 for r in rows])
    bm25_hit = _avg([r.bm25.hit_at_5 for r in rows])
    agent_hit = _avg([r.agent.hit_at_5 for r in rows])
    bm25_mrr = _avg([r.bm25.mrr for r in rows])
    agent_mrr = _avg([r.agent.mrr for r in rows])

    # Aggregate filtered_out reason codes across all users.
    reason_totals: Counter[str] = Counter()
    for r in rows:
        reason_totals.update(r.filtered_out_reason_codes)

    summary = {
        "n_users": len(rows),
        "judge_n_users": min(judge_n_users, len(rows)),
        "strategies": {
            "bm25_only": {
                "recall_at_5": round(bm25_recall, 4),
                "ndcg_at_5": round(bm25_ndcg, 4),
                "hit_at_5": round(bm25_hit, 4),
                "mrr": round(bm25_mrr, 4),
                "avg_judge_score": round(_avg(bm25_judge_scores), 3) if bm25_judge_scores else None,
            },
            "bm25_plus_rerank": {
                "recall_at_5": round(agent_recall, 4),
                "ndcg_at_5": round(agent_ndcg, 4),
                "hit_at_5": round(agent_hit, 4),
                "mrr": round(agent_mrr, 4),
                "avg_judge_score": round(_avg(agent_judge_scores), 3) if agent_judge_scores else None,
            },
        },
        "deltas": {
            "recall_at_5": round(agent_recall - bm25_recall, 4),
            "ndcg_at_5": round(agent_ndcg - bm25_ndcg, 4),
            "hit_at_5": round(agent_hit - bm25_hit, 4),
            "mrr": round(agent_mrr - bm25_mrr, 4),
        },
        "filtered_out_reason_totals": dict(reason_totals),
        "per_user": [_row_to_dict(r) for r in rows],
    }
    return summary


def _row_to_dict(r: UserEvalRow) -> dict[str, Any]:
    return {
        "user_id": r.user_id,
        "bm25_top_ids": r.bm25_top_ids,
        "agent_top_ids": r.agent_top_ids,
        "bm25": asdict(r.bm25),
        "agent": asdict(r.agent),
        "filtered_out_reason_codes": r.filtered_out_reason_codes,
    }


def format_summary(summary: dict[str, Any]) -> str:
    s = summary["strategies"]
    d = summary["deltas"]
    lines = [
        f"n_users         : {summary['n_users']}",
        f"judge_n_users   : {summary['judge_n_users']}",
        "",
        f"{'metric':16s} {'BM25 only':>11s}  {'BM25+rerank':>12s}  {'delta':>8s}",
    ]
    for m in ["recall_at_5", "ndcg_at_5", "hit_at_5", "mrr"]:
        lines.append(
            f"{m:16s} {s['bm25_only'][m]:>11.4f}  "
            f"{s['bm25_plus_rerank'][m]:>12.4f}  {d[m]:>+8.4f}"
        )
    if s["bm25_only"]["avg_judge_score"] is not None:
        lines.append(
            f"{'avg_judge':16s} {s['bm25_only']['avg_judge_score']:>11.3f}  "
            f"{s['bm25_plus_rerank']['avg_judge_score']:>12.3f}  "
            f"{(s['bm25_plus_rerank']['avg_judge_score'] - s['bm25_only']['avg_judge_score']):>+8.3f}"
        )
    lines.append("")
    lines.append("filtered_out reason codes (total across users):")
    for code, n in sorted(
        summary["filtered_out_reason_totals"].items(), key=lambda kv: -kv[1]
    ):
        lines.append(f"  {code:18s} {n}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--judge-n", type=int, default=8)
    parser.add_argument("--gen-model", default="claude-sonnet-4-6")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001")
    args = parser.parse_args()

    _LOG_DIR.mkdir(exist_ok=True)
    summary = run_eval(
        n_users=args.n,
        judge_n_users=args.judge_n,
        gen_model=args.gen_model,
        judge_model=args.judge_model,
    )
    out_path = _LOG_DIR / "eval_summary.json"
    out_path.write_text(json.dumps(summary, indent=2))
    text_path = _LOG_DIR / "eval_summary.txt"
    text_path.write_text(format_summary(summary))
    print(format_summary(summary))
    print(f"\nFull report -> {out_path}")


if __name__ == "__main__":
    main()
