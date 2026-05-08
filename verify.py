"""Static checks for the personalization-agent-service repo. No API calls."""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-DUMMY-FOR-IMPORT-ONLY")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(label: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((label, PASS, str(detail)))
    except Exception as e:
        results.append((label, FAIL, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


# ---- imports --------------------------------------------------------------

check("import shared", lambda: importlib.import_module("shared"))
check("import shared.llm_client", lambda: importlib.import_module("shared.llm_client"))
check("import shared.observability", lambda: importlib.import_module("shared.observability"))

check("import retrieval", lambda: importlib.import_module("project3_personalization_agent.retrieval"))
check("import agent", lambda: importlib.import_module("project3_personalization_agent.agent"))
check("import api", lambda: importlib.import_module("project3_personalization_agent.api"))
check("import eval", lambda: importlib.import_module("project3_personalization_agent.eval"))
check("import data.generate_content", lambda: importlib.import_module("project3_personalization_agent.data.generate_content"))


# ---- logic checks ---------------------------------------------------------


def check_bm25():
    from project3_personalization_agent.retrieval import Retriever, load_items

    items = load_items(ROOT / "project3_personalization_agent/data/content_items.json")
    retriever = Retriever(items)
    cands = retriever.search(
        query="sms-marketing winback-campaigns ai-personalization",
        k=15,
        target_stage="consideration",
        persona="lifecycle_marketer",
    )
    assert len(cands) > 0
    for c in cands:
        assert c.metadata["funnel_stage"] == "consideration"
        assert "lifecycle_marketer" in c.metadata["persona_fit"]
    return f"{len(cands)} candidates; all filter-conformant"


def check_rrf():
    from project3_personalization_agent.retrieval import (
        Candidate,
        reciprocal_rank_fusion,
    )

    a = [
        Candidate("x", 0.9, {}),
        Candidate("y", 0.8, {}),
        Candidate("z", 0.7, {}),
    ]
    b = [
        Candidate("y", 1.0, {}),
        Candidate("z", 0.5, {}),
        Candidate("w", 0.4, {}),
    ]
    fused = reciprocal_rank_fusion([a, b], top_k=3)
    ids = [c.item_id for c in fused]
    assert ids[0] == "y"
    return f"fused order: {ids}"


def check_metrics():
    from project3_personalization_agent.eval import (
        recall_at_k,
        ndcg_at_k,
        hit_at_k,
        reciprocal_rank,
    )
    rel = {"a", "b", "c"}
    assert recall_at_k(["a", "b", "c", "x"], rel, 3) == 1.0
    assert abs(ndcg_at_k(["a", "b", "c", "x"], rel, 3) - 1.0) < 1e-9
    assert hit_at_k(["x", "y", "a"], rel, 3) == 1
    assert hit_at_k(["x", "y", "z"], rel, 3) == 0
    assert reciprocal_rank(["x", "a", "b"], rel) == 0.5
    assert reciprocal_rank(["x", "y", "z"], rel) == 0.0
    return "recall / ndcg / hit / mrr correct"


check("BM25 retrieval + filters", check_bm25)
check("reciprocal rank fusion", check_rrf)
check("ranking metrics correctness", check_metrics)


# ---- summary --------------------------------------------------------------

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print()
print("=" * 70)
print(f"VERIFY RESULTS: {n_pass} passed, {n_fail} failed")
print("=" * 70)
for label, status, detail in results:
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{status}]  {label:55s}{suffix}")
print()
sys.exit(0 if n_fail == 0 else 1)
