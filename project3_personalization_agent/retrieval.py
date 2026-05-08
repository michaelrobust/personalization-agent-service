"""BM25 candidate retrieval. Dense-retriever-ready via reciprocal rank fusion."""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-']*")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class IndexedDoc:
    item_id: str
    text: str
    tokens: list[str]
    metadata: dict


@dataclass
class Candidate:
    item_id: str
    score: float
    metadata: dict


class BM25Index:
    """Lucene-style BM25. k1 and b are the standard tuning constants."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.docs: list[IndexedDoc] = []
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._dl: list[int] = []

    def add(self, item_id: str, text: str, metadata: dict) -> None:
        toks = tokenize(text)
        self.docs.append(IndexedDoc(item_id=item_id, text=text, tokens=toks, metadata=metadata))

    def build(self) -> None:
        n = len(self.docs)
        df: Counter[str] = Counter()
        for d in self.docs:
            df.update(set(d.tokens))
        self._idf = {
            term: math.log((n - dfi + 0.5) / (dfi + 0.5) + 1.0)
            for term, dfi in df.items()
        }
        self._dl = [len(d.tokens) for d in self.docs]
        self._avgdl = (sum(self._dl) / max(1, n)) if n else 0.0

    def search(
        self,
        query: str,
        k: int = 50,
        filter_fn=None,
    ) -> list[Candidate]:
        q_toks = tokenize(query)
        scores: list[tuple[float, int]] = []
        for i, d in enumerate(self.docs):
            if filter_fn and not filter_fn(d.metadata):
                continue
            scores.append((self._score(q_toks, i), i))
        scores.sort(reverse=True)
        out: list[Candidate] = []
        for s, i in scores[:k]:
            d = self.docs[i]
            out.append(Candidate(item_id=d.item_id, score=s, metadata=d.metadata))
        return out

    def _score(self, q_toks: list[str], doc_idx: int) -> float:
        d = self.docs[doc_idx]
        tf = Counter(d.tokens)
        dl = self._dl[doc_idx]
        score = 0.0
        for q in q_toks:
            if q not in self._idf:
                continue
            f = tf.get(q, 0)
            if f == 0:
                continue
            idf = self._idf[q]
            denom = f + self.k1 * (1 - self.b + self.b * dl / max(1.0, self._avgdl))
            score += idf * (f * (self.k1 + 1)) / denom
        return score


def reciprocal_rank_fusion(
    rankings: list[list[Candidate]],
    k_const: int = 60,
    top_k: int = 50,
) -> list[Candidate]:
    """Fuse multiple ranked candidate lists into one. RRF is rank-based, not
    score-based, so it works across heterogeneous retrievers (BM25 + dense)
    without needing score normalization."""
    accum: dict[str, float] = {}
    metadata_by_id: dict[str, dict] = {}
    for ranking in rankings:
        for rank, cand in enumerate(ranking):
            accum[cand.item_id] = accum.get(cand.item_id, 0.0) + 1.0 / (k_const + rank + 1)
            metadata_by_id[cand.item_id] = cand.metadata
    fused = sorted(accum.items(), key=lambda kv: -kv[1])
    return [
        Candidate(item_id=iid, score=s, metadata=metadata_by_id[iid])
        for iid, s in fused[:top_k]
    ]


# ---- High-level retriever wrapper ------------------------------------------


@dataclass
class ContentDoc:
    item_id: str
    title: str
    body: str
    topics: list[str]
    funnel_stage: str
    persona_fit: list[str]


def _doc_text(item: ContentDoc) -> str:
    # Repeat topic + persona tags so they get TF weight.
    topic_blob = " ".join(item.topics * 3)
    persona_blob = " ".join(item.persona_fit * 2)
    return f"{item.title}. {item.body}. {topic_blob}. {persona_blob}. stage_{item.funnel_stage}"


class Retriever:
    """BM25 retriever with hooks for future dense retrieval + RRF fusion."""

    def __init__(self, items: Sequence[ContentDoc]):
        self.items_by_id = {it.item_id: it for it in items}
        self.bm25 = BM25Index()
        for it in items:
            self.bm25.add(
                item_id=it.item_id,
                text=_doc_text(it),
                metadata={
                    "topics": it.topics,
                    "funnel_stage": it.funnel_stage,
                    "persona_fit": it.persona_fit,
                },
            )
        self.bm25.build()

    def search(
        self,
        query: str,
        k: int = 25,
        target_stage: str | None = None,
        persona: str | None = None,
    ) -> list[Candidate]:
        def filter_fn(meta: dict) -> bool:
            if target_stage and meta["funnel_stage"] != target_stage:
                return False
            if persona and persona not in meta["persona_fit"]:
                return False
            return True

        return self.bm25.search(query, k=k, filter_fn=filter_fn)

    def search_hybrid(
        self,
        query: str,
        dense_search_fn,
        k: int = 25,
        **filters,
    ) -> list[Candidate]:
        """Return RRF fusion of BM25 + a dense retriever. Pass a callable
        that takes (query, k) and returns a list[Candidate]."""
        bm25_ranked = self.search(query=query, k=k * 2, **filters)
        dense_ranked = dense_search_fn(query, k * 2)
        return reciprocal_rank_fusion([bm25_ranked, dense_ranked], top_k=k)


def load_items(path: Path) -> list[ContentDoc]:
    raw = json.loads(path.read_text())
    return [ContentDoc(**r) for r in raw]


def load_users(path: Path) -> list[dict]:
    return json.loads(path.read_text())
