"""Hybrid retrieval: dense + BM25, fused by RRF, then diversified with MMR.

Why hybrid: dense embeddings are good at paraphrase ("time off" -> "annual
leave") and bad at rare literal tokens (an invoice number, "FR-4"). BM25 is the
reverse. Reciprocal rank fusion combines the two rankings without needing their
scores to be on the same scale.
"""

from __future__ import annotations

import numpy as np

from .config import settings
from .store import Hit, Store, embed

RRF_K = 60  # standard damping constant; larger = flatter weighting of top ranks


def reciprocal_rank_fusion(
    rankings: list[list[int]], weights: list[float] | None = None
) -> dict[int, float]:
    weights = weights or [1.0] * len(rankings)
    fused: dict[int, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, idx in enumerate(ranking):
            fused[idx] = fused.get(idx, 0.0) + weight / (RRF_K + rank + 1)
    return fused


def mmr(
    relevance: np.ndarray,
    candidate_vecs: np.ndarray,
    candidate_ids: list[int],
    k: int,
    lambda_: float,
) -> list[int]:
    """Maximal Marginal Relevance: trade relevance against redundancy.

    Without this, four near-identical chunks from the same paragraph crowd out
    the one chunk from another document that actually completes the answer.

    `relevance` must already be scaled to roughly [0, 1] and is passed in rather
    than recomputed from the query: in hybrid mode it carries the fused
    dense+BM25 ranking. Deriving it from cosine here would silently discard the
    BM25 half of the search.
    """
    if len(candidate_ids) <= k:
        return candidate_ids

    relevance = np.asarray(relevance, dtype="float32").ravel()
    similarity = candidate_vecs @ candidate_vecs.T

    selected = [int(np.argmax(relevance))]
    while len(selected) < k:
        best, best_score = None, -np.inf
        for i in range(len(candidate_ids)):
            if i in selected:
                continue
            redundancy = max(similarity[i][j] for j in selected)
            score = lambda_ * relevance[i] - (1 - lambda_) * redundancy
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        selected.append(best)

    return [candidate_ids[i] for i in selected]


def search(store: Store, query: str, k: int | None = None) -> list[Hit]:
    """Return the k best chunks for a query, most relevant first."""
    k = k or settings.top_k
    pool = max(settings.candidate_k, k * 4)

    dense = store.dense(query, pool)
    dense_ids = [i for i, _ in dense]

    if settings.use_hybrid:
        sparse_ids = [i for i, _ in store.sparse(query, pool)]
        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], [1.0, 0.7])
    else:
        sparse_ids = []
        fused = reciprocal_rank_fusion([dense_ids])

    if not fused:
        return []

    ordered = sorted(fused, key=lambda i: fused[i], reverse=True)[:pool]

    # Diversify among the fused candidates, then restore fused order. Relevance
    # for MMR is the fused score scaled to [0, 1] so the BM25 half survives.
    top = fused[ordered[0]]
    relevance = np.array([fused[i] / top for i in ordered], dtype="float32")
    query_vec = embed([query], store.model_name)
    chosen = mmr(relevance, store.vectors[ordered], ordered, k, settings.mmr_lambda)
    chosen.sort(key=lambda i: fused[i], reverse=True)

    dense_rank = {idx: r for r, idx in enumerate(dense_ids)}
    sparse_rank = {idx: r for r, idx in enumerate(sparse_ids)}
    # Cosine of every chosen chunk against the query, including chunks that only
    # BM25 surfaced - those have no dense score of their own.
    similarity = (store.vectors[chosen] @ query_vec.ravel()) if chosen else []

    return [
        Hit(
            chunk=store.chunks[idx],
            score=fused[idx],
            similarity=float(sim),
            dense_rank=dense_rank.get(idx),
            sparse_rank=sparse_rank.get(idx),
        )
        for idx, sim in zip(chosen, similarity)
    ]


def is_confident(hits: list[Hit]) -> bool:
    """Whether retrieval looks good enough to attempt an answer.

    Gated on raw cosine similarity, not the fused score: RRF is rank-based, so
    its top score is identical for a perfect match and for nonsense.
    """
    return bool(hits) and max(h.similarity for h in hits) >= settings.min_similarity


def build_context(hits: list[Hit], max_chars: int | None = None) -> tuple[str, list[Hit]]:
    """Format hits as a numbered context block the model can cite by index."""
    max_chars = max_chars or settings.max_context_chars
    parts, used, total = [], [], 0
    for hit in hits:
        block = f"[{len(used) + 1}] ({hit.chunk.label})\n{hit.chunk.text}"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(hit)
        total += len(block)
    return "\n\n".join(parts), used
