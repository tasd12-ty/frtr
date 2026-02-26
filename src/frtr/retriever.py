"""Stage 2: Hybrid lexical-dense retrieval with Reciprocal Rank Fusion (RRF).

Implements Algorithm 1, lines 12-19:
  q ← E_text(q)
  R_v ← top-K_v by cos(q, v_d)           (Dense search)
  R_s ← top-K_s by BM25(q, text_d)       (Lexical search)
  for d in R_v ∪ R_s:
      RRF(d) ← Σ_{r∈{v,s}} 1/(k + rank_r(d))
  C ← top-K by RRF score with provenance labels
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import FRTRConfig
from .embeddings.base import EmbeddingBackend
from .vectordb import HybridVectorStore


@dataclass
class RetrievedChunk:
    """A chunk returned by the retrieval stage."""

    chunk_id: str
    text: str
    score: float
    source: str  # "dense", "lexical", or "fused"
    unit_type: str
    sheet_name: str
    workbook_name: str
    metadata: dict


def reciprocal_rank_fusion(
    dense_results: list[tuple[str, float, str, dict]],
    lexical_results: list[tuple[str, float, str, dict]],
    k: int = 60,
    top_k: int = 10,
) -> list[tuple[str, float, str, dict, str]]:
    """Reciprocal Rank Fusion (RRF) as described in Section 3.3.

    RRF(d) = Σ_{r∈R} 1/(k + rank_r(d))

    where k=60 stabilizes low-ranked items, and R is the set of
    retrieval methods {dense, lexical}.

    Args:
        dense_results:   (chunk_id, score, text, metadata) from dense search
        lexical_results: (chunk_id, score, text, metadata) from BM25
        k:               RRF smoothing constant (default 60)
        top_k:           Number of final results to return

    Returns:
        List of (chunk_id, rrf_score, text, metadata, source_label)
    """
    rrf_scores: dict[str, float] = {}
    chunk_info: dict[str, tuple[str, dict]] = {}
    sources: dict[str, list[str]] = {}

    # Score from dense retrieval
    for rank, (cid, score, text, meta) in enumerate(dense_results, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        chunk_info[cid] = (text, meta)
        sources.setdefault(cid, []).append("dense")

    # Score from lexical retrieval
    for rank, (cid, score, text, meta) in enumerate(lexical_results, start=1):
        rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in chunk_info:
            chunk_info[cid] = (text, meta)
        sources.setdefault(cid, []).append("lexical")

    # Sort by RRF score descending
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for cid, rrf_score in sorted_items[:top_k]:
        text, meta = chunk_info[cid]
        source_label = "+".join(sources[cid])
        results.append((cid, rrf_score, text, meta, source_label))

    return results


class HybridRetriever:
    """Hybrid retrieval engine combining dense and lexical search with RRF.

    This implements Stage 2 of the FRTR pipeline.
    """

    def __init__(
        self,
        store: HybridVectorStore,
        embedder: EmbeddingBackend,
        config: FRTRConfig,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._config = config

    def retrieve(
        self,
        query: str,
        workbook_filter: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query using hybrid search + RRF.

        Args:
            query:           Natural language question
            workbook_filter: If set, only retrieve from this workbook

        Returns:
            Top-K fused chunks with provenance metadata
        """
        # Embed the query (text modality)
        query_embedding = self._embedder.embed_text(query)

        # Dense retrieval: top-K_v by cosine similarity
        dense_results = self._store.query_dense(
            query_embedding, top_k=self._config.k_v
        )

        # Lexical retrieval: top-K_s by BM25
        lexical_results = self._store.query_bm25(
            query, top_k=self._config.k_s
        )

        # Optional: filter by workbook
        if workbook_filter:
            dense_results = [
                r for r in dense_results
                if r[3].get("workbook_name") == workbook_filter
            ]
            lexical_results = [
                r for r in lexical_results
                if r[3].get("workbook_name") == workbook_filter
            ]

        # RRF fusion
        fused = reciprocal_rank_fusion(
            dense_results,
            lexical_results,
            k=self._config.rrf_k,
            top_k=self._config.k_final,
        )

        # Convert to RetrievedChunk objects
        chunks = []
        for cid, score, text, meta, source in fused:
            chunks.append(
                RetrievedChunk(
                    chunk_id=cid,
                    text=text,
                    score=score,
                    source=source,
                    unit_type=meta.get("unit_type", "unknown"),
                    sheet_name=meta.get("sheet_name", ""),
                    workbook_name=meta.get("workbook_name", ""),
                    metadata=meta,
                )
            )

        return chunks
