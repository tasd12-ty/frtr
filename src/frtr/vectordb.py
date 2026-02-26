"""Hybrid vector store: ChromaDB (dense) + rank-bm25 (lexical).

ChromaDB stores the dense embeddings and metadata. BM25 is maintained
as an in-memory index over the text fields for lexical retrieval.
Together they enable the hybrid retrieval described in Algorithm 1.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Optional

import chromadb
import numpy as np
from rank_bm25 import BM25Okapi

from .indexer import ChunkUnit


def _chunk_id(chunk: ChunkUnit, idx: int) -> str:
    """Generate a stable unique ID for a chunk."""
    raw = f"{chunk.workbook_name}:{chunk.sheet_name}:{chunk.unit_type}:{idx}"
    return hashlib.md5(raw.encode()).hexdigest()


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for BM25."""
    import re
    return re.findall(r"\w+", text.lower())


class HybridVectorStore:
    """Combined dense (ChromaDB) + sparse (BM25) vector store.

    Supports:
    - Adding chunks with embeddings, text, and metadata
    - Dense retrieval via cosine similarity (ChromaDB)
    - Lexical retrieval via BM25 scoring
    - Persistence to disk
    """

    def __init__(self, persist_dir: Optional[Path] = None) -> None:
        if persist_dir:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(
                path=str(persist_dir / "chroma")
            )
        else:
            self._chroma_client = chromadb.EphemeralClient()

        self._collection = self._chroma_client.get_or_create_collection(
            name="frtr_chunks",
            metadata={"hnsw:space": "cosine"},
        )
        self._persist_dir = persist_dir

        # BM25 index (in-memory)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_corpus: list[list[str]] = []
        self._bm25_ids: list[str] = []
        self._bm25_texts: list[str] = []

        # Chunk metadata lookup
        self._chunk_metadata: dict[str, dict] = {}

    @property
    def size(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: list[ChunkUnit]) -> None:
        """Add a list of ChunkUnits to both dense and lexical indices."""
        if not chunks:
            return

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                continue

            chunk_id = _chunk_id(chunk, i)
            ids.append(chunk_id)
            embeddings.append(chunk.embedding.tolist())
            documents.append(chunk.text)

            meta = {
                "unit_type": chunk.unit_type,
                "sheet_name": chunk.sheet_name,
                "workbook_name": chunk.workbook_name,
            }
            # Flatten chunk.metadata for ChromaDB (only simple types)
            for k, v in chunk.metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
            metadatas.append(meta)

            # Store full metadata
            self._chunk_metadata[chunk_id] = {**meta, **chunk.metadata}

            # BM25 corpus
            self._bm25_corpus.append(_tokenize(chunk.text))
            self._bm25_ids.append(chunk_id)
            self._bm25_texts.append(chunk.text)

        # Add to ChromaDB in batches (max 41666 per batch)
        batch_size = 5000
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            self._collection.add(
                ids=ids[start:end],
                embeddings=embeddings[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        # Rebuild BM25 index
        if self._bm25_corpus:
            self._bm25 = BM25Okapi(self._bm25_corpus)

    def query_dense(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float, str, dict]]:
        """Dense retrieval via cosine similarity.

        Returns: list of (chunk_id, score, text, metadata)
        """
        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self._collection.count()),
            include=["distances", "documents", "metadatas"],
        )

        items = []
        if results["ids"] and results["ids"][0]:
            for cid, dist, doc, meta in zip(
                results["ids"][0],
                results["distances"][0],
                results["documents"][0],
                results["metadatas"][0],
            ):
                # ChromaDB returns cosine distance; convert to similarity
                score = 1.0 - dist
                items.append((cid, score, doc, meta))
        return items

    def query_bm25(
        self, query: str, top_k: int = 20
    ) -> list[tuple[str, float, str, dict]]:
        """Lexical retrieval via BM25.

        Returns: list of (chunk_id, score, text, metadata)
        """
        if self._bm25 is None or not self._bm25_corpus:
            return []

        tokens = _tokenize(query)
        scores = self._bm25.get_scores(tokens)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        items = []
        for idx in top_indices:
            if scores[idx] > 0:
                cid = self._bm25_ids[idx]
                meta = self._chunk_metadata.get(cid, {})
                items.append((cid, float(scores[idx]), self._bm25_texts[idx], meta))
        return items

    def save_bm25(self) -> None:
        """Persist BM25 index to disk."""
        if self._persist_dir and self._bm25_corpus:
            bm25_path = self._persist_dir / "bm25_index.pkl"
            data = {
                "corpus": self._bm25_corpus,
                "ids": self._bm25_ids,
                "texts": self._bm25_texts,
                "metadata": self._chunk_metadata,
            }
            with open(bm25_path, "wb") as f:
                pickle.dump(data, f)

    def load_bm25(self) -> bool:
        """Load BM25 index from disk. Returns True if successful."""
        if self._persist_dir:
            bm25_path = self._persist_dir / "bm25_index.pkl"
            if bm25_path.exists():
                with open(bm25_path, "rb") as f:
                    data = pickle.load(f)
                self._bm25_corpus = data["corpus"]
                self._bm25_ids = data["ids"]
                self._bm25_texts = data["texts"]
                self._chunk_metadata = data["metadata"]
                if self._bm25_corpus:
                    self._bm25 = BM25Okapi(self._bm25_corpus)
                return True
        return False
