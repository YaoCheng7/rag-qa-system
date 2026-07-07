"""FAISS-backed vector store with JSONL metadata sidecar for chunk text/source."""
import json
import logging
import threading
from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np

from app import config

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised for vector store build/load/search failures."""


@dataclass
class RetrievedChunk:
    source: str
    chunk_index: int
    text: str
    score: float


class VectorStore:
    """Wraps a FAISS index (inner product / cosine, since embeddings are normalized)."""

    def __init__(self):
        self._index: Optional[faiss.Index] = None
        self._metadata: List[dict] = []
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._index is not None

    @property
    def size(self) -> int:
        return self._index.ntotal if self._index is not None else 0

    def build(self, embeddings: np.ndarray, metadata: List[dict]) -> None:
        if embeddings.shape[0] != len(metadata):
            raise VectorStoreError("Embeddings count does not match metadata count")
        if embeddings.shape[0] == 0:
            raise VectorStoreError("Cannot build an index with zero vectors")

        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        with self._lock:
            self._index = index
            self._metadata = metadata

    def save(self) -> None:
        if self._index is None:
            raise VectorStoreError("No index to save; build or load one first")

        config.VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            faiss.write_index(self._index, str(config.FAISS_INDEX_PATH))
            with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
                for item in self._metadata:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
        except OSError as exc:
            raise VectorStoreError(f"Failed to persist vector store: {exc}") from exc

    def load(self) -> None:
        if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
            raise VectorStoreError(
                "Vector store files not found. Call POST /build_index first."
            )

        try:
            index = faiss.read_index(str(config.FAISS_INDEX_PATH))
            metadata = []
            with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metadata.append(json.loads(line))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise VectorStoreError(f"Failed to load vector store: {exc}") from exc

        if index.ntotal != len(metadata):
            raise VectorStoreError(
                "Corrupt vector store: index size does not match metadata count"
            )

        with self._lock:
            self._index = index
            self._metadata = metadata

    def ensure_loaded(self) -> None:
        if not self.is_loaded:
            self.load()

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[RetrievedChunk]:
        self.ensure_loaded()
        if self._index.ntotal == 0:
            return []

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        scores, indices = self._index.search(query, k)

        results: List[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            meta = self._metadata[idx]
            results.append(
                RetrievedChunk(
                    source=meta["source"],
                    chunk_index=meta["chunk_index"],
                    text=meta["text"],
                    score=float(score),
                )
            )
        return results


# Module-level singleton used by the FastAPI app.
store = VectorStore()
