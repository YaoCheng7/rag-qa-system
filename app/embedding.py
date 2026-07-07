"""Sentence-transformers based embedding model wrapper, loaded lazily as a singleton."""
import logging
import threading
from typing import List

import numpy as np

from app import config

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


class EmbeddingError(Exception):
    """Raised when embedding text fails."""


def _get_model():
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise EmbeddingError(
                        "sentence-transformers is not installed. "
                        "Run: pip install sentence-transformers"
                    ) from exc

                logger.info("Loading embedding model: %s", config.EMBEDDING_MODEL_NAME)
                try:
                    _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
                except Exception as exc:
                    raise EmbeddingError(
                        f"Failed to load embedding model '{config.EMBEDDING_MODEL_NAME}': {exc}"
                    ) from exc
    return _model


def get_embedding_dimension() -> int:
    model = _get_model()
    return model.get_sentence_embedding_dimension()


def embed_texts(texts: List[str], batch_size: int = 32) -> np.ndarray:
    """Embed a list of texts, returning an (N, dim) float32 numpy array."""
    if not texts:
        return np.empty((0, get_embedding_dimension()), dtype="float32")

    model = _get_model()
    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
    except Exception as exc:
        raise EmbeddingError(f"Failed to embed texts: {exc}") from exc

    return embeddings.astype("float32")


def embed_query(query: str) -> np.ndarray:
    return embed_texts([query])[0]
