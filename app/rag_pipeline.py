"""Orchestrates document loading, chunking, embedding, indexing, and RAG Q&A."""
import logging
from dataclasses import dataclass
from typing import List

from app import config, embedding, llm
from app.document_loader import load_documents
from app.text_splitter import split_documents
from app.vector_store import VectorStoreError, store

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant answering questions using only the provided "
    "context. If the context does not contain the answer, say you don't know "
    "instead of making something up. Cite sources by their [source] tag when relevant."
)


class RAGError(Exception):
    """Raised when the index build or ask flow fails."""


@dataclass
class SourceChunk:
    source: str
    chunk_index: int
    text: str
    score: float


@dataclass
class AskResult:
    answer: str
    sources: List[SourceChunk]


def build_index() -> dict:
    """Load raw documents, split, embed, and (re)build the FAISS index."""
    try:
        documents = load_documents(config.RAW_DATA_DIR)
    except Exception as exc:
        raise RAGError(f"Failed to load documents: {exc}") from exc

    if not documents:
        raise RAGError(
            f"No supported documents (.txt, .md, .pdf) found in {config.RAW_DATA_DIR}"
        )

    chunks = split_documents(documents, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    if not chunks:
        raise RAGError("Document splitting produced zero chunks")

    texts = [c.text for c in chunks]
    try:
        vectors = embedding.embed_texts(texts)
    except embedding.EmbeddingError as exc:
        raise RAGError(str(exc)) from exc

    metadata = [
        {"source": c.source, "chunk_index": c.chunk_index, "text": c.text} for c in chunks
    ]

    try:
        store.build(vectors, metadata)
        store.save()
    except VectorStoreError as exc:
        raise RAGError(f"Failed to build/save vector store: {exc}") from exc

    return {
        "documents_indexed": len(documents),
        "chunks_indexed": len(chunks),
    }


def _format_context(chunks: List[SourceChunk]) -> str:
    parts = []
    for chunk in chunks:
        parts.append(f"[{chunk.source}#{chunk.chunk_index}]\n{chunk.text}")
    return "\n\n---\n\n".join(parts)


def ask(question: str, top_k: int = None) -> AskResult:
    if not question or not question.strip():
        raise RAGError("Question must not be empty")

    top_k = top_k or config.TOP_K

    try:
        query_vector = embedding.embed_query(question)
    except embedding.EmbeddingError as exc:
        raise RAGError(str(exc)) from exc

    try:
        retrieved = store.search(query_vector, top_k)
    except VectorStoreError as exc:
        raise RAGError(str(exc)) from exc

    sources = [
        SourceChunk(source=r.source, chunk_index=r.chunk_index, text=r.text, score=r.score)
        for r in retrieved
    ]

    if not sources:
        return AskResult(
            answer="No relevant context was found in the knowledge base to answer this question.",
            sources=[],
        )

    context = _format_context(sources)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    messages = [
        llm.ChatMessage(role="system", content=SYSTEM_PROMPT),
        llm.ChatMessage(role="user", content=user_prompt),
    ]

    try:
        answer = llm.chat_completion(messages)
    except llm.LLMError as exc:
        raise RAGError(str(exc)) from exc

    return AskResult(answer=answer, sources=sources)
