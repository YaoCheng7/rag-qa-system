"""Split loaded documents into overlapping text chunks for embedding."""
from dataclasses import dataclass
from typing import List

from app.document_loader import LoadedDocument


@dataclass
class TextChunk:
    source: str
    chunk_index: int
    text: str


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a single string into overlapping chunks by character count.

    A simple, dependency-free character-based splitter that tries to break
    on paragraph/sentence boundaries when possible, falling back to a hard
    cut at chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text = text.strip()
    if not text:
        return []

    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + chunk_size, length)

        if end < length:
            # Prefer to break at the last paragraph or sentence boundary
            # within the window, so chunks stay semantically coherent.
            window = text[start:end]
            break_point = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if break_point > chunk_size // 2:
                end = start + break_point + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= length:
            break
        start = max(end - chunk_overlap, start + 1)

    return chunks


def split_documents(
    documents: List[LoadedDocument], chunk_size: int, chunk_overlap: int
) -> List[TextChunk]:
    all_chunks: List[TextChunk] = []
    for doc in documents:
        pieces = split_text(doc.text, chunk_size, chunk_overlap)
        for idx, piece in enumerate(pieces):
            all_chunks.append(TextChunk(source=doc.source, chunk_index=idx, text=piece))
    return all_chunks
