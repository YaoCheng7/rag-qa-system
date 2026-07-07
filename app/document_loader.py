"""Load raw documents (.txt, .md, .pdf) from a directory into plain text."""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf"}


@dataclass
class LoadedDocument:
    source: str  # relative file path, used for citation
    text: str


class DocumentLoadError(Exception):
    """Raised when a document cannot be read or parsed."""


def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fall back to a lenient decode rather than failing the whole build.
        return path.read_text(encoding="utf-8", errors="ignore")


def _read_pdf_file(path: Path) -> str:
    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError) as exc:
        raise DocumentLoadError(f"Failed to open PDF '{path}': {exc}") from exc

    pages_text = []
    for page_number, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as exc:  # pypdf can raise assorted errors per-page
            logger.warning("Skipping unreadable page %d in %s: %s", page_number, path, exc)
    return "\n".join(pages_text)


def load_documents(raw_dir: Path) -> List[LoadedDocument]:
    """Recursively load all supported documents under raw_dir.

    Files that fail to parse are logged and skipped instead of aborting the
    whole build.
    """
    if not raw_dir.exists():
        raise DocumentLoadError(f"Raw data directory does not exist: {raw_dir}")

    documents: List[LoadedDocument] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            if path.suffix.lower() == ".pdf":
                text = _read_pdf_file(path)
            else:
                text = _read_text_file(path)
        except DocumentLoadError as exc:
            logger.warning("Skipping document %s: %s", path, exc)
            continue

        text = text.strip()
        if not text:
            logger.warning("Skipping empty document: %s", path)
            continue

        documents.append(LoadedDocument(source=str(path.relative_to(raw_dir)), text=text))

    if not documents:
        logger.warning("No readable documents found in %s", raw_dir)

    return documents
