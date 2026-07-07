"""FastAPI application exposing the RAG document Q&A endpoints."""
import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app import config, rag_pipeline
from app.vector_store import store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="RAG Document QA System",
    description="Ask questions over local documents (txt/md/pdf) using FAISS retrieval + an LLM.",
    version="1.0.0",
)


class BuildIndexResponse(BaseModel):
    status: str
    documents_indexed: int
    chunks_indexed: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int | None = Field(None, gt=0, description="Number of chunks to retrieve")


class SourceChunkResponse(BaseModel):
    source: str
    chunk_index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunkResponse]


class HealthResponse(BaseModel):
    status: str
    index_loaded: bool
    index_size: int


@app.post("/build_index", response_model=BuildIndexResponse)
def build_index() -> BuildIndexResponse:
    try:
        result = rag_pipeline.build_index()
    except rag_pipeline.RAGError as exc:
        logger.exception("build_index failed")
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return BuildIndexResponse(
        status="ok",
        documents_indexed=result["documents_indexed"],
        chunks_indexed=result["chunks_indexed"],
    )


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        result = rag_pipeline.ask(request.question, top_k=request.top_k)
    except rag_pipeline.RAGError as exc:
        logger.exception("ask failed")
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceChunkResponse(
                source=s.source, chunk_index=s.chunk_index, text=s.text, score=s.score
            )
            for s in result.sources
        ],
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if not store.is_loaded:
        try:
            store.load()
        except Exception:
            pass  # Index simply hasn't been built yet; health is still "ok".

    return HealthResponse(
        status="ok",
        index_loaded=store.is_loaded,
        index_size=store.size,
    )
