"""Central configuration loaded from environment variables (.env supported)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Data / storage paths ---
RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw")))
PROCESSED_DATA_DIR = Path(os.getenv("PROCESSED_DATA_DIR", str(BASE_DIR / "data" / "processed")))
VECTOR_STORE_DIR = Path(os.getenv("VECTOR_STORE_DIR", str(BASE_DIR / "vector_store")))
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "index.faiss"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.jsonl"

# --- Text splitting ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --- Embedding ---
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5"
)

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "4"))

# --- LLM (Chat Completions compatible API) ---
API_BASE_URL = os.getenv("API_BASE_URL", "")
API_KEY = os.getenv("API_KEY", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# Ensure required directories exist at import time.
for _dir in (RAW_DATA_DIR, PROCESSED_DATA_DIR, VECTOR_STORE_DIR):
    _dir.mkdir(parents=True, exist_ok=True)
