from pydantic_settings import BaseSettings
from functools import lru_cache
from pydantic import ConfigDict
from typing import Optional
from pathlib import Path

# Root .env is two levels up from this file (PLN-RAG/config.py -> root)
_ROOT_ENV = Path(__file__).resolve().parent.parent / ".env"
_LOCAL_ENV = Path(__file__).resolve().parent / ".env"

# Use root .env if it exists, otherwise fall back to PLN-RAG/.env
_ENV_FILE = str(_ROOT_ENV) if _ROOT_ENV.exists() else str(_LOCAL_ENV)


class Settings(BaseSettings):
    # LLM — provide either OpenAI or Gemini credentials
    openai_api_key: Optional[str] = None
    openai_model: str = "openai/gpt-4o-mini"

    # Gemini (uses Google's OpenAI-compatible endpoint)
    gemini_api_key: Optional[str] = None
    gemini_model: str = "gemini/gemini-2.0-flash"

    # Options: "nl2pln" | "canonical_pln" | "manhin"
    parser: str = "canonical_pln"
    nl2pln_module_path: str = "data/simba_all.json"
    canonical_pln_nl2pln_module_path: str = "data/simba_canonical_pln.json"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pln_rag"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "nomic-embed-text"

    # Atomspace persistence
    atomspace_path: str = "data/atomspace/kb.metta"

    # FAISS predicate store (used by Manhin parser)
    faiss_path: str = "data/faiss"

    # Processing
    chunk_size: int = 512  # chars per chunk
    chunk_overlap: int = 64  # overlap between chunks
    context_top_k: int = 10  # atoms to retrieve as parser context

    # Reasoning
    chaining_timeout: int = 30  # seconds before proof search is killed
    chaining_max_steps: int = 100

    # Query execution
    query_fallback_enabled: bool = True

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
