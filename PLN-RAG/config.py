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
    gemini_model: str = "gemini/gemini-2.5-flash"

    # LangExtract parser (NL -> LangExtract objects -> canonical PLN)
    langextract_api_key: Optional[str] = None
    langextract_model_id: str = "gemini-2.5-flash"
    langextract_model_url: Optional[str] = None
    langextract_examples_path: str = "data/langextract_examples.json"
    langextract_extraction_passes: int = 1
    langextract_max_workers: int = 1
    langextract_skip_fuzzy: bool = True
    langextract_chunk_size: Optional[int] = None

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "pln_rag"
    ollama_url: str = "http://localhost:11434/api/embeddings"
    ollama_model: str = "nomic-embed-text"
    use_vector_store: bool = True

    # Atomspace persistence
    atomspace_path: str = "data/atomspace/kb.metta"

    # FAISS predicate store (no longer used)
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
    query_alignment_enabled: bool = True
    query_alignment_top_k: int = 8
    query_alignment_min_score: float = 0.55
    query_candidate_max_tries: int = 5
    answer_generation_enabled: bool = True
    source_lookup_max_atoms: int = 0

    # Predicate registry / mapping graph
    predicate_registry_enabled: bool = True
    predicate_registry_path: str = "data/predicate_registry.json"
    predicate_mapping_enabled: bool = True
    predicate_mapping_llm_enabled: bool = True
    predicate_mapping_collection: str = "pln_rag_predicates"
    predicate_mapping_top_k: int = 6
    predicate_mapping_max_candidates: int = 6
    predicate_mapping_min_score: float = 0.62
    predicate_mapping_proof_threshold: float = 0.82
    predicate_mapping_timeout: int = 20
    predicate_mapping_total_timeout: int = 25

    model_config = ConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
