"""
PLN-RAG backend adapter for benchmarks.

Calls PLNRAGService in-process — no Docker needed.
Requires PeTTaChainer, NL2PLN, Qdrant, and Ollama to be available.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from typing import List

# Make PLN-RAG importable
_PLNRAG_ROOT = Path(__file__).resolve().parents[1] / "PLN-RAG"
if str(_PLNRAG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLNRAG_ROOT))

from runners import IngestResult, QueryResult, TimedMixin


class PLNRAGBackend(TimedMixin):
    """
    Benchmark adapter for PLN-RAG (NL2PLN → PeTTaChainer).

    Uses PLNRAGService in-process for both ingest and query.
    Each benchmark case gets its own isolated Qdrant collection and
    atomspace file so cases don't leak state.
    """

    name = "plnrag"

    def __init__(self):
        self._service = None
        self._run_id = uuid.uuid4().hex[:8]
        self._case_id: str = ""
        self._atomspace_path: str = ""
        self._collection: str = ""

    def _init_service(self, case_id: str = "default"):
        """Lazy-init with isolated storage per case."""
        from config import get_settings

        self._case_id = case_id
        self._collection = f"bench_{self._run_id}_{case_id}".replace("-", "_")
        self._atomspace_path = f"data/atomspace/bench_{self._run_id}_{case_id}.metta"

        # Override settings for isolation
        os.environ["QDRANT_COLLECTION"] = self._collection
        os.environ["ATOMSPACE_PATH"] = self._atomspace_path
        os.environ["QUERY_FALLBACK_ENABLED"] = "true"
        get_settings.cache_clear()

        from parsers import get_parser
        from core.service import PLNRAGService

        parser = get_parser()
        self._service = PLNRAGService(parser)

    def ingest(self, texts: List[str]) -> IngestResult:
        """Ingest texts into the PLN-RAG knowledge base."""
        import asyncio

        if self._service is None:
            self._init_service()

        async def _do_ingest():
            return await self._service.ingest_batch(texts)

        result, elapsed, error = self._timed(asyncio.run, _do_ingest())

        if error:
            return IngestResult(error=error, latency_s=elapsed)

        all_atoms = []
        for item in result:
            all_atoms.extend(item.atoms)

        return IngestResult(
            atoms=all_atoms,
            atom_count=len(all_atoms),
            latency_s=elapsed,
        )

    def query(self, question: str) -> QueryResult:
        """Ask a question against the PLN-RAG knowledge base."""
        import asyncio

        if self._service is None:
            return QueryResult(
                answer="No service initialized",
                error="Must call ingest() first",
            )

        async def _do_query():
            return await self._service.query(question)

        result, elapsed, error = self._timed(asyncio.run, _do_query())

        if error:
            return QueryResult(
                answer=f"Query failed: {error}",
                error=error,
                latency_s=elapsed,
            )

        return QueryResult(
            answer=result.answer,
            raw_atoms=[result.raw_proof] if result.raw_proof else [],
            latency_s=elapsed,
        )

    def get_atoms(self) -> List[str]:
        """Read atoms from the atomspace file."""
        if not os.path.exists(self._atomspace_path):
            return []
        with open(self._atomspace_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def reset(self) -> None:
        """Clear this case's atomspace and Qdrant collection."""
        if self._service:
            try:
                self._service.reset("all")
            except Exception:
                pass

        # Clean up atomspace file
        if self._atomspace_path and os.path.exists(self._atomspace_path):
            try:
                os.remove(self._atomspace_path)
            except Exception:
                pass

        self._service = None

    def set_case_id(self, case_id: str) -> None:
        """Set the case ID for isolated storage. Call before ingest()."""
        self.reset()
        self._init_service(case_id)
