"""
Demo backend adapter for benchmarks.

Calls the LangExtract → AtomSpace pipeline in-process.
No Docker, no network — just Python imports.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

# Make lang-extract/src importable
_DEMO_SRC = Path(__file__).resolve().parents[2] / "lang-extract" / "src"
if str(_DEMO_SRC) not in sys.path:
    sys.path.insert(0, str(_DEMO_SRC))

from runners import IngestResult, QueryResult, TimedMixin


class DemoBackend(TimedMixin):
    """
    Benchmark adapter for the Demo (LangExtract → Hyperon MeTTa).

    Only measures NL → AtomSpace extraction quality.
    """

    name = "demo"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
    ):
        self._model_id = model_id or os.environ.get("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
        self._api_key = api_key
        self._atoms: List[str] = []

    def ingest(self, texts: List[str]) -> IngestResult:
        """Run extraction on joined texts and populate the atom list."""
        from langextract_atomspace.pipeline import run_extraction

        combined = " ".join(texts)
        result, elapsed, error = self._timed(
            run_extraction,
            text=combined,
            model_id=self._model_id,
            api_key=self._api_key,
            skip_fuzzy=True,
        )

        if error:
            return IngestResult(error=error, latency_s=elapsed)

        self._atoms = [str(a) for a in result["atoms"]]

        return IngestResult(
            atoms=self._atoms,
            atom_count=len(self._atoms),
            latency_s=elapsed,
        )

    def get_atoms(self) -> List[str]:
        return list(self._atoms)

    def query(self, query_spec: dict | str) -> QueryResult:
        """Translate current MeTTa atoms to PLN and query PeTTaChainer."""
        if isinstance(query_spec, dict):
            query = query_spec.get("pln_query") or query_spec.get("query") or ""
        else:
            query = query_spec

        if not query.strip():
            return QueryResult(error="missing pln_query")

        def _run():
            from langextract_atomspace.reasoning import PeTTaClient, translate_atoms_to_pln

            translation = translate_atoms_to_pln(self._atoms)
            if not translation.statements:
                return QueryResult(
                    query=query,
                    error="no PLN statements translated from extracted atoms",
                    translation_rejected_count=translation.rejected_count,
                )

            client = PeTTaClient()
            load_result = client.load(translation.statements, reset=True)
            query_result = client.query(query)
            return QueryResult(
                query=query,
                proof_traces=query_result.proof_traces,
                raw={
                    "load_added": len(load_result.added),
                    "load_rejected": load_result.rejected,
                    "atomspace_size": load_result.atomspace_size,
                    "translation_rejected": [
                        {"atom": item.atom, "reason": item.reason}
                        for item in translation.rejected
                    ],
                    "service_raw": query_result.raw,
                },
                translated_statement_count=translation.statement_count,
                translation_rejected_count=translation.rejected_count,
            )

        result, elapsed, error = self._timed(_run)
        if error:
            return QueryResult(query=query, latency_s=elapsed, error=error)
        result.latency_s = elapsed
        return result

    def reset(self) -> None:
        self._atoms = []
