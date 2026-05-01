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

# Make demo/src importable
_DEMO_SRC = Path(__file__).resolve().parents[2] / "demo" / "src"
if str(_DEMO_SRC) not in sys.path:
    sys.path.insert(0, str(_DEMO_SRC))

from runners import IngestResult, TimedMixin


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

    def reset(self) -> None:
        self._atoms = []
