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

from runners import IngestResult, QueryResult, TimedMixin


class DemoBackend(TimedMixin):
    """
    Benchmark adapter for the Demo (LangExtract → Hyperon MeTTa).

    Uses run_extraction() for ingest and MeTTa.run() for queries.
    """

    name = "demo"

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
    ):
        self._model_id = model_id or os.environ.get("LANGEXTRACT_MODEL_ID", "gemini-2.5-flash")
        self._api_key = api_key
        self._metta = None
        self._atoms: List[str] = []
        self._metta_str: str = ""
        self._extraction_result: dict | None = None

    def ingest(self, texts: List[str]) -> IngestResult:
        """Run extraction on joined texts and load into MeTTa runner."""
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

        self._extraction_result = result
        self._metta = result["metta"]
        self._atoms = [str(a) for a in result["atoms"]]
        self._metta_str = result["metta_str"]

        return IngestResult(
            atoms=self._atoms,
            atom_count=len(self._atoms),
            latency_s=elapsed,
        )

    def query(self, question: str) -> QueryResult:
        """
        Run a MeTTa query against the loaded atoms.

        Since Demo has no NL→query translation, we try a set of
        common query patterns and return whatever matches.
        """
        if self._metta is None:
            return QueryResult(
                answer="No atoms loaded",
                error="Must call ingest() first",
            )

        # Try multiple query shapes to find atoms that match the question
        query_templates = self._generate_query_templates(question)

        all_results: List[str] = []
        total_elapsed = 0.0

        for query in query_templates:
            result, elapsed, error = self._timed(self._metta.run, query)
            total_elapsed += elapsed
            if error:
                continue
            if result:
                for block in result:
                    for atom in block:
                        atom_str = str(atom)
                        if atom_str and atom_str not in ("()", "[]", ""):
                            all_results.append(atom_str)

        # Dedupe
        seen = set()
        unique_results = []
        for r in all_results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        answer = " ".join(unique_results) if unique_results else "No proof found"

        return QueryResult(
            answer=answer,
            raw_atoms=unique_results,
            latency_s=total_elapsed,
        )

    def get_atoms(self) -> List[str]:
        return list(self._atoms)

    def reset(self) -> None:
        self._metta = None
        self._atoms = []
        self._metta_str = ""
        self._extraction_result = None

    def _generate_query_templates(self, question: str) -> List[str]:
        """
        Generate MeTTa query templates from a question.

        This is a heuristic — the Demo has no NL→query translator,
        so we try common patterns based on the atoms in the KB.
        """
        templates = []

        # Get all predicate heads from atoms
        heads = set()
        for atom_str in self._atoms:
            if atom_str.startswith("(") and not atom_str.startswith("(="):
                parts = atom_str.strip("()").split()
                if parts:
                    heads.add(parts[0])

        # Normalize question for keyword matching
        q = question.lower().strip().rstrip("?").strip()
        q_words = set(q.split())

        # Strategy 1: Match all atoms of each predicate head and filter by question keywords
        for head in heads:
            if head in ("=", ":", "not"):
                continue
            templates.append(f"! (match &self ({head} $a $b) ({head} $a $b))")
            templates.append(f"! (match &self ({head} $a) ({head} $a))")
            templates.append(f"! (match &self ({head} $a $b $c) ({head} $a $b $c))")

        # Strategy 2: Try evaluating rule heads (forward-chain one step)
        templates.append("! (match &self (= $head $body) $head)")

        # Strategy 3: Get everything
        templates.append("! (match &self $x $x)")

        return templates
