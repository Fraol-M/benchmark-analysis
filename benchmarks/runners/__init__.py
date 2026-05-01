"""
Benchmark runner protocol and shared types.

Every backend adapter implements the Backend protocol.
The orchestrator (compare.py) only depends on this interface.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, List


@dataclass
class IngestResult:
    """What a backend returns after ingesting text."""
    atoms: List[str] = field(default_factory=list)
    atom_count: int = 0
    latency_s: float = 0.0
    error: str | None = None


@runtime_checkable
class Backend(Protocol):
    """
    Pluggable interface for benchmark backends.

    Two implementations:
      - DemoBackend   (langextract → MeTTa / Hyperon)
      - PLNRAGBackend (NL2PLN → PeTTaChainer)

    Benchmarks measure NL → AtomSpace extraction quality only.
    No query/reasoning step is performed.
    """

    name: str

    def ingest(self, texts: List[str]) -> IngestResult:
        """Parse texts into symbolic atoms and load them into the KB."""
        ...

    def get_atoms(self) -> List[str]:
        """Return all atoms currently in the KB."""
        ...

    def reset(self) -> None:
        """Clear all state so the next case starts fresh."""
        ...


class TimedMixin:
    """Utility to time a callable and return (result, elapsed_s, error)."""

    @staticmethod
    def _timed(fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            result = fn(*args, **kwargs)
            return result, time.perf_counter() - t0, None
        except Exception as exc:
            return None, time.perf_counter() - t0, str(exc)
