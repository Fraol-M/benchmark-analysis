"""
PLN-RAG backend adapter for benchmarks — HTTP client mode.

Talks to the PLN-RAG FastAPI service running in Docker.
No local Python imports of PLN-RAG code required.

Start the stack first:
    docker compose up --build   (from PLN-RAG/)

Then run benchmarks normally:
    python benchmarks/compare.py --backend plnrag
"""

from __future__ import annotations

import time
from typing import List

try:
    import httpx
except ImportError:
    raise ImportError("httpx is required: pip install httpx")

from runners import IngestResult, TimedMixin

# Default base URL — override with PLNRAG_URL env var if needed
import os
_BASE_URL = os.environ.get("PLNRAG_URL", "http://localhost:8000")
_TIMEOUT = 120  # seconds — parsing + reasoning can be slow


class PLNRAGBackend(TimedMixin):
    """
    Benchmark adapter for PLN-RAG running as a Docker service.

    Uses the /ingest, /reset, and /health HTTP endpoints.
    Only measures NL → AtomSpace extraction quality.
    """

    name = "plnrag"

    def __init__(self, base_url: str = _BASE_URL):
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(timeout=_TIMEOUT)
        self._atoms: List[str] = []
        self._wait_for_service()

    # ── lifecycle ────────────────────────────────────────────────────────────

    def _wait_for_service(self, retries: int = 12, delay: float = 5.0) -> None:
        """Block until the PLN-RAG API is reachable, or raise."""
        for attempt in range(1, retries + 1):
            try:
                resp = self._client.get(f"{self._base}/health", timeout=5)
                if resp.status_code == 200:
                    info = resp.json()
                    print(
                        f"  [plnrag] API ready — parser={info.get('parser')} "
                        f"atoms={info.get('atomspace_size', 0)}"
                    )
                    return
            except Exception:
                pass
            print(f"  [plnrag] Waiting for API ({attempt}/{retries})…")
            time.sleep(delay)
        raise RuntimeError(
            f"PLN-RAG API not reachable at {self._base}. "
            "Make sure Docker is running: docker compose up --build  (from PLN-RAG/)"
        )

    # ── Backend protocol ─────────────────────────────────────────────────────

    def ingest(self, texts: List[str]) -> IngestResult:
        """POST /ingest and collect returned atoms."""
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{self._base}/ingest",
                json={"texts": texts},
                timeout=_TIMEOUT,
            )
            resp.raise_for_status()
        except Exception as exc:
            return IngestResult(error=str(exc), latency_s=time.perf_counter() - t0)

        elapsed = time.perf_counter() - t0
        data = resp.json()

        all_atoms: List[str] = []
        for item in data.get("results", []):
            all_atoms.extend(item.get("atoms", []))

        self._atoms = all_atoms
        return IngestResult(
            atoms=all_atoms,
            atom_count=len(all_atoms),
            latency_s=elapsed,
        )

    def get_atoms(self) -> List[str]:
        return list(self._atoms)

    def reset(self) -> None:
        """DELETE /reset to clear atomspace + vector DB between cases."""
        self._atoms = []
        try:
            self._client.delete(
                f"{self._base}/reset",
                json={"scope": "all"},
                timeout=30,
            )
        except Exception as exc:
            print(f"  [plnrag] reset warning: {exc}")

    def set_case_id(self, case_id: str) -> None:
        """Called by the orchestrator before each case — just reset."""
        self.reset()
