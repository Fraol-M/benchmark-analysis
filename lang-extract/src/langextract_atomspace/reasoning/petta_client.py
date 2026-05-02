from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable

import httpx


class PeTTaClientError(RuntimeError):
    pass


@dataclass
class PeTTaLoadResult:
    added: list[str] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    atomspace_size: int = 0


@dataclass
class PeTTaQueryResult:
    query: str
    proof_traces: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def has_proof(self) -> bool:
        return bool(self.proof_traces)


class PeTTaClient:
    """HTTP client for the Dockerized PeTTaChainer reasoner."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 120.0,
    ):
        self.base_url = (
            base_url
            or os.environ.get("PETTA_REASONER_URL")
            or "http://localhost:8010"
        ).rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def reset(self) -> dict[str, Any]:
        return self._request("POST", "/reset")

    def load(
        self,
        statements: Iterable[str],
        *,
        reset: bool = False,
    ) -> PeTTaLoadResult:
        data = self._request(
            "POST",
            "/load",
            json={"statements": list(statements), "reset": reset},
        )
        return PeTTaLoadResult(
            added=data.get("added", []),
            rejected=data.get("rejected", []),
            atomspace_size=data.get("atomspace_size", 0),
        )

    def query(self, query: str) -> PeTTaQueryResult:
        data = self._request("POST", "/query", json={"query": query})
        return PeTTaQueryResult(
            query=data.get("query", query),
            proof_traces=[str(item) for item in data.get("proof_traces", [])],
            raw=data,
        )

    def _request(self, method: str, path: str, **kwargs) -> dict[str, Any]:
        try:
            response = self._client.request(method, f"{self.base_url}{path}", **kwargs)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            raise PeTTaClientError(
                f"PeTTa reasoner request failed ({method} {path}): {exc}"
            ) from exc
