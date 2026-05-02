from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from pettachainer.pettachainer import PeTTaChainer


class LoadRequest(BaseModel):
    statements: list[str] = Field(default_factory=list)
    reset: bool = False


class QueryRequest(BaseModel):
    query: str


class PeTTaReasoner:
    def __init__(self) -> None:
        self.atomspace_path = Path(
            os.environ.get("PETTA_ATOMSPACE_PATH", "/app/data/atomspace/kb.metta")
        )
        self.atomspace_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handler = PeTTaChainer()
        self._load_from_disk()

    def _load_from_disk(self) -> None:
        if not self.atomspace_path.exists():
            return
        for line in self.atomspace_path.read_text(encoding="utf-8").splitlines():
            atom = line.strip()
            if not atom:
                continue
            try:
                self._handler.add_atom(atom)
            except Exception:
                # Keep startup resilient if an older file contains bad atoms.
                pass

    def reset(self) -> None:
        with self._lock:
            self._handler = PeTTaChainer()
            if self.atomspace_path.exists():
                self.atomspace_path.unlink()

    def load(self, statements: list[str], *, reset: bool = False) -> dict[str, Any]:
        if reset:
            self.reset()

        added: list[str] = []
        rejected: list[dict[str, str]] = []

        with self._lock:
            with self.atomspace_path.open("a", encoding="utf-8") as handle:
                for statement in statements:
                    clean = " ".join(statement.split())
                    if not clean:
                        continue
                    try:
                        self._handler.add_atom(clean)
                    except Exception as exc:
                        rejected.append({"statement": clean, "reason": str(exc)})
                        continue
                    handle.write(clean + "\n")
                    added.append(clean)

        return {
            "added": added,
            "rejected": rejected,
            "atomspace_size": self.size,
        }

    def query(self, query: str) -> list[str]:
        clean = " ".join(query.split())
        if not clean:
            return []
        result = self._handler.query(clean)
        return [str(item) for item in result] if result else []

    @property
    def size(self) -> int:
        if not self.atomspace_path.exists():
            return 0
        return sum(
            1
            for line in self.atomspace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )


app = FastAPI(title="LangExtract PeTTa Reasoner", version="0.1.0")
reasoner = PeTTaReasoner()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "atomspace_size": reasoner.size,
        "atomspace_path": str(reasoner.atomspace_path),
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    reasoner.reset()
    return {"status": "ok", "atomspace_size": reasoner.size}


@app.post("/load")
def load(req: LoadRequest) -> dict[str, Any]:
    return reasoner.load(req.statements, reset=req.reset)


@app.post("/query")
def query(req: QueryRequest) -> dict[str, Any]:
    proof_traces = reasoner.query(req.query)
    return {
        "query": req.query,
        "proof_traces": proof_traces,
        "has_proof": bool(proof_traces),
    }
