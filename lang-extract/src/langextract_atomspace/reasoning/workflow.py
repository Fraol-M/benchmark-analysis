from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .pln_adapter import suggest_pln_queries, translate_atoms_to_pln


def prepare_pln_reasoning_payload(extraction_result: dict[str, Any]) -> dict[str, Any]:
    """Build all Phase 2 artifacts from a Phase 1 extraction result."""
    atoms = extraction_result.get("atoms") or []
    translation = translate_atoms_to_pln(atoms)
    suggestions = suggest_pln_queries(atoms)
    return {
        "statements": translation.statements,
        "rejected": [asdict(item) for item in translation.rejected],
        "atom_map": translation.atom_map,
        "suggested_queries": suggestions,
        "statement_count": translation.statement_count,
        "rejected_count": translation.rejected_count,
    }
