from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from langextract_atomspace.reasoning import (  # noqa: E402
    build_pln_query,
    suggest_pln_queries,
    translate_atoms_to_pln,
)


def test_translate_fact_to_pln_statement() -> None:
    result = translate_atoms_to_pln(["(eats kebede fish)"])

    assert result.rejected == []
    assert result.statements == [
        "(: kebede_fish_eats_fact (Eats kebede fish) (STV 1.0 1.0))"
    ]


def test_translate_isa_and_inheritance_to_isa() -> None:
    result = translate_atoms_to_pln([
        "(isa Sam frog)",
        "(Inheritance frog animal)",
    ])

    assert result.rejected == []
    assert result.statements == [
        "(: sam_frog_isa_fact (IsA sam frog) (STV 1.0 1.0))",
        "(: frog_animal_inheritance_fact (IsA frog animal) (STV 1.0 1.0))",
    ]


def test_translate_rule_with_single_premise() -> None:
    result = translate_atoms_to_pln([
        "(= (smart $x) (match &self (eats $x fish) True))",
    ])

    assert result.rejected == []
    assert result.statements == [
        "(: x_smart_rule (Implication (Premises (Eats $x fish)) "
        "(Conclusions (Smart $x))) (STV 1.0 1.0))"
    ]


def test_translate_rule_with_multiple_premises() -> None:
    result = translate_atoms_to_pln([
        "(= (can-fly $x) (match &self (, (isa $x animal) (has-wings $x)) True))",
    ])

    assert result.rejected == []
    assert result.statements == [
        "(: x_can_fly_rule (Implication (Premises (IsA $x animal) "
        "(HasWings $x)) (Conclusions (CanFly $x))) (STV 1.0 1.0))"
    ]


def test_translate_negation() -> None:
    result = translate_atoms_to_pln(["(not (isa Tom frog))"])

    assert result.rejected == []
    assert result.statements == [
        "(: neg_1 (Not (IsA tom frog)) (STV 1.0 1.0))"
    ]


def test_rejects_malformed_atom() -> None:
    result = translate_atoms_to_pln(["(eats kebede fish"])

    assert result.statements == []
    assert len(result.rejected) == 1
    assert "closing parenthesis" in result.rejected[0].reason


def test_build_and_suggest_queries() -> None:
    atoms = [
        "(eats kebede fish)",
        "(= (smart $x) (match &self (eats $x fish) True))",
    ]

    assert build_pln_query("(smart kebede)") == "(: $prf (Smart kebede) $tv)"
    assert suggest_pln_queries(atoms) == [
        "(: $prf (Eats kebede fish) $tv)",
        "(: $prf (Smart $x) $tv)",
    ]
