from __future__ import annotations

import sys
import json
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace

if "pettachainer.pettachainer" not in sys.modules:
    package = ModuleType("pettachainer")
    module = ModuleType("pettachainer.pettachainer")
    module.PeTTaChainer = object
    package.pettachainer = module
    sys.modules["pettachainer"] = package
    sys.modules["pettachainer.pettachainer"] = module

from core.reasoning.reasoner import Reasoner


class ReasonerPolarityTests(unittest.TestCase):
    def setUp(self):
        self.reasoner = Reasoner.__new__(Reasoner)

    def test_explicit_negative_is_a_negative_answer(self):
        self.reasoner.query = lambda query, seed_terms=None: (
            ["negative-proof"] if "(Not " in query else []
        )

        outcome = self.reasoner.query_polarity(
            "(: $prf (Authorized alice) $tv)"
        )

        self.assertEqual("negative", outcome.status)
        self.assertEqual(["negative-proof"], outcome.negative_proof)

    def test_contradiction_is_reported_as_both(self):
        self.reasoner.query = lambda query, seed_terms=None: ["proof"]

        outcome = self.reasoner.query_polarity(
            "(: $prf (Online camera_2) $tv)"
        )

        self.assertEqual("both", outcome.status)

    def test_no_positive_or_negative_proof_is_unknown(self):
        self.reasoner.query = lambda query, seed_terms=None: []

        outcome = self.reasoner.query_polarity(
            "(: $prf (Polluted lake_2) $tv)"
        )

        self.assertEqual("unknown", outcome.status)

    def test_factor_explanation_exposes_grounded_rule_requirements(self):
        self.reasoner.get_atoms = lambda: [
            "(: risk_rule (Implication (Premises (Obese $x) "
            "(Or (Sedentary $x) (FamilyHistory $x))) "
            "(Conclusions (IncreasesDiabetesRisk $x))) (STV 1.0 1.0))"
        ]
        self.reasoner.query_polarity = lambda query, seed_terms=None: SimpleNamespace(
            status="unknown"
        )
        self.reasoner.query = lambda query, seed_terms=None: []

        explanations = self.reasoner.explain_requirements(
            {"increase", "diabetes", "risk"},
            entity="abebe",
            direction="increase",
        )

        self.assertEqual(1, len(explanations))
        rendered = str(explanations[0]["requirements"])
        self.assertIn("(Obese abebe)", rendered)
        self.assertIn("(Sedentary abebe)", rendered)
        self.assertIn("(FamilyHistory abebe)", rendered)

    def test_or_premise_succeeds_when_one_alternative_is_proved(self):
        self.reasoner._prove_grounded_premise = lambda premise, seen: (
            ["b-proof"] if premise == ["B", "x"] else []
        )

        states = self.reasoner._prove_premise(
            ["Or", ["A", "x"], ["B", "x"]],
            {},
            set(),
        )

        self.assertEqual([({}, ["b-proof"])], states)

    def test_proof_sources_use_exact_atom_provenance(self):
        atom = "(: online_fact (Online camera_2) (STV 1.0 1.0))"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "provenance.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "atom": atom,
                        "source": {"text": "Camera-2 is online."},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            self.reasoner._provenance_path = str(path)

            sources = self.reasoner.sources_for_proof([atom])

        self.assertEqual(["Camera-2 is online."], sources)


if __name__ == "__main__":
    unittest.main()
