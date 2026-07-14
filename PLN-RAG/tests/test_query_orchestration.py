from __future__ import annotations

import unittest
import sys
from types import ModuleType

if "pettachainer.pettachainer" not in sys.modules:
    package = ModuleType("pettachainer")
    module = ModuleType("pettachainer.pettachainer")
    module.PeTTaChainer = object
    package.pettachainer = module
    sys.modules["pettachainer"] = package
    sys.modules["pettachainer.pettachainer"] = module

from core.orchestration.service import PLNRAGService


class QueryOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.service = PLNRAGService.__new__(PLNRAGService)

    def test_qdrant_target_is_context_only(self):
        candidates = self.service._query_candidates(
            "Is Abebe consuming excessive carbohydrates?",
            ["(: $prf (ConsumesHigherCarbohydrates abebe) $tv)"],
            [],
        )

        self.assertEqual([], candidates)

    def test_candidate_with_wrong_known_arity_is_rejected(self):
        class FakeReasoner:
            def predicate_arities(self):
                return {"QualifiesForMeritScholarship": {1}}

        self.service._reasoner = FakeReasoner()

        candidates = self.service._query_candidates(
            "Does Lena qualify for a merit scholarship?",
            [],
            [
                "(: $prf (QualifiesForMeritScholarship lena merit) $tv)",
                "(: $prf (QualifiesForMeritScholarship lena) $tv)",
            ],
        )

        self.assertEqual(
            [("(: $prf (QualifiesForMeritScholarship lena) $tv)", "parser")],
            candidates,
        )

    def test_matching_parser_target_is_executable(self):
        candidates = self.service._query_candidates(
            "Is Abebe consuming excessive carbohydrates?",
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
            ["(: $prf (ConsumesHigherCarbohydrates abebe) $tv)"],
        )

        self.assertEqual(
            [("(: $prf (ConsumesHigherCarbohydrates abebe) $tv)", "parser")],
            candidates,
        )

    def test_rejected_parser_target_is_not_restored(self):
        candidates = self.service._query_candidates(
            "Is Abebe currently obese?",
            [],
            ["(: $prf (HasIncreasedWeight abebe) $tv)"],
        )

        self.assertEqual([], candidates)

    def test_sufficiency_never_executes_supporting_fact(self):
        candidates = self.service._query_candidates(
            "Is Abebe's pasta consumption alone sufficient to cause diabetes?",
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
            ["(: $prf (EatsFrequently abebe pasta) $tv)"],
        )

        self.assertEqual([], candidates)

    def test_trusted_atomspace_conclusion_recovers_missing_parser_query(self):
        class FakeReasoner:
            def proposition_signatures(self):
                return [
                    {
                        "head": "ClassifiedAsPolluted",
                        "args": ["$lake"],
                        "arity": 1,
                        "variables": ["$lake"],
                        "negated": False,
                        "role": "conclusion",
                    },
                    {
                        "head": "IsA",
                        "args": ["lake_aster", "lake"],
                        "arity": 2,
                        "variables": [],
                        "negated": False,
                        "role": "fact",
                    },
                ]

            def predicate_arities(self):
                return {"ClassifiedAsPolluted": {1}}

        self.service._reasoner = FakeReasoner()
        trusted = self.service._deterministic_query_candidates(
            "Is Lake Aster classified as polluted?"
        )
        candidates = self.service._query_candidates(
            "Is Lake Aster classified as polluted?",
            [],
            [],
            trusted,
        )

        self.assertEqual(
            [
                (
                    "(: $prf (ClassifiedAsPolluted lake_aster) $tv)",
                    "deterministic",
                )
            ],
            candidates,
        )


if __name__ == "__main__":
    unittest.main()
