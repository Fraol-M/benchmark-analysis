from __future__ import annotations

import os
import sys
import tempfile
import unittest
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.answer_generator import AnswerGenerator
from core.query_alignment import (
    build_aligned_queries,
    extract_forward_seed_terms,
    filter_queries_by_question_intent,
)


ABEBE_RULE = (
    "(: x_consume_higher_carbohydrate_rule "
    "(Implication "
    "(Premises "
    "(IsA $x person) "
    "(EatsFrequently $x pasta) "
    "(EatsLargePortions $x) "
    "(IsRefined pasta) "
    "(Not (IsWholeGrain pasta))) "
    "(Conclusions (ConsumesHigherCarbohydrates $x))) "
    "(STV 1.0 1.0))"
)
ABEBE_FACTS = [
    "(: abebe_is_person (IsA abebe person) (STV 1.0 1.0))",
    "(: abebe_pasta_eat_frequently_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))",
    "(: abebe_eat_large_portion_fact (EatsLargePortions abebe) (STV 1.0 1.0))",
    "(: pasta_is_refined_fact (IsRefined pasta) (STV 1.0 1.0))",
    "(: pasta_is_whole_grain_neg (Not (IsWholeGrain pasta)) (STV 1.0 1.0))",
]


class ForwardFirstQueryTests(unittest.TestCase):
    def test_qdrant_alignment_ranks_stored_rule_conclusion_over_generic_consume(self):
        result = build_aligned_queries(
            "Is Abebe consuming excessive carbohydrates?",
            [
                {
                    "score": 0.91,
                    "nl": "People who frequently eat pasta consume higher carbohydrates. Abebe eats pasta.",
                    "query_targets": [
                        "(IsA abebe person)",
                        "(Consumes abebe)",
                        "(ConsumesHigherCarbohydrates $x)",
                    ],
                    "pln": [ABEBE_RULE, *ABEBE_FACTS],
                }
            ],
        )

        self.assertEqual(
            result.queries[0],
            "(: $prf (ConsumesHigherCarbohydrates abebe) $tv)",
        )
        self.assertIn("(: $prf (Consumes abebe) $tv)", result.queries)

    def test_qdrant_seed_terms_are_grounded_facts_only(self):
        seeds = extract_forward_seed_terms(
            [{"score": 0.9, "pln": [ABEBE_RULE, *ABEBE_FACTS]}]
        )

        self.assertIn("(IsA abebe person)", seeds)
        self.assertIn("(Not (IsWholeGrain pasta))", seeds)
        self.assertFalse(any("Implication" in seed for seed in seeds))

    def test_obesity_risk_question_filters_high_carb_intermediate_target(self):
        question = "Is Abebe at risk of becoming obese?"

        result = build_aligned_queries(
            question,
            [
                {
                    "score": 0.91,
                    "nl": "Abebe eats refined pasta in large portions, which supports high carb consumption.",
                    "query_targets": [
                        "(ConsumesHighCarbs $x)",
                        "(LeadsToObesity $x)",
                        "(AtRiskOfObesity $x)",
                    ],
                    "pln": [],
                }
            ],
        )

        self.assertNotIn("(: $prf (ConsumesHighCarbs abebe) $tv)", result.queries)
        self.assertIn("(: $prf (LeadsToObesity abebe) $tv)", result.queries)
        self.assertIn("(: $prf (AtRiskOfObesity abebe) $tv)", result.queries)

    def test_parser_candidates_are_filtered_to_question_intent(self):
        filtered = filter_queries_by_question_intent(
            "Is Abebe at risk of becoming obese?",
            [
                "(: $prf (ReducesObesityRisk abebe) $tv)",
                "(: $prf (AtElevatedDiabetesRisk abebe) $tv)",
                "(: $prf (AtRiskOfObesity abebe) $tv)",
                "(: $prf (ConsumesHighCarbs abebe) $tv)",
            ],
        )

        self.assertIn("(: $prf (ReducesObesityRisk abebe) $tv)", filtered)
        self.assertIn("(: $prf (AtRiskOfObesity abebe) $tv)", filtered)
        self.assertNotIn("(: $prf (AtElevatedDiabetesRisk abebe) $tv)", filtered)
        self.assertNotIn("(: $prf (ConsumesHighCarbs abebe) $tv)", filtered)

    def test_answer_generator_uses_executed_target_generically(self):
        answer = AnswerGenerator().generate(
            "Is Abebe consuming excessive carbohydrates?",
            [
                "(: abebe_consumes_derived "
                "(ConsumesHigherCarbohydrates abebe) "
                "(STV 1.0 1.0))"
            ],
            executed_query="(: $prf (ConsumesHigherCarbohydrates abebe) $tv)",
        )

        self.assertEqual(
            answer,
            "Yes. The proof establishes (ConsumesHigherCarbohydrates abebe).",
        )

    def test_reasoner_forward_first_query_derives_from_qdrant_seed_facts(self):
        if importlib.util.find_spec("pettachainer") is None:
            self.skipTest("pettachainer is not installed in this Python environment")

        with tempfile.TemporaryDirectory() as temp_dir:
            old_atomspace = os.environ.get("ATOMSPACE_PATH")
            os.environ["ATOMSPACE_PATH"] = str(Path(temp_dir) / "kb.metta")
            try:
                from config import get_settings

                get_settings.cache_clear()
                from core.reasoner import Reasoner

                reasoner = Reasoner()
                reasoner.add_statements([ABEBE_RULE, *ABEBE_FACTS])

                proof = reasoner.query(
                    "(: $prf (ConsumesHigherCarbohydrates abebe) $tv)",
                    seed_terms=[
                        "(IsA abebe person)",
                        "(EatsFrequently abebe pasta)",
                        "(EatsLargePortions abebe)",
                        "(IsRefined pasta)",
                        "(Not (IsWholeGrain pasta))",
                    ],
                )
            finally:
                if old_atomspace is None:
                    os.environ.pop("ATOMSPACE_PATH", None)
                else:
                    os.environ["ATOMSPACE_PATH"] = old_atomspace
                from config import get_settings

                get_settings.cache_clear()

        self.assertTrue(proof)
        self.assertTrue(
            any("(ConsumesHigherCarbohydrates abebe)" in item for item in proof)
        )


if __name__ == "__main__":
    unittest.main()
