from __future__ import annotations

import unittest

from core.query.intent import (
    QuestionMode,
    parse_question_intent,
    query_intent_score,
    query_matches_intent,
)


class QueryIntentTests(unittest.TestCase):
    def test_sufficiency_is_not_reduced_to_supporting_fact(self):
        intent = parse_question_intent(
            "Is Abebe's pasta consumption alone sufficient to cause diabetes?"
        )

        self.assertEqual(QuestionMode.SUFFICIENCY, intent.mode)
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (EatsFrequently abebe pasta) $tv)",
            )
        )

    def test_factor_question_has_non_atomic_answer_mode(self):
        intent = parse_question_intent("What factors increase Abebe's diabetes risk?")

        self.assertEqual(QuestionMode.FACTORS, intent.mode)
        self.assertEqual("increase", intent.direction)
        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (AtElevatedDiabetesRisk abebe) $tv)",
            )
        )

    def test_cross_domain_boolean_targets_require_relation_match(self):
        cases = [
            (
                "Is Motor-1 overheating?",
                "(: $prf (Overheating motor_1) $tv)",
                "(: $prf (Running motor_1) $tv)",
            ),
            (
                "Is Alice authorized?",
                "(: $prf (Authorized alice) $tv)",
                "(: $prf (Employee alice) $tv)",
            ),
            (
                "Is Lake-2 polluted?",
                "(: $prf (Polluted lake_2) $tv)",
                "(: $prf (NearFactory lake_2) $tv)",
            ),
        ]
        for question, expected, unrelated in cases:
            with self.subTest(question=question):
                intent = parse_question_intent(question)
                self.assertTrue(query_matches_intent(intent, expected))
                self.assertFalse(query_matches_intent(intent, unrelated))

    def test_entity_renaming_does_not_change_intent_shape(self):
        first = parse_question_intent("Is Abebe consuming higher carbohydrates?")
        second = parse_question_intent("Is Selam consuming higher carbohydrates?")

        self.assertEqual(first.mode, second.mode)
        self.assertEqual(first.terms - set(first.entities), second.terms - set(second.entities))

    def test_property_value_can_carry_question_semantics(self):
        intent = parse_question_intent("Is Tom white?")

        self.assertTrue(
            query_matches_intent(intent, "(: $prf (Has tom color white) $tv)")
        )

    def test_status_predicate_does_not_answer_plain_state_question(self):
        intent = parse_question_intent("Is Abebe currently obese?")

        self.assertFalse(
            query_matches_intent(
                intent,
                "(: $prf (ClinicallyClassifiedAsObese abebe) $tv)",
            )
        )
        self.assertTrue(
            query_matches_intent(intent, "(: $prf (Obese abebe) $tv)")
        )

    def test_more_specific_risk_candidate_scores_higher(self):
        intent = parse_question_intent("Is Abebe at elevated risk of diabetes?")

        self.assertGreater(
            query_intent_score(
                intent,
                "(: $prf (AtElevatedDiabetesRisk abebe) $tv)",
            ),
            query_intent_score(
                intent,
                "(: $prf (PlaysARoleInDiabetesRisk abebe) $tv)",
            ),
        )


if __name__ == "__main__":
    unittest.main()
