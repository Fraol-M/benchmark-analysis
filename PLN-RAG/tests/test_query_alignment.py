import unittest

from core.query_alignment import build_aligned_queries, extract_query_targets


class QueryAlignmentTests(unittest.TestCase):
    def test_extracts_fact_target(self):
        targets = extract_query_targets(
            ["(: sam_dog_fact (IsA sam dog) (STV 1.0 1.0))"]
        )

        self.assertEqual(targets, ["(IsA sam dog)"])

    def test_extracts_negated_fact_target(self):
        targets = extract_query_targets(
            ["(: abebe_obese_neg (Not (IsObese abebe)) (STV 1.0 1.0))"]
        )

        self.assertEqual(targets, ["(Not (IsObese abebe))"])

    def test_extracts_rule_conclusion_target(self):
        targets = extract_query_targets(
            [
                "(: diabetes_risk_rule "
                "(Implication "
                "(Premises (IsObese $x)) "
                "(Conclusions (AtElevatedDiabetesRisk $x))) "
                "(STV 1.0 1.0))"
            ]
        )

        self.assertEqual(targets, ["(AtElevatedDiabetesRisk $x)"])

    def test_ignores_malformed_statement(self):
        self.assertEqual(extract_query_targets(["(: broken (Smart abebe)"]), [])

    def test_builds_grounded_fact_query(self):
        result = build_aligned_queries(
            "Is Abebe genius?",
            [
                {
                    "score": 0.91,
                    "nl": "Abebe is smart.",
                    "query_targets": ["(Smart abebe)"],
                }
            ],
        )

        self.assertEqual(result.queries, ["(: $prf (Smart abebe) $tv)"])

    def test_builds_grounded_rule_conclusion_query(self):
        result = build_aligned_queries(
            "Is Abebe at elevated risk of diabetes?",
            [
                {
                    "score": 0.86,
                    "nl": "Obese people with another factor are at elevated diabetes risk.",
                    "query_targets": ["(AtElevatedDiabetesRisk $x)"],
                }
            ],
        )

        self.assertEqual(
            result.queries,
            ["(: $prf (AtElevatedDiabetesRisk abebe) $tv)"],
        )

    def test_builds_negated_fact_query(self):
        result = build_aligned_queries(
            "Is Abebe currently obese?",
            [
                {
                    "score": 0.88,
                    "nl": "Abebe has not been clinically classified as obese.",
                    "query_targets": ["(Not (IsObese abebe))"],
                }
            ],
        )

        self.assertEqual(
            result.queries,
            ["(: $prf (Not (IsObese abebe)) $tv)"],
        )


if __name__ == "__main__":
    unittest.main()
