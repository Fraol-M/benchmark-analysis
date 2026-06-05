from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.pln_postprocessor import PLNPostprocessor


HIGH_CARB_FACT = (
    "(: abebe_carbs_fact "
    "(ConsumesHigherCarbohydrates abebe) "
    "(STV 1.0 1.0))"
)
OBESITY_RULE = (
    "(: obesity_rule "
    "(Implication "
    "(Premises "
    "(HasHighCarbohydrateIntake $x) "
    "(TotalCalorieIntakeExceedsEnergyExpenditure $x)) "
    "(Conclusions (LeadsToObesity $x))) "
    "(STV 1.0 1.0))"
)
DIABETES_RISK_RULE = (
    "(: diabetes_rule "
    "(Implication "
    "(Premises (AtElevatedDiabetesRisk $x)) "
    "(Conclusions (NeedsMedicalMonitoring $x))) "
    "(STV 1.0 1.0))"
)
SUGAR_INTAKE_RULE = (
    "(: sugar_rule "
    "(Implication "
    "(Premises (HasConsistentlyHighSugarIntake $x)) "
    "(Conclusions (AtElevatedDiabetesRisk $x))) "
    "(STV 1.0 1.0))"
)
NO_SUGAR_FACT = (
    "(: abebe_no_sugar "
    "(Not (HasConsistentlyHighSugarIntake abebe)) "
    "(STV 1.0 1.0))"
)


class SchemaAlignmentTests(unittest.TestCase):
    def test_same_chunk_bridge_connects_high_carbs_to_high_carb_intake(self):
        result = PLNPostprocessor().process(
            text=(
                "Abebe consumes higher carbohydrates. High carbohydrate intake "
                "leads to obesity when calories exceed expenditure."
            ),
            statements=[HIGH_CARB_FACT, OBESITY_RULE],
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertIn(
            "(: consumes_higher_carbohydrates_to_has_high_carbohydrate_intake_bridge "
            "(Implication (Premises (ConsumesHigherCarbohydrates $x)) "
            "(Conclusions (HasHighCarbohydrateIntake $x))) (STV 0.9 0.8))",
            result.statements,
        )
        self.assertTrue(
            any(
                decision.get("action") == "bridge_added"
                and decision.get("source") == "(ConsumesHigherCarbohydrates abebe)"
                and decision.get("target") == "(HasHighCarbohydrateIntake $x)"
                for decision in result.alignment_decisions
            )
        )

    def test_context_rule_can_supply_bridge_target(self):
        result = PLNPostprocessor().process(
            text="Abebe consumes higher carbohydrates.",
            statements=[HIGH_CARB_FACT],
            queries=[],
            context=[OBESITY_RULE],
            plan_queries=False,
        )

        self.assertTrue(
            any(
                decision.get("source_origin") == "current"
                and decision.get("target_origin") == "context"
                and "HasHighCarbohydrateIntake" in decision.get("statement", "")
                for decision in result.alignment_decisions
            )
        )

    def test_unsafe_diabetes_risk_bridge_is_rejected(self):
        result = PLNPostprocessor().process(
            text="Abebe consumes higher carbohydrates.",
            statements=[HIGH_CARB_FACT, DIABETES_RISK_RULE],
            queries=[],
            context=[],
            plan_queries=False,
        )

        joined = "\n".join(result.statements)
        self.assertNotIn("to_at_elevated_diabetes_risk_bridge", joined)
        self.assertFalse(
            any(
                decision.get("target") == "(AtElevatedDiabetesRisk $x)"
                for decision in result.alignment_decisions
            )
        )

    def test_high_carb_does_not_bridge_to_high_sugar_intake(self):
        result = PLNPostprocessor().process(
            text=(
                "Abebe consumes higher carbohydrates, but he does not consume "
                "many sugary foods."
            ),
            statements=[HIGH_CARB_FACT, SUGAR_INTAKE_RULE, NO_SUGAR_FACT],
            queries=[],
            context=[],
            plan_queries=False,
        )

        joined = "\n".join(result.statements)
        self.assertNotIn("has_consistently_high_sugar_intake_bridge", joined)
        self.assertFalse(
            any(
                decision.get("target") == "(HasConsistentlyHighSugarIntake $x)"
                and decision.get("action") == "bridge_added"
                for decision in result.alignment_decisions
            )
        )

    def test_explicit_negation_blocks_equivalent_target_bridge(self):
        no_high_carb_intake = (
            "(: abebe_no_high_carb_intake "
            "(Not (HasHighCarbohydrateIntake abebe)) "
            "(STV 1.0 1.0))"
        )
        result = PLNPostprocessor().process(
            text="Abebe consumes higher carbohydrates.",
            statements=[HIGH_CARB_FACT, OBESITY_RULE, no_high_carb_intake],
            queries=[],
            context=[],
            plan_queries=False,
        )

        joined = "\n".join(result.statements)
        self.assertNotIn("to_has_high_carbohydrate_intake_bridge", joined)
        self.assertTrue(
            any(
                decision.get("action") == "bridge_rejected"
                and decision.get("reason") == "explicit_negation_conflict"
                and decision.get("target") == "(HasHighCarbohydrateIntake $x)"
                for decision in result.alignment_decisions
            )
        )

    def test_obese_type_assertion_becomes_property_predicate(self):
        result = PLNPostprocessor().process(
            text="Obese people have higher diabetes risk.",
            statements=[
                "(: obesity_rule "
                "(Implication "
                "(Premises (IsA $x obese)) "
                "(Conclusions (IncreasesRiskOfType2Diabetes $x))) "
                "(STV 1.0 1.0))",
                "(: abebe_is_obese_neg (Not (IsObese abebe)) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        joined = "\n".join(result.statements)
        self.assertIn("(Premises (IsObese $x))", joined)
        self.assertNotIn("(IsA $x obese)", joined)

    def test_property_type_normalization_is_schema_driven(self):
        result = PLNPostprocessor().process(
            text="Massive stars can become supernovas. Betelgeuse is massive.",
            statements=[
                "(: supernova_rule "
                "(Implication "
                "(Premises (IsA $x massive)) "
                "(Conclusions (CanBecomeSupernova $x))) "
                "(STV 1.0 1.0))",
                "(: betelgeuse_massive_fact (IsMassive betelgeuse) (STV 1.0 1.0))",
            ],
            queries=[],
            context=[],
            plan_queries=False,
        )

        joined = "\n".join(result.statements)
        self.assertIn("(Premises (IsMassive $x))", joined)
        self.assertNotIn("(IsA $x massive)", joined)


if __name__ == "__main__":
    unittest.main()
