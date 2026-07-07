from __future__ import annotations

import unittest
from unittest.mock import patch

from core.pln.postprocessor import PLNPostprocessor
from core.pln.predicate_mapping import (
    LLMPredicateRelationClassifier,
    PredicateCard,
    PredicateMappingEngine,
    RelationProposal,
    ValidatedMapping,
)
from core.pln.predicate_registry import PredicateRegistry
from core.pln.schema_alignment import PLNSchemaAligner
from core.extraction.langextract_examples import default_examples_path
from storage.vector_store import VectorStore


class FakeClassifier:
    def __init__(self, relations):
        self.relations = relations
        self.calls = []

    def classify(self, source, target):
        self.calls.append((source.predicate, target.predicate))
        relation, confidence = self.relations.get(
            (source.predicate, target.predicate),
            ("unrelated", 0.99),
        )
        return RelationProposal(
            relation=relation,
            confidence=confidence,
            reason="fake_classifier_result",
            classifier="fake",
            argument_mapping=list(range(source.arity)),
        )


class FakePredicateCardStore:
    def __init__(self):
        self.cards = []

    def upsert_predicate_cards(self, cards):
        self.cards = list(cards)

    def search_predicate_cards(
        self,
        search_text,
        *,
        arity,
        top_k,
        min_score,
    ):
        if "Orbits" not in search_text:
            return []
        return [
            {
                "score": 0.94,
                "payload": {
                    "kind": "predicate_card",
                    "key": "RevolvesAround/2",
                    "predicate": "RevolvesAround",
                    "arity": 2,
                    "argument_types": ["planet", "star"],
                    "definition": "A planet revolves around a star.",
                    "labels": ["revolves around"],
                    "examples": ["A planet revolves around a star."],
                    "source_atoms": ["(RevolvesAround $x $star)"],
                    "roles": ["premise"],
                    "origin": "qdrant",
                },
            }
        ]


class FakeResponse:
    def __init__(self, payload=None, status_code=200):
        self.payload = payload or {}
        self.status_code = status_code

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise AssertionError(f"unexpected HTTP status {self.status_code}")


class FakeQdrantClient:
    def __init__(self):
        self.put_calls = []
        self.post_calls = []

    def get(self, url):
        return FakeResponse({"result": {"points_count": 0}})

    def put(self, url, json):
        self.put_calls.append((url, json))
        return FakeResponse()

    def post(self, url, json):
        self.post_calls.append((url, json))
        return FakeResponse(
            {
                "result": [
                    {
                        "id": "card-id",
                        "score": 0.91,
                        "payload": {
                            "kind": "predicate_card",
                            "key": "RevolvesAround/2",
                            "predicate": "RevolvesAround",
                            "arity": 2,
                            "argument_types": ["planet", "star"],
                            "roles": ["premise"],
                        },
                    }
                ]
            }
        )


class FakeBatchEmbeddingClient:
    def __init__(self):
        self.post_calls = []

    def post(self, url, json):
        self.post_calls.append((url, json))
        inputs = json.get("input", [])
        return FakeResponse(
            {"embeddings": [[float(index), 0.5] for index, _ in enumerate(inputs)]}
        )


class RetryClassifier:
    def __init__(self):
        self.calls = 0

    def classify_batch(self, pairs, timeout_seconds=None):
        self.calls += 1
        if self.calls == 1:
            return [None for _ in pairs]
        return [
            RelationProposal(
                relation="source_implies_target",
                confidence=0.99,
                reason="retry_succeeded",
                classifier="fake",
                argument_mapping=[0],
            )
            for _ in pairs
        ]


def registry_with(relations, *, store=None, threshold=0.82):
    aligner = PLNSchemaAligner(PLNPostprocessor.STRUCTURAL_HEADS)
    classifier = FakeClassifier(relations)
    engine = PredicateMappingEngine(
        classifier=classifier,
        proof_threshold=threshold,
        retrieval_min_score=0.62,
        retrieval_top_k=8,
        max_candidates=16,
    )
    registry = PredicateRegistry(
        schema_aligner=aligner,
        mapping_engine=engine,
        autosave=False,
    )
    registry.set_card_store(store)
    return registry, classifier


class DynamicPredicateMappingTests(unittest.TestCase):
    def test_default_langextract_examples_path_survives_core_refactor(self):
        self.assertTrue(default_examples_path().exists())

    def test_carbohydrate_mapping_creates_validated_bridge(self):
        registry, _ = registry_with(
            {
                (
                    "ConsumesHigherCarbohydrates",
                    "HasHighCarbohydrateIntake",
                ): ("source_implies_target", 0.91)
            }
        )
        statements = [
            "(: carb_rule (Implication (Premises (IsA $x person) "
            "(EatsFrequently $x pasta)) (Conclusions "
            "(ConsumesHigherCarbohydrates $x))) (STV 1.0 1.0))",
            "(: obesity_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x) "
            "(TotalCalorieIntakeExceedsEnergyExpenditure $x)) "
            "(Conclusions (LeadsToObesity $x))) (STV 1.0 1.0))",
        ]

        _, decisions = registry.align_statements(
            statements=statements,
            context=[],
            source_text="People who eat pasta consume more carbohydrates.",
        )
        bridges, bridge_decisions = registry.build_validated_bridges(
            statements,
            [],
        )

        self.assertTrue(
            any(item.get("action") == "mapping_approved" for item in decisions)
        )
        self.assertEqual(1, len(bridges))
        self.assertIn(
            "(Premises (ConsumesHigherCarbohydrates $x))",
            bridges[0],
        )
        self.assertIn(
            "(Conclusions (HasHighCarbohydrateIntake $x))",
            bridges[0],
        )
        self.assertTrue(bridge_decisions[0]["proof_safe"])

    def test_postprocessor_emits_only_gate_approved_bridge(self):
        registry, _ = registry_with(
            {
                (
                    "ConsumesHigherCarbohydrates",
                    "HasHighCarbohydrateIntake",
                ): ("source_implies_target", 0.91)
            }
        )
        processor = PLNPostprocessor(predicate_registry=registry)
        statements = [
            "(: carb_rule (Implication (Premises (IsA $x person) "
            "(EatsFrequently $x pasta)) (Conclusions "
            "(ConsumesHigherCarbohydrates $x))) (STV 1.0 1.0))",
            "(: obesity_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x) "
            "(TotalCalorieIntakeExceedsEnergyExpenditure $x)) "
            "(Conclusions (LeadsToObesity $x))) (STV 1.0 1.0))",
        ]

        result = processor.process(
            text="People who eat pasta consume more carbohydrates.",
            statements=statements,
            queries=[],
            context=[],
            plan_queries=False,
        )

        self.assertTrue(
            any(
                "ConsumesHigherCarbohydrates $x" in statement
                and "HasHighCarbohydrateIntake $x" in statement
                and "_bridge" in statement
                for statement in result.statements
            )
        )
        self.assertTrue(
            any(
                item.get("action") == "bridge_added"
                and item.get("proof_safe") is True
                for item in result.alignment_decisions
            )
        )

    def test_qdrant_candidate_handles_unseen_astronomy_vocabulary(self):
        store = FakePredicateCardStore()
        registry, classifier = registry_with(
            {
                ("Orbits", "RevolvesAround"): (
                    "source_implies_target",
                    0.93,
                )
            },
            store=store,
        )
        statements = [
            "(: orbit_rule (Implication (Premises (IsA $x planet) "
            "(IsA $star star)) (Conclusions (Orbits $x $star))) "
            "(STV 1.0 1.0))",
            "(: stable_orbit_rule (Implication (Premises "
            "(RevolvesAround $x $star)) (Conclusions "
            "(HasStableOrbit $x))) (STV 1.0 1.0))",
        ]

        registry.align_statements(
            statements=statements,
            context=[],
            source_text="A planet orbits a star and revolves around it.",
        )
        bridges, _ = registry.build_validated_bridges(statements, [])

        self.assertIn(("Orbits", "RevolvesAround"), classifier.calls)
        self.assertTrue(store.cards)
        self.assertTrue(
            any(
                "(Premises (Orbits $x $y))" in bridge
                and "(Conclusions (RevolvesAround $x $y))" in bridge
                for bridge in bridges
            )
        )

    def test_related_mapping_is_recorded_but_never_bridged(self):
        registry, _ = registry_with(
            {
                (
                    "ConsumesHigherCarbohydrates",
                    "HasConsistentlyHighSugarIntake",
                ): ("related", 0.98)
            }
        )
        statements = [
            "(: carb_rule (Implication (Premises (IsA $x person)) "
            "(Conclusions (ConsumesHigherCarbohydrates $x))) "
            "(STV 1.0 1.0))",
            "(: sugar_rule (Implication (Premises "
            "(HasConsistentlyHighSugarIntake $x)) (Conclusions "
            "(HasSugarRisk $x))) (STV 1.0 1.0))",
        ]

        _, decisions = registry.align_statements(
            statements=statements,
            context=[],
            source_text="Carbohydrates and sugar are related topics.",
        )
        bridges, _ = registry.build_validated_bridges(statements, [])

        self.assertFalse(bridges)
        self.assertTrue(
            any(
                item.get("action") == "mapping_recorded_not_proof_safe"
                for item in decisions
            )
        )

    def test_validation_gate_rejects_argument_type_mismatch(self):
        registry, classifier = registry_with(
            {
                ("Treats", "Orbits"): ("source_implies_target", 0.99)
            }
        )
        source = PredicateCard(
            predicate="Treats",
            arity=2,
            argument_types=["doctor", "patient"],
            roles=["conclusion"],
        )
        target = PredicateCard(
            predicate="Orbits",
            arity=2,
            argument_types=["planet", "star"],
            roles=["premise"],
        )
        proposal = classifier.classify(source, target)
        validated = registry.mapping_engine.validate(
            source,
            target,
            proposal,
            retrieval_score=0.95,
            candidate_source="test",
        )

        self.assertFalse(validated.proof_safe)
        self.assertIn("argument_type_mismatch", validated.reason)

    def test_validation_gate_rejects_reordered_arguments(self):
        registry, _ = registry_with({})
        source = PredicateCard(
            predicate="Owns",
            arity=2,
            argument_types=["person", "object"],
            roles=["conclusion"],
        )
        target = PredicateCard(
            predicate="Possesses",
            arity=2,
            argument_types=["person", "object"],
            roles=["premise"],
        )
        proposal = RelationProposal(
            relation="source_implies_target",
            confidence=0.99,
            reason="arguments_were_swapped",
            classifier="fake",
            argument_mapping=[1, 0],
        )

        validated = registry.mapping_engine.validate(
            source,
            target,
            proposal,
            retrieval_score=0.95,
            candidate_source="test",
        )

        self.assertFalse(validated.proof_safe)
        self.assertIn("argument_order_not_identity", validated.reason)

    def test_vector_store_uses_separate_predicate_collection(self):
        store = VectorStore()
        client = FakeQdrantClient()
        store._client = client
        store.embed = lambda text: [0.1, 0.2, 0.3]
        card = PredicateCard(
            predicate="Orbits",
            arity=2,
            argument_types=["planet", "star"],
            definition="A planet orbits a star.",
            roles=["conclusion"],
        ).to_payload()

        store.upsert_predicate_cards([card])
        matches = store.search_predicate_cards(
            "planet orbit star",
            arity=2,
            top_k=8,
            min_score=0.62,
        )

        self.assertTrue(client.put_calls)
        self.assertIn(store._predicate_collection, client.put_calls[0][0])
        self.assertEqual("RevolvesAround", matches[0]["payload"]["predicate"])
        search_payload = client.post_calls[0][1]
        self.assertEqual(2, search_payload["filter"]["must"][1]["match"]["value"])

    def test_vector_store_batches_and_caches_predicate_embeddings(self):
        store = VectorStore()
        client = FakeBatchEmbeddingClient()
        store._client = client

        first = store.embed_many(["predicate one", "predicate two"])
        second = store.embed_many(["predicate one", "predicate two"])

        self.assertEqual(first, second)
        self.assertEqual(1, len(client.post_calls))
        self.assertTrue(client.post_calls[0][0].endswith("/api/embed"))

    def test_failed_candidate_is_retried_on_next_discover_pass(self):
        aligner = PLNSchemaAligner(PLNPostprocessor.STRUCTURAL_HEADS)
        classifier = RetryClassifier()
        engine = PredicateMappingEngine(
            classifier=classifier,
            proof_threshold=0.82,
            retrieval_min_score=0.62,
            retrieval_top_k=8,
            max_candidates=6,
        )

        source_card = PredicateCard(
            predicate="ConsumesHigherCarbohydrates",
            arity=1,
            argument_types=["person"],
            roles=["conclusion"],
        )
        target_card = PredicateCard(
            predicate="HasHighCarbohydrateIntake",
            arity=1,
            argument_types=["person"],
            roles=["premise"],
        )

        first_mappings, first_decisions = engine.discover(
            current_cards=[source_card],
            known_cards=[target_card],
            card_store=None,
            known_mapping_keys=set(),
        )
        second_mappings, second_decisions = engine.discover(
            current_cards=[source_card],
            known_cards=[target_card],
            card_store=None,
            known_mapping_keys=set(),
        )

        self.assertFalse(first_mappings)
        self.assertTrue(
            any(
                item.get("action") == "mapping_candidate_unclassified"
                for item in first_decisions
            )
        )
        self.assertTrue(second_mappings)
        self.assertEqual(2, classifier.calls)
        self.assertTrue(
            any(item.proof_safe for item in second_mappings)
        )

    def test_new_proof_safe_evidence_upgrades_related_candidate(self):
        registry, _ = registry_with({})
        registry.ensure_mapping(
            source="ConsumesHigherCarbohydrates",
            target="HasHighCarbohydrateIntake",
            arity=1,
            relation="related",
            confidence=0.98,
            reason="early_weak_evidence",
            proof_safe=False,
        )
        source = PredicateCard(
            predicate="ConsumesHigherCarbohydrates",
            arity=1,
            argument_types=["person"],
            roles=["conclusion"],
        )
        target = PredicateCard(
            predicate="HasHighCarbohydrateIntake",
            arity=1,
            argument_types=["person"],
            roles=["premise"],
        )
        registry.ensure_validated_mapping(
            ValidatedMapping(
                source=source,
                target=target,
                relation="source_implies_target",
                confidence=0.86,
                reason="new_directional_evidence",
                classifier="fake",
                proof_safe=True,
                status="approved",
            )
        )

        mapping = registry.mapping_for_direction(
            "ConsumesHigherCarbohydrates",
            "HasHighCarbohydrateIntake",
            1,
        )
        self.assertIsNotNone(mapping)
        self.assertEqual("source_implies_target", mapping.relation)

    def test_explicit_negation_blocks_an_approved_bridge(self):
        registry, _ = registry_with({})
        registry.ensure_mapping(
            source="ConsumesHigherCarbohydrates",
            target="HasHighCarbohydrateIntake",
            arity=1,
            relation="source_implies_target",
            confidence=0.93,
            reason="validated_relation",
            proof_safe=True,
        )
        statements = [
            "(: carb_rule (Implication (Premises (IsA $x person)) "
            "(Conclusions (ConsumesHigherCarbohydrates $x))) "
            "(STV 1.0 1.0))",
            "(: obesity_rule (Implication (Premises "
            "(HasHighCarbohydrateIntake $x)) (Conclusions "
            "(LeadsToObesity $x))) (STV 1.0 1.0))",
            "(: abebe_high_carb_neg (Not "
            "(HasHighCarbohydrateIntake abebe)) (STV 1.0 1.0))",
        ]

        bridges, decisions = registry.build_validated_bridges(statements, [])

        self.assertFalse(bridges)
        self.assertTrue(
            any(
                item.get("reason") == "explicit_negation_conflict"
                for item in decisions
            )
        )

    def test_model_proof_safe_flag_is_ignored(self):
        classifier = LLMPredicateRelationClassifier(
            enabled=True,
            gemini_api_key=None,
            gemini_model="gemini-2.5-flash",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
        )
        classifier._call_openai = lambda prompt, timeout_seconds=None: (
            '{"relation":"related","confidence":0.99,'
            '"argument_mapping":[0],"reason":"same topic",'
            '"proof_safe":true}'
        )
        source = PredicateCard(
            predicate="ConsumesHigherCarbohydrates",
            arity=1,
            argument_types=["person"],
            roles=["conclusion"],
        )
        target = PredicateCard(
            predicate="HasConsistentlyHighSugarIntake",
            arity=1,
            argument_types=["person"],
            roles=["premise"],
        )
        proposal = classifier.classify(source, target)
        engine = PredicateMappingEngine(
            classifier=classifier,
            proof_threshold=0.82,
            retrieval_min_score=0.62,
            retrieval_top_k=8,
        )
        validated = engine.validate(
            source,
            target,
            proposal,
            retrieval_score=0.95,
            candidate_source="test",
        )

        self.assertEqual("related", proposal.relation)
        self.assertFalse(validated.proof_safe)
        self.assertIn("relation_not_proof_safe", validated.reason)

    def test_model_exact_match_is_narrowed_to_source_direction(self):
        registry, _ = registry_with({})
        source = PredicateCard(
            predicate="ConsumesHigherCarbohydrates",
            arity=1,
            argument_types=["person"],
            roles=["conclusion"],
        )
        target = PredicateCard(
            predicate="HasHighCarbohydrateIntake",
            arity=1,
            argument_types=["person"],
            roles=["premise"],
        )
        proposal = RelationProposal(
            relation="exactMatch",
            confidence=0.9,
            reason="model_considers_terms_equivalent",
            classifier="gemini:gemini-2.5-flash",
            argument_mapping=[0],
        )

        validated = registry.mapping_engine.validate(
            source,
            target,
            proposal,
            retrieval_score=0.94,
            candidate_source="test",
        )

        self.assertTrue(validated.proof_safe)
        self.assertEqual("source_implies_target", validated.relation)
        self.assertIn("narrowed_to_source_direction", validated.reason)

    def test_classifier_batches_compact_cards_in_one_request(self):
        classifier = LLMPredicateRelationClassifier(
            enabled=True,
            gemini_api_key=None,
            gemini_model="gemini-2.5-flash",
            openai_api_key="test-key",
            openai_model="gpt-4o-mini",
        )
        calls = []

        def fake_call(prompt, timeout_seconds=None):
            calls.append(prompt)
            return (
                '{"results":['
                '{"id":0,"relation":"source_implies_target",'
                '"confidence":0.91,"argument_mapping":[0],'
                '"reason":"same intake meaning"},'
                '{"id":1,"relation":"related","confidence":0.74,'
                '"argument_mapping":[0],"reason":"same risk topic"}'
                ']}'
            )

        classifier._call_openai = fake_call
        long_evidence = "evidence " * 200
        pairs = [
            (
                PredicateCard(
                    predicate="ConsumesHigherCarbohydrates",
                    arity=1,
                    argument_types=["person"],
                    examples=[long_evidence],
                    roles=["conclusion"],
                ),
                PredicateCard(
                    predicate="HasHighCarbohydrateIntake",
                    arity=1,
                    argument_types=["person"],
                    examples=[long_evidence],
                    roles=["premise"],
                ),
            ),
            (
                PredicateCard(
                    predicate="WeightIncreased",
                    arity=1,
                    argument_types=["person"],
                    examples=[long_evidence],
                    roles=["fact"],
                ),
                PredicateCard(
                    predicate="AtObesityRisk",
                    arity=1,
                    argument_types=["person"],
                    examples=[long_evidence],
                    roles=["premise"],
                ),
            ),
        ]

        proposals = classifier.classify_batch(pairs)

        self.assertEqual(1, len(calls))
        self.assertLess(len(calls[0]), 3000)
        self.assertEqual("source_implies_target", proposals[0].relation)
        self.assertEqual("related", proposals[1].relation)

    def test_rejected_mapping_is_cached_between_chunks(self):
        registry, classifier = registry_with({})
        statements = [
            "(: source_rule (Implication (Premises (IsA $x person)) "
            "(Conclusions (ConsumesHighCarbs $x))) (STV 1.0 1.0))",
            "(: target_rule (Implication (Premises (HasHighCarbs $x)) "
            "(Conclusions (HasDietRisk $x))) (STV 1.0 1.0))",
        ]

        registry.align_statements(
            statements=statements,
            context=[],
            source_text="A person consumes high carbohydrates.",
        )
        first_call_count = len(classifier.calls)
        registry.align_statements(
            statements=statements,
            context=[],
            source_text="A person consumes high carbohydrates.",
        )

        self.assertGreater(first_call_count, 0)
        self.assertEqual(first_call_count, len(classifier.calls))

    def test_mapping_budget_skips_semantic_classification(self):
        classifier = FakeClassifier({})
        engine = PredicateMappingEngine(
            classifier=classifier,
            proof_threshold=0.82,
            retrieval_min_score=0.62,
            retrieval_top_k=6,
            max_candidates=6,
            total_timeout_seconds=1,
        )
        current = PredicateCard(
            predicate="ConsumesHighCarbs",
            arity=1,
            argument_types=["person"],
            roles=["conclusion"],
        )
        known = PredicateCard(
            predicate="HasHighCarbs",
            arity=1,
            argument_types=["person"],
            roles=["premise"],
        )

        with patch(
            "core.pln.predicate_mapping.time.monotonic",
            side_effect=[0.0, 2.0],
        ):
            mappings, decisions = engine.discover(
                current_cards=[current],
                known_cards=[known],
                card_store=None,
                known_mapping_keys=set(),
            )

        self.assertFalse(mappings)
        self.assertFalse(classifier.calls)
        self.assertTrue(
            any(item.get("action") == "mapping_budget_exhausted" for item in decisions)
        )


if __name__ == "__main__":
    unittest.main()
