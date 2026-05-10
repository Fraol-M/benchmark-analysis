import asyncio
import sys
import types
import unittest


pettachainer_pkg = types.ModuleType("pettachainer")
pettachainer_mod = types.ModuleType("pettachainer.pettachainer")


class DummyPeTTaChainer:
    def add_atom(self, _atom):
        return None

    def query(self, _query):
        return []


pettachainer_mod.PeTTaChainer = DummyPeTTaChainer
sys.modules.setdefault("pettachainer", pettachainer_pkg)
sys.modules.setdefault("pettachainer.pettachainer", pettachainer_mod)

from core.parser import ParseResult
from core.service import PLNRAGService


class FakeParser:
    def __init__(self, queries):
        self._queries = queries

    def parse_query(self, _question, _context):
        return ParseResult(queries=list(self._queries))


class FakeVectorStore:
    def __init__(self, matches):
        self._matches = matches
        self.search_called = False

    def retrieve_context(self, _text, top_k):
        return [], []

    def search(self, _text, top_k, min_score=None):
        self.search_called = True
        return list(self._matches), []


class ExplodingVectorStore(FakeVectorStore):
    def search(self, _text, top_k, min_score=None):
        raise AssertionError("alignment search should be disabled")


class FakeReasoner:
    def __init__(self, proof_by_query):
        self._proof_by_query = proof_by_query
        self.queries = []

    def add_statements(self, _statements):
        return []

    def query(self, query):
        self.queries.append(query)
        return self._proof_by_query.get(query, [])


class FakeAnswerGenerator:
    def generate(self, _question, proof_traces):
        return "proved" if proof_traces else "unknown"


def make_service(*, matches, parser_queries, proofs, alignment_enabled=True):
    svc = PLNRAGService.__new__(PLNRAGService)
    svc._parser = FakeParser(parser_queries)
    svc._reasoner = FakeReasoner(proofs)
    svc._vector_store = FakeVectorStore(matches) if alignment_enabled else ExplodingVectorStore([])
    svc._answer_gen = FakeAnswerGenerator()
    svc._context_top_k = 10
    svc._query_fallback_enabled = True
    svc._query_alignment_enabled = alignment_enabled
    svc._query_alignment_top_k = 8
    svc._query_alignment_min_score = 0.55
    svc._enrich_context = lambda context: context
    return svc


class ServiceQueryAlignmentTests(unittest.TestCase):
    def test_qdrant_aligned_query_runs_before_parser_query(self):
        aligned = "(: $prf (Smart abebe) $tv)"
        parser = "(: $prf (Excellent abebe) $tv)"
        svc = make_service(
            matches=[{"score": 0.9, "nl": "Abebe is smart.", "query_targets": ["(Smart abebe)"]}],
            parser_queries=[parser],
            proofs={aligned: ["proof"]},
        )

        response = asyncio.run(svc.query("Is Abebe genius?"))

        self.assertEqual(response.executed_query, aligned)
        self.assertEqual(response.query_source, "qdrant_alignment")
        self.assertTrue(response.alignment_used)
        self.assertEqual(svc._reasoner.queries, [aligned, parser])

    def test_parser_query_runs_after_failed_alignment_query(self):
        aligned = "(: $prf (Smart abebe) $tv)"
        parser = "(: $prf (Excellent abebe) $tv)"
        svc = make_service(
            matches=[{"score": 0.9, "nl": "Abebe is smart.", "query_targets": ["(Smart abebe)"]}],
            parser_queries=[parser],
            proofs={parser: ["proof"]},
        )

        response = asyncio.run(svc.query("Is Abebe excellent?"))

        self.assertEqual(response.executed_query, parser)
        self.assertEqual(response.query_source, "parser")
        self.assertFalse(response.alignment_used)
        self.assertEqual(svc._reasoner.queries, [aligned, parser])

    def test_disabled_alignment_preserves_parser_only_behavior(self):
        parser = "(: $prf (Excellent abebe) $tv)"
        svc = make_service(
            matches=[],
            parser_queries=[parser],
            proofs={parser: ["proof"]},
            alignment_enabled=False,
        )

        response = asyncio.run(svc.query("Is Abebe excellent?"))

        self.assertEqual(response.executed_query, parser)
        self.assertEqual(response.query_source, "parser")
        self.assertFalse(response.alignment_used)
        self.assertEqual(svc._reasoner.queries, [parser])


if __name__ == "__main__":
    unittest.main()
