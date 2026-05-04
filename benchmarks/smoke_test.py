"""Quick smoke test for the benchmark framework (no API calls)."""
import sys
sys.path.insert(0, ".")

from compare import load_cases
from metrics.scoring import (
    score_extraction_accuracy,
    score_extraction,
    score_reasoning,
    AggregateMetrics,
    ReasoningAggregateMetrics,
)
from compare import generate_report, generate_reasoning_report

# 1. Load cases
cases = load_cases()
short_cases = [c for c in cases if c["category"] != "paragraph"]
para_cases = [c for c in cases if c["category"] == "paragraph"]
print(f"1. Loaded {len(cases)} cases ({len(short_cases)} short, {len(para_cases)} paragraph): OK")

# 2. Filter
filtered = load_cases(filter_ids=["fish-smart-yesno", "frog-basic-facts"])
assert len(filtered) == 2
print(f"2. Filtered to {len(filtered)} cases: OK")

# 3. Extraction accuracy scoring
s = score_extraction_accuracy(
    "t1",
    ["(isa kebede human)", "(eats kebede fish)", "(smart kebede)"],
    ["kebede", "fish", "smart"],
)
assert s.correct
assert s.keyword_hits == 3
print(f"3. Accuracy scoring: hits={s.keyword_hits}/{s.keyword_total} precision={s.keyword_precision:.2f}: OK")

# 4. Extraction richness scoring
e = score_extraction(
    "t1",
    ["(isa Sam frog)", "(croaks Sam)", "(= (green $x) (and (frog $x) (croaks $x)))"],
    sentence_count=3,
)
assert e.atom_count == 3
assert e.has_rules and e.has_facts
assert e.rule_count == 1 and e.fact_count == 2
print(f"4. Richness scoring: atoms={e.atom_count} rules={e.rule_count} facts={e.fact_count} aps={e.atoms_per_sentence:.2f}: OK")

# 5. Report generation
mock_agg = AggregateMetrics(
    backend="mock",
    total_cases=3,
    correct_count=2,
    accuracy=0.667,
    avg_keyword_precision=0.75,
    avg_atom_count=4.0,
    avg_unique_heads=2.5,
    avg_rule_ratio=0.25,
    avg_atoms_per_sentence=2.0,
    avg_ingest_latency_s=1.2,
    total_errors=0,
    cases_with_rules=2,
    cases_with_facts=3,
    category_accuracy={"fact-extraction": 1.0, "paragraph": 0.5},
    category_avg_atoms={"fact-extraction": 3.0, "paragraph": 8.0},
    hop_accuracy={0: 1.0, 1: 0.5},
)
mock_results = [
    {
        "case_id": "test-1",
        "category": "fact-extraction",
        "hop_depth": 0,
        "ingest_latency_s": 1.1,
        "ingest_error": None,
        "accuracy_score": {
            "correct": True,
            "keyword_hits": 2,
            "keyword_total": 2,
            "keyword_precision": 1.0,
            "matched_keywords": ["frog", "sam"],
            "missing_keywords": [],
            "atom_sample": ["(isa sam frog)"],
        },
        "extraction_score": {
            "case_id": "test-1",
            "atom_count": 3,
            "unique_heads": 2,
            "head_list": ["isa", "eats"],
            "has_rules": False,
            "has_facts": True,
            "rule_count": 0,
            "fact_count": 3,
            "rule_ratio": 0.0,
            "atoms_per_sentence": 1.5,
            "error": None,
        },
    },
]
report = generate_report({"mock": mock_results}, {"mock": mock_agg}, "report/smoke_test.md")
assert "Extraction Accuracy" in report
print(f"5. Report generation: {len(report)} chars: OK")

# 6. Reasoning scoring
r = score_reasoning(
    "r1",
    ["(Smart kebede)"],
    expected_proof=True,
    expected_terms=["Smart", "kebede"],
)
assert r.correct
print(f"6. Reasoning scoring: proof={r.has_proof} terms={r.term_hits}/{r.term_total}: OK")

# 7. Reasoning report generation
mock_reasoning_agg = ReasoningAggregateMetrics(
    backend="mock",
    total_cases=2,
    correct_count=1,
    accuracy=0.5,
    avg_query_latency_s=0.2,
    total_errors=0,
    expected_proof_cases=1,
    expected_no_proof_cases=1,
    false_negatives=1,
    false_positives=0,
    category_accuracy={"single-hop-rule": 1.0, "transitive-chain": 0.0},
    hop_accuracy={1: 1.0, 2: 0.0},
)
mock_reasoning_results = [
    {
        "case_id": "reasoning-1",
        "category": "single-hop-rule",
        "hop_depth": 1,
        "expected_proof": True,
        "query_latency_s": 0.2,
        "query_error": None,
        "reasoning_score": {
            "correct": True,
            "has_proof": True,
        },
    }
]
reasoning_report = generate_reasoning_report(
    {"mock": mock_reasoning_results},
    {"mock": mock_reasoning_agg},
    "report/smoke_reasoning.md",
)
assert "Reasoning" in reasoning_report
print(f"7. Reasoning report generation: {len(reasoning_report)} chars: OK")

print("\nAll smoke tests passed!")
