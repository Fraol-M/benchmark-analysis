"""Quick smoke test for the benchmark framework."""
import sys
sys.path.insert(0, ".")

# Test 1: Load cases
from compare import load_cases
cases = load_cases()
print(f"1. Loaded {len(cases)} cases: OK")

# Test 2: Filter cases
filtered = load_cases(filter_ids=["fish-smart-yesno", "frog-green-rule"])
print(f"2. Filtered to {len(filtered)} cases: OK")

# Test 3: Scoring
from metrics.scoring import score_answer, score_extraction
s = score_answer("t1", "Yes, Kebede is smart", "Yes", ["yes", "smart"])
assert s.correct
print(f"3. Answer scoring: OK (correct={s.correct}, precision={s.keyword_precision:.2f})")

e = score_extraction("t1", ["(isa Sam frog)", "(croaks Sam)"])
assert e.atom_count == 2
print(f"4. Extraction scoring: OK (atoms={e.atom_count}, heads={e.unique_heads})")

# Test 4: Report generation (with mock data)
from compare import generate_report
from metrics.scoring import AggregateMetrics
mock_agg = AggregateMetrics(
    backend="mock",
    total_cases=2,
    correct_count=1,
    accuracy=0.5,
    avg_keyword_precision=0.5,
    avg_atom_count=5.0,
    avg_ingest_latency_s=1.0,
    avg_query_latency_s=0.1,
    total_errors=0,
    cases_with_rules=1,
    cases_with_facts=2,
    category_accuracy={"fact-extraction": 1.0, "single-hop-rule": 0.0},
    hop_accuracy={0: 1.0, 1: 0.0},
)
mock_results = [
    {
        "case_id": "test-1",
        "category": "fact-extraction",
        "hop_depth": 0,
        "answer": "Sam is a frog",
        "atom_count": 3,
        "answer_score": {"correct": True},
        "ingest_error": None,
        "query_error": None,
    },
]
report = generate_report(
    {"mock": mock_results},
    {"mock": mock_agg},
    "report/smoke_test.md",
)
print(f"5. Report generation: OK ({len(report)} chars)")
print(f"\nAll smoke tests passed!")
