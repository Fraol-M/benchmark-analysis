# Benchmark Analysis Plan — demo vs PLN-RAG

Compares this demo (langextract → MeTTa AtomSpace via Hyperon) against `../PLN-RAG/` (NL2PLN → PeTTaChainer/PLN). Goal: measure where each wins, where each loses, on the same NL input.

---

## Core question to settle first

What are we trying to learn? Three different benchmarks answer three different questions:

1. **"Which one extracts atoms more accurately?"** → extraction-only benchmark.
2. **"Which one reasons better given the same facts?"** → reasoning-only benchmark.
3. **"Which one answers a user's question better end-to-end?"** → full pipeline benchmark.

Skip "main vs main" string comparison. Output formats differ:

- demo: bare MeTTa, e.g. `(isa Sam frog)`, `(= (hungry $x) (match &self ...))`.
- PLN-RAG: typed PLN, e.g. `(: kebede_eats_fish (Eats kebede fish) (STV 1.0 1.0))` w/ `(Implication (Premises ...) (Conclusions ...))`.

Atom-level string diff is meaningless. Compare *behavior*, not text.

Recommend: do all three layers, report separately. Tells where one wins, not just which.

---

## Entry points to call (per layer)

| Layer | Demo entry | PLN-RAG entry |
|---|---|---|
| Extraction only | `run_extraction(text)` → `result["atoms"]` | `get_parser().parse(text, context=[])` → `ParseResult.statements` |
| Reasoning only | Build `MeTTa()`, load gold atoms, `.run(query)` | `Reasoner.add_statements(gold)` + `.query(pln_query)` |
| End-to-end | `run_extraction` + `metta.run(query)` | HTTP `POST /ingest` + `/query` (or in-process `PLNRAGService.ingest_batch` + `.query`) |

End-to-end **does** hit main entry points. Other layers bypass them on purpose to isolate the component under test.

---

## Benchmark dimensions

### A. Extraction quality

- Precision/recall vs gold atom set.
- Atom redundancy: # distinct atoms emitted per unique semantic fact (paraphrase test).
- Symbol consistency rate: same concept across 3 paraphrases → same symbol Y/N.
- Failure rate: parse errors, malformed atoms, free-var facts.

### B. Reasoning quality (given identical KB)

- Yes/no question accuracy.
- Multi-hop chain depth handled (1, 2, 3, 4+ steps).
- Open-ended question recall (`Who eats flies?` returns all witnesses).
- Negation correctness.

### C. End-to-end

- Answer accuracy on (text, question, gold_answer) triples.
- Latency P50/P95.
- $ per query (tokens × price).

### D. Robustness

- Paraphrase: 3 phrasings of same fact → consistent atoms?
- Hyphenated/numeric IDs (`Object-77`, `rack-01`).
- Long doc (10+ chunks): predicate drift across chunks?
- Adversarial: contradictory statements, scope ambiguity.

---

## Datasets

1. **simba_all.json 7 demos** (PLN-RAG built-in gold) — already have NL + statements + queries. Use for A+B+C. Smallest free dataset.
2. **demo `DEFAULT_TEXT`** (Sam frog) — known good for demo. Hand-craft PLN gold.
3. **Hand-crafted 30-50 QA triples** covering matrix above. Biggest effort, biggest signal.
4. **External**: bAbI deductive subset, ProofWriter, RuleTaker. Public, gold reasoning chains. Best for credibility.

Start with simba 7 + 20 hand-crafted. Add external later.

---

## Metric details

- **Extraction P/R**: needs gold atoms in each system's format. Two paths:
  - (a) Maintain per-system gold (twice the work, less ambiguous).
  - (b) Define abstract-atom shape `(predicate, args[])` w/ canonicalized symbols, translate both systems into it, compare.
  - v1 → use (a).
- **Equivalence judging for free-form answers**: LLM-as-judge (small Gemini call) OR keyword-overlap on entity tokens.
- **Determinism**: set `temperature=0`. Run n=3 per case, report mean + variance. LLMs still drift.

---

## Proposed file layout

```
benchmarks/
├── datasets/
│   ├── simba_cases.jsonl        # 7 PLN-RAG demos
│   ├── manual_qa.jsonl          # 30 hand-crafted (text, question, gold)
│   └── babi_subset.jsonl        # later
├── runners/
│   ├── demo_backend.py          # adapter: ingest/query → demo
│   └── plnrag_backend.py        # adapter: ingest/query → PLN-RAG HTTP
├── metrics/
│   ├── extraction.py            # P/R, redundancy, consistency
│   ├── reasoning.py             # answer accuracy, hop depth
│   └── system.py                # latency, cost, tokens
├── compare.py                   # orchestrator
└── report/
    ├── results.json
    └── report.md                # tables + commentary
```

---

## Common backend interface

```python
class Backend(Protocol):
    name: str
    def ingest(self, text: str) -> IngestResult:  # atoms, latency, cost
        ...
    def query(self, question: str) -> QueryResult:  # answer, raw_atoms, latency, cost
        ...
    def reset(self) -> None: ...
```

Two impls. Same loop iterates both. Apples-to-apples within each metric.

---

## Implementation phases

### Phase 1 — Skeleton (1-2 days)

- Backend adapters (HTTP for PLN-RAG, in-process for demo).
- Latency + cost instrumentation.
- Run 7 simba cases through both. Just collect results, no scoring yet.

### Phase 2 — Metrics (2-3 days)

- Hand-craft 30 QA triples (NL text + question + gold answer).
- Implement answer-judge (LLM or rule-based).
- Wire extraction P/R against per-system gold.

### Phase 3 — Layered runs (1-2 days)

- Add extraction-only and reasoning-only modes.
- Run all three layers × all datasets.

### Phase 4 — Report (1 day)

- Tables: accuracy, latency, cost per system per dataset.
- Plots: hop-depth vs accuracy curve, paraphrase consistency.
- Failure-mode breakdown (where each loses).

Total: ~1 week.

---

## Caveats / fairness traps

- **LLM mismatch.** Demo = Gemini 2.5 Flash, PLN-RAG default = OpenAI gpt-4o-mini. Different cost/quality baseline. Either pin same LLM (PLN-RAG also accepts OpenAI; demo accepts OpenAI/Ollama via `model_id`) or explicitly disclaim and report both.
- **Setup cost.** PLN-RAG needs Docker + Qdrant + Ollama up before any test runs. Build a `docker compose up && wait_healthy` step into runner.
- **Cold cache.** First Gemini/OpenAI call slow. Warm w/ throwaway request before timing.
- **Non-determinism.** `temperature=0` + n=3 averaging.
- **PLN-RAG advantage.** Stateful: each ingest sees prior atoms. Demo stateless. For end-to-end fairness, reset PLN-RAG between cases or be explicit that PLN-RAG benefits from accumulated context.
- **Gold atom format drift.** Authoring per-system gold is expensive. Consider letting one human write NL + gold answer only; both systems get same NL, answer judged on QA, not atom string.

---

## Recommended scope for v1

- **Layer**: end-to-end (C) + extraction (A). Skip reasoning-only (B) first round; needs pre-built KBs in both formats.
- **Datasets**: simba 7 + manual 20 = 27 cases.
- **Metrics**: answer accuracy + latency + cost. Save P/R for v2.
- **Backends**: both at default config + LLM-pinned variant.

---

## Open questions

1. PLN-RAG in Docker for benchmark, or in-process via `PLNRAGService(...)`? In-process = faster, fewer moving parts.
2. Same-LLM run included, or only default-config?
3. External benchmarks (bAbI, ProofWriter) v1 or v2?

Confirm answers before implementing runner skeleton.
