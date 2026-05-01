# Benchmark: Demo vs PLN-RAG

Compares two NL → symbolic reasoning pipelines on the same inputs.

## Quick Start

```bash
# Run Demo backend only (requires langextract + hyperon + Gemini API key)
python benchmarks/compare.py --backend demo

# Run PLN-RAG backend only (requires PeTTaChainer + Qdrant + Ollama + OpenAI key)
python benchmarks/compare.py --backend plnrag

# Run both and compare
python benchmarks/compare.py --backend both

# Run specific cases only
python benchmarks/compare.py --backend demo --cases fish-smart-yesno,frog-green-rule

# Custom output path
python benchmarks/compare.py --backend both --output benchmarks/report/my_run.md
```

## Structure

```
benchmarks/
├── README.md              ← this file
├── compare.py             ← main orchestrator
├── datasets/
│   └── cases.json         ← 30 QA test cases
├── runners/
│   ├── __init__.py        ← Backend protocol + shared types
│   ├── demo_backend.py    ← LangExtract → Hyperon adapter
│   └── plnrag_backend.py  ← NL2PLN → PeTTaChainer adapter
├── metrics/
│   ├── __init__.py
│   └── scoring.py         ← answer accuracy + extraction metrics
└── report/
    ├── results.md         ← generated markdown report
    └── results.json       ← raw data (auto-generated)
```

## Test Cases (30 total)

| Category | Count | Examples |
|---|---|---|
| fact-extraction | 5 | Basic fact lookup, properties, types |
| single-hop-rule | 6 | Simba-style: "X does Y → X is Z" |
| multi-premise-rule | 2 | 2-3 premise conjunction rules |
| multi-variable-rule | 2 | Shared variables across premises |
| inheritance | 2 | Class hierarchy (1 and 2 hop) |
| transitive-chain | 2 | 2-hop and 3-hop transitive chains |
| open-question | 2 | "Who/What" questions |
| negation | 1 | Explicit negation |
| negative | 1 | No proof expected |
| paraphrase | 6 | 2 sets × 3 phrasings each |

Each case has: NL text, question, gold answer, gold keywords, category, hop depth.

## Scoring

- **Answer accuracy**: At least one gold keyword found in the system's answer
- **Keyword precision**: Fraction of gold keywords matched
- **Extraction stats**: Atom count, predicate diversity, rules vs facts
- **Breakdowns**: By category, by reasoning depth (hop count)

## Prerequisites

### Demo backend
- `pip install hyperon langextract beautifulsoup4`
- `LANGEXTRACT_API_KEY` set in environment or `.env`

### PLN-RAG backend
- Docker + Qdrant + Ollama running on host
- PeTTaChainer + NL2PLN installed
- `OPENAI_API_KEY` set in environment or `.env`
