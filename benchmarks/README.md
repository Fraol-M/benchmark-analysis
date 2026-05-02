# Benchmark: NL → AtomSpace Extraction

Measures how well each pipeline converts natural language text into symbolic
atoms. No query or reasoning step is involved — the benchmark is purely about
extraction quality, speed, and structural richness of the output.

---

## What it measures

Given a set of natural language sentences, each pipeline must:

1. Parse the text into symbolic atoms (MeTTa / PLN format)
2. Return those atoms for scoring

The benchmark then checks:

- **Extraction accuracy** — do the expected concept keywords appear anywhere in the extracted atoms?
- **Keyword precision** — what fraction of all expected keywords were found?
- **Atom count** — how many atoms were produced?
- **Unique predicate heads** — how many distinct relation types were used?
- **Rule ratio** — what proportion of atoms are rules (implications) vs facts?
- **Atoms per sentence** — extraction density relative to input size?
- **Ingest latency** — how long did the extraction take in seconds?

---

## How it works step by step

### Step 1 — Load test cases

`compare.py` reads `datasets/cases.json`. Each case contains:

```json
{
  "id": "fish-smart-yesno",
  "category": "single-hop-rule",
  "hop_depth": 1,
  "texts": ["People who eat fish are smart.", "Kebede eats fish."],
  "expected_keywords": ["kebede", "fish", "smart"]
}
```

- `texts` — the natural language input fed to the pipeline
- `expected_keywords` — concept words that should appear somewhere in the extracted atoms
- `category` — the type of linguistic/logical structure being tested
- `hop_depth` — how many inference steps the text implies (0 = direct fact, 3 = deep chain)

### Step 2 — Reset backend state

Before each case the backend is reset so atoms from a previous case cannot
leak into the next one. For PLN-RAG this calls `DELETE /reset` on the API.
For the Demo backend it clears the in-memory MeTTa space.

### Step 3 — Ingest

The `texts` list is passed to the backend's `ingest()` method.

**Demo backend** calls `run_extraction()` from the LangExtract → Hyperon
pipeline directly in Python. LangExtract sends the text to Gemini, which
returns structured extractions. These are translated into MeTTa atoms and
loaded into a `GroundingSpaceRef`.

**PLN-RAG backend** sends a `POST /ingest` request to the PLN-RAG FastAPI
service running in Docker. Inside the container the text is chunked, embedded
via Ollama, parsed into PLN atoms by NL2PLN + DSPy (using Gemini), stored in
Qdrant, and written to an atomspace file.

### Step 4 — Collect atoms

After ingest, `get_atoms()` retrieves the full list of atom strings from the
backend. These are the raw symbolic representations that will be scored.

Example atoms the Demo might produce for `"Kebede eats fish"`:

```
(eats kebede fish)
(IsA kebede person)
(smart kebede)
```

### Step 5 — Score extraction accuracy

`score_extraction_accuracy()` checks whether each expected keyword appears
anywhere in the atom strings (case-insensitive, token-level match).

```
expected_keywords: ["kebede", "fish", "smart"]
atoms blob:        "(eats kebede fish) (smart kebede)"

kebede → found ✓
fish   → found ✓
smart  → found ✓

keyword_hits      = 3
keyword_total     = 3
keyword_precision = 1.0
correct           = True  (at least one hit)
```

A case is marked **correct** if at least one keyword is found. Precision
measures how completely the pipeline captured all expected concepts.

### Step 6 — Score extraction richness

`score_extraction()` analyses the structure of the atoms independently of
the expected keywords:

- counts total atoms
- identifies predicate heads (the first symbol inside each `(...)`)
- classifies each atom as a rule (`(= ...)` or contains `Implication`) or a fact
- computes rule ratio and atoms-per-sentence density

### Step 7 — Aggregate and report

After all cases run, `compute_aggregate()` rolls up per-case scores into
backend-level summaries broken down by:

- overall accuracy and precision
- category (fact-extraction, single-hop-rule, paragraph, etc.)
- hop depth (0 = direct fact, 1 = one inference step, 2–3 = chains)

A markdown report is written to `benchmarks/report/results.md` and raw JSON
to `benchmarks/report/results.json`.

---

## Dataset categories

| Category | Cases | What it tests |
|---|---|---|
| `fact-extraction` | 5 | Direct property and type lookups, no inference |
| `single-hop-rule` | 7 | One general rule applied to one specific fact |
| `multi-premise-rule` | 2 | Rules requiring multiple conditions simultaneously |
| `multi-variable-rule` | 2 | Rules where variables bind across several premises |
| `inheritance` | 2 | Class hierarchy (1-hop and 2-hop) |
| `transitive-chain` | 2 | Multi-hop causal chains (2-hop and 3-hop) |
| `open-question` | 2 | Multiple entities with different properties |
| `negation` | 1 | Explicit negation in text |
| `negative` | 1 | No implied conclusion — tests against hallucination |
| `paraphrase` | 6 | Same meaning expressed 3 different ways (synonym, passive, quantifier) |
| `paragraph` | 7 | Large unstructured paragraphs (medical, legal, ecology, infrastructure, etc.) |

### Hop depth

| Depth | Meaning |
|---|---|
| 0 | Direct fact — no inference needed |
| 1 | One rule application |
| 2 | Two chained inferences |
| 3 | Three chained inferences (hardest) |

A pipeline that scores well at hop 0 but drops at hop 2–3 extracts facts
well but struggles to represent multi-step logical structure.

---

## Backends

### Demo (LangExtract → Hyperon MeTTa)

- Runs fully in Python, no Docker needed
- Uses LangExtract with Gemini to extract structured relations
- Translates extractions into MeTTa atoms via a predicate mapping layer
- Requires: `LANGEXTRACT_API_KEY` in `.env`

### PLN-RAG (NL2PLN → PeTTaChainer)

- Runs as a Docker stack — no local Python dependencies needed
- Text is chunked, embedded (Ollama / nomic-embed-text), parsed by NL2PLN
  with DSPy + Gemini, stored in Qdrant, and written to a MeTTa atomspace file
- The benchmark talks to it over HTTP (`POST /ingest`, `DELETE /reset`)
- Requires: Docker Desktop running + `GEMINI_API_KEY` in `.env`

---

## Prerequisites

### Demo backend
```
pip install -r demo/requirements.txt
```
Set in root `.env`:
```
LANGEXTRACT_API_KEY=your-gemini-key
LANGEXTRACT_MODEL_ID=gemini-2.5-flash
```

### PLN-RAG backend
```
# from PLN-RAG/
docker compose up --build
```
Set in root `.env`:
```
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini/gemini-2.5-flash
```
The compose stack starts PLN-RAG, Qdrant, and Ollama. On first run it
automatically pulls the `nomic-embed-text` embedding model (~270 MB).

---

## Running

```bash
# Demo only
python benchmarks/compare.py --backend demo

# PLN-RAG only (Docker must be running)
python benchmarks/compare.py --backend plnrag

# Both side by side
python benchmarks/compare.py --backend both

# Subset of cases
python benchmarks/compare.py --backend demo --cases fish-smart-yesno,para-hospital-staff

# Custom output path
python benchmarks/compare.py --backend demo --output benchmarks/report/my_run.md
```

Verify the framework itself works without any API calls:
```bash
python benchmarks/smoke_test.py
```

---

## Output

| File | Contents |
|---|---|
| `benchmarks/report/results.md` | Human-readable report with summary tables and per-case breakdown |
| `benchmarks/report/results.json` | Raw scores and atom samples for every case |

---

## Project structure

```
benchmarks/
├── compare.py              orchestrator — loads cases, runs backends, writes report
├── smoke_test.py           fast self-test with no API calls
├── datasets/
│   └── cases.json          37 test cases across 11 categories
├── metrics/
│   └── scoring.py          accuracy + richness scoring functions
├── runners/
│   ├── __init__.py         Backend protocol + IngestResult + TimedMixin
│   ├── demo_backend.py     LangExtract → MeTTa in-process adapter
│   └── plnrag_backend.py   PLN-RAG HTTP client adapter
└── report/
    ├── results.md          generated report (git-ignored content)
    └── results.json        raw data (git-ignored content)
```
