# Phase 1 README: Natural Language to MeTTa/PLN Atoms

This document explains Phase 1 of the PLN-RAG pipeline: converting natural language into MeTTa/PLN atoms and storing them in the atomspace.

## Goal

Turn user-provided text into structured logical statements that the reasoner can use later.

## Main Components

- `core/service.py` (`PLNRAGService.ingest_batch`, `_ingest_single`)
- `core/chunker.py` (`Chunker.chunk`)
- `storage/vector_store.py` (`retrieve_context`, `store`)
- parser implementation (`parsers/nl2pln_parser.py`, `parsers/canonical_pln_parser.py`, or `parsers/manhin_parser.py`)
- `core/reasoner.py` (`Reasoner.add_statements`)

## End-to-End Example

Example input text:

```text
People who eat fish are smart. Kebede eats fish.
```

### Step 1: API receives ingest request

Input:

```json
{
  "texts": ["People who eat fish are smart. Kebede eats fish."]
}
```

Action:

- `/ingest` endpoint calls `PLNRAGService.ingest_batch(texts)`.

Output:

- A per-item processing result container is created (`IngestItemResult`), not final yet.

### Step 2: Chunking

Input:

- Raw text string.

Action:

- `Chunker.chunk(text)` splits long input into manageable chunks.

Output:

- A list of chunk strings, for example:

```text
[
  "People who eat fish are smart. Kebede eats fish."
]
```

### Step 3: Retrieve context for parsing

Input:

- Current chunk text.

Action:

- `VectorStore.retrieve_context(chunk, top_k)` gets similar prior items from Qdrant.
- `_enrich_context(...)` appends recent atoms from `data/atomspace/kb.metta`.

Output:

- `context`: list of existing atoms/rules (may be empty on first ingest).
- `vector`: embedding vector for this chunk.

### Step 4: Parse natural language to PLN

Input:

- Chunk text.
- Context atoms.

Action:

- Config-selected parser runs, usually `parse(chunk, context)`.
- Parser uses DSPy+LLM (for `nl2pln`/`canonical_pln`) to generate logical statements.

Output (`ParseResult.statements`, illustrative shape):

```text
[
  "(: rule1 (Implication (Premises (eat_fish $x)) (Conclusions (smart $x))) (STV 0.9 0.8))",
  "(: fact1 (eat_fish kebede) (STV 1.0 0.95))"
]
```

Notes:

- Exact atom syntax can vary by parser.
- If parser returns no statements, the chunk is skipped.

### Step 5: Add atoms to PeTTaChainer atomspace

Input:

- Parsed statement list.

Action:

- `Reasoner.add_statements(statements)`:
  - normalizes whitespace,
  - calls `PeTTaChainer.add_atom(...)`,
  - appends each successful atom to `data/atomspace/kb.metta`.

Output:

- `added`: list of statements successfully inserted.
- Persistent side effect: atomspace file now contains the new atoms.

### Step 6: Store mapping in vector DB

Input:

- Original chunk text.
- Added atom list.
- Chunk embedding vector from Step 3.

Action:

- `VectorStore.store(chunk, added, vector)` writes payload:
  - `nl`: original chunk
  - `pln`: list of added atoms

Output:

- Qdrant now has retrievable NL <-> PLN linkage for later query-time context.

### Step 7: Return ingest response

Output example:

```json
{
  "processed_count": 1,
  "results": [
    {
      "text": "People who eat fish are smart. Kebede eats fish.",
      "atoms": [
        "(: rule1 (Implication (Premises (eat_fish $x)) (Conclusions (smart $x))) (STV 0.9 0.8))",
        "(: fact1 (eat_fish kebede) (STV 1.0 0.95))"
      ],
      "status": "success"
    }
  ]
}
```

## Phase 1 Summary

Phase 1 creates the knowledge substrate:

1. Text is chunked.
2. Context is retrieved.
3. NL is translated into logic atoms.
4. Atoms are inserted into PeTTaChainer atomspace.
5. NL-to-atom mapping is indexed in Qdrant.

No proof search happens in Phase 1. It only prepares data for reasoning.
