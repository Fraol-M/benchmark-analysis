# Phase 2 README: Reasoning and Query over MeTTa/PLN Atomspace

This document explains Phase 2 of the PLN-RAG pipeline: taking a natural language question, converting it into a PLN query, running reasoning in PeTTaChainer, and returning an answer.

## Goal

Use previously ingested atoms/rules to prove or disprove a query, then present result in natural language.

## Main Components

- `core/service.py` (`PLNRAGService.query`)
- parser (`parse_query` or `parse`)
- `core/reasoner.py` (`Reasoner.query`)
- `storage/vector_store.py` (`retrieve_context`)
- `core/answer_generator.py` (`AnswerGenerator.generate`)

## End-to-End Example

Assume Phase 1 already inserted atoms equivalent to:

```text
(: rule1 (Implication (Premises (eat_fish $x)) (Conclusions (smart $x))) (STV 0.9 0.8))
(: fact1 (eat_fish kebede) (STV 1.0 0.95))
```

User question:

```text
Is Kebede smart?
```

### Step 1: API receives query request

Input:

```json
{
  "question": "Is Kebede smart?"
}
```

Action:

- `/query` endpoint calls `PLNRAGService.query(question)`.

Output:

- Query pipeline starts, no final answer yet.

### Step 2: Retrieve context for question translation

Input:

- Question text.

Action:

- `VectorStore.retrieve_context(question, top_k)` finds relevant past atoms.
- `_enrich_context(...)` merges in recent atoms from `data/atomspace/kb.metta`.

Output:

- Context atoms to guide parser, e.g. predicates like `eat_fish`, `smart`, and existing rules.

### Step 3: Parse question into PLN query candidates

Input:

- Question text.
- Retrieved/merged context.

Action:

- Parser runs `parse_query(...)` (if available) or `parse(...)`.
- Parser can return multiple query candidates for fallback execution order.

Output (`ParseResult.queries`, illustrative):

```text
[
  "(: $prf (smart kebede) $tv)",
  "(: $prf (smart $x) $tv)"
]
```

Possible extra output:

- `ParseResult.statements` may contain helper statements to add before querying.

### Step 4: Optional helper statements are inserted

Input:

- `parse_result.statements` (if any).

Action:

- `Reasoner.add_statements(...)` adds helper atoms/rules to atomspace.

Output:

- Atomspace is updated before proof search.

### Step 5: PeTTaChainer reasoning/proof search

Input:

- Ordered query candidates from Step 3.

Action:

- For each candidate, service calls `Reasoner.query(candidate)`.
- `Reasoner.query` delegates to `PeTTaChainer.query(...)`.
- Stops at first candidate that returns non-empty proof traces.
- If `QUERY_FALLBACK_ENABLED=false`, only first query is executed.

Output:

- `proof_traces`: proof objects/strings if derivation succeeds, else `[]`.
- `executed_query`: the candidate that was actually run last.
- `fallback_used`: true if a non-first candidate solved it.

### Step 6: Query status classification

Input:

- Original question and executed/original query relationship.

Action:

- Service labels query as:
  - `well_aligned`
  - `weakly_aligned`
  - `no_query`

Output:

- `query_status` included in response, useful for debugging parse-vs-reasoning mismatch.

### Step 7: Source sentence extraction

Input:

- `proof_traces`.

Action:

- Service extracts atoms from proof traces.
- Performs reverse lookup in Qdrant to recover closest original NL source sentences.

Output:

- `sources`: list of supporting natural language snippets.

### Step 8: Natural-language answer generation

Input:

- Original question.
- Proof trace strings.

Action:

- `AnswerGenerator.generate(question, proof_traces)` uses DSPy+LLM to verbalize proof.
- If no proof: returns "I don't know - no proof was found..." (or weak-alignment message).

Output:

- Final answer text.

### Step 9: Return final query response

Output example:

```json
{
  "question": "Is Kebede smart?",
  "pln_query": "(: $prf (smart kebede) $tv)",
  "original_query": "(: $prf (smart kebede) $tv)",
  "executed_query": "(: $prf (smart kebede) $tv)",
  "fallback_used": false,
  "query_status": "well_aligned",
  "raw_proof": "[ ...proof trace from PeTTaChainer... ]",
  "sources": ["People who eat fish are smart.", "Kebede eats fish."],
  "answer": "Yes. Based on the rule and the fact that Kebede eats fish, Kebede is smart."
}
```

## Phase 2 Summary

Phase 2 does the actual reasoning:

1. Question -> query candidates.
2. Queries run in PeTTaChainer against atomspace.
3. Proof traces are produced (or not).
4. Proof is turned into user-facing natural language answer.

The core reasoner is PeTTaChainer; LLM is used for query translation and answer narration.
