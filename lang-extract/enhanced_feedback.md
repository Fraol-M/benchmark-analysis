# Enhanced Feedback — Demo LangExtract Implementation

A detailed code review and improvement roadmap for the **LangExtract → AtomSpace** demo pipeline.  
Based on a full source-code reading of both this project and the sister **PLN-RAG** project.

---

## ✅ What's Done Well

### 1. Few-Shot Extraction Examples
The 8 `ExampleData` objects in `examples.py` are diverse, well-structured, and cover a wide range of extraction patterns — binary/unary facts, type declarations, inheritance, conditional rules with `(and ...)` conjunctions, negation, hyphenated/numeric identifiers (`Object-77`, `Rack-01`), transitive chains, and multi-variable joins. This is the backbone of extraction quality and it's solid.

### 2. Canonicalization
`translator.py` implements proper-name protection via `build_canonicalization_context()` — it scans the source text to detect words that are always capitalized outside sentence-initial position, and preserves their casing. Common nouns are lowercased and singularized. Tokens with hyphens or digits are preserved verbatim. This prevents many duplicate-atom issues.

### 3. Safety Filter
`is_safe_extraction()` catches real problems before they pollute the AtomSpace:
- Drops facts containing free variables (e.g., `(eats $x flies)` without a rule wrapper)
- Drops rules with empty or unparseable bodies
- Drops rules with empty head predicates

This mirrors PLN-RAG's `_filter_statements` + `_has_valid_implication_shape` and is one of the most important correctness features.

### 4. Streaming Pipeline
`run_extraction_stream()` yields progress events (`phase`, `fetched`, `chunked`, `chunk_done`, `done`, `error`) that the Streamlit UI consumes for live feedback — progress bars, chunk counters, elapsed time, in-flight count. This is production-grade UX for a potentially slow LLM pipeline.

### 5. Source Reverse-Lookup
The `atom_to_source` map built in `populate_space()` links every generated atom back to its source sentence + character interval. This is shown in the UI's "Atom → source sentence" expander. PLN-RAG has to do extra Qdrant lookups to achieve the same thing — LangExtract's alignment gives it for free.

### 6. Coreference-Aware Chunking
`_should_merge_with_previous()` and `_merge_sentence_groups()` keep sentences starting with pronouns or discourse markers (`it`, `they`, `this`, `therefore`, `however`) glued to the previous chunk. This prevents broken pronoun references when the LLM processes chunks in isolation.

### 7. Cross-Chunk Predicate Reuse
In sequential mode, `run_extraction_stream()` accumulates predicate heads from completed chunks and appends them as prompt hints to subsequent chunks (`"Preferred predicate heads: eats-flies, isa, croaks"`). This steers the LLM toward consistent vocabulary across a long document.

---

## 🔴 High-Impact Improvements (Priority Order)

### 1. Rules Only Do 1-Hop Matching — The Biggest Reasoning Gap

**Problem:**  
Rules are encoded as:
```lisp
(= (has $x color green)
   (match &self (, (isa $x frog) (croaks $x) (eats-flies $x)) True))
```

This only matches **existing ground atoms** in the space. It never fires other rules to derive intermediate facts. So a 2-hop chain like:

```
"Frogs croak. Sam is a frog. Things that croak are loud."
```

produces:
```lisp
(isa Sam frog)
(= (croaks $x) (match &self (isa $x frog) True))       ; rule, not a ground fact
(= (loud $x) (match &self (croaks $x) True))            ; can't find (croaks Sam) — it's a rule, not a fact
```

The query `(match &self (loud $who) $who)` returns nothing because `(croaks Sam)` was never materialized as a ground atom.

**Fix:**  
After loading all atoms, run a **forward-chaining saturation loop** — repeatedly evaluate all rule heads and add derived atoms until no new atoms are produced:

```python
def saturate(metta: MeTTa, max_iterations: int = 10) -> int:
    """Forward-chain: evaluate all rules, add derived facts, repeat until fixpoint."""
    total_new = 0
    for _ in range(max_iterations):
        # Get all rule heads
        rules = metta.run("! (match &self (= $head $body) $head)")
        new_atoms = set()
        for block in rules:
            for atom in block:
                atom_str = str(atom)
                if atom_str and not atom_str.startswith("$"):
                    new_atoms.add(atom_str)

        if not new_atoms:
            break

        added = 0
        for atom_str in new_atoms:
            try:
                metta.run(atom_str)
                added += 1
            except Exception:
                pass

        if added == 0:
            break
        total_new += added
    return total_new
```

This is ~30 lines and transforms the demo from an "extraction viewer" into an actual reasoner.

---

### 2. Add NL Question → MeTTa Query Translation

**Problem:**  
The Query REPL requires users to write raw MeTTa:
```lisp
! (match &self (isa $who frog) $who)
```

This is expert-only. Non-technical users cannot interact with the extracted knowledge.

**Fix:**  
Add a single LLM call (using the same Gemini key already configured) that translates natural language questions into MeTTa queries:

```python
def nl_to_query(question: str, atom_sample: list[str], model_id: str = "gemini-2.5-flash") -> str:
    """Translate a natural language question into a MeTTa match expression."""
    prompt = f"""Given these atoms in the knowledge base:
{chr(10).join(atom_sample[:20])}

Translate this question into a MeTTa query:
Question: {question}

Return ONLY the MeTTa expression, e.g.: ! (match &self (isa $who frog) $who)"""

    # Use langextract or direct Gemini call
    ...
```

This would let users type `"Who is a frog?"` instead of `! (match &self (isa $who frog) $who)`.

---

### 3. Add NL Answer Generation

**Problem:**  
Query results are bare atoms: `Sam`. The user has to mentally reconstruct "Sam is a frog" from the query context.

**Fix:**  
One more LLM call after query execution:

```python
def atoms_to_answer(question: str, result_atoms: list[str]) -> str:
    """Convert query results into a natural language answer."""
    prompt = f"""Question: {question}
Query results (MeTTa atoms): {result_atoms}
Write a concise natural language answer based strictly on these results."""
    ...
```

PLN-RAG's `AnswerGenerator` is only ~40 lines using DSPy. A direct Gemini version would be even simpler.

Combined with improvement #2, this closes the full UX loop:
```
User: "Who is a frog?" → System: "Sam is a frog."
```

---

### 4. `_singularize()` Misses Common Irregular Nouns

**Problem:**  
The current singularizer in `translator.py` handles suffix rules (`-ies` → `-y`, `-ses` → `-se`, `-s` → remove) but misses common irregular plurals:

| Input | Current Output | Correct Output |
|---|---|---|
| `men` | `men` | `man` |
| `women` | `women` | `woman` |
| `children` | `children` | `child` |
| `people` | `people` | `person` |
| `mice` | `mice` | `mouse` |
| `teeth` | `teeth` | `tooth` |
| `feet` | `feet` | `foot` |
| `geese` | `geese` | `goose` |
| `data` | `data` | `datum` |
| `phenomena` | `phenomena` | `phenomenon` |

This causes duplicate atoms: `(Inheritance children animal)` and `(Inheritance child animal)` for the same concept.

**Fix:**  
Add an irregular lookup table before the suffix rules in `_singularize()`:

```python
_IRREGULAR_PLURALS = {
    "men": "man", "women": "woman", "children": "child",
    "people": "person", "mice": "mouse", "teeth": "tooth",
    "feet": "foot", "geese": "goose", "oxen": "ox",
    "data": "datum", "phenomena": "phenomenon",
    "criteria": "criterion", "analyses": "analysis",
    "leaves": "leaf", "lives": "life", "knives": "knife",
}

def _singularize(word: str) -> str:
    if word in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[word]
    # ... existing suffix rules ...
```

---

### 5. Add Basic Persistence (Save/Load AtomSpace)

**Problem:**  
Every Streamlit rerun re-calls the LLM. Extracting the same document twice wastes API credits (~$0.01–0.05 per extraction).

**Fix:**  
Add "Save" and "Load" buttons in the UI that write/read a `.metta` file:

```python
# Save
def save_atomspace(metta_str: str, path: str = "data/atomspace.metta"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(metta_str)

# Load on startup
def load_atomspace(path: str = "data/atomspace.metta") -> str | None:
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None
```

PLN-RAG's `Reasoner` uses exactly this pattern (append-only file + load on init).

---

### 6. Disambiguate `isa` vs `Inheritance` in Extraction Prompt

**Problem:**  
The extraction prompt teaches both:
- `fact` with `predicate=isa` → `(isa Sam frog)` — instance membership ("Sam is one frog")
- `inheritance` → `(Inheritance frog animal)` — class hierarchy ("all frogs are animals")

But the LLM sometimes confuses these. "A cat is an animal" could produce either form. They have different semantics and querying them requires different patterns.

**Fix:**  
Add explicit disambiguation to `EXTRACTION_PROMPT`:

```
IMPORTANT distinction:
- Use 'fact' with predicate='isa' for INSTANCES: "Sam is a frog" means Sam is one specific frog
- Use 'inheritance' for CLASS HIERARCHIES: "Frogs are animals" means the entire class of frogs is a subclass of animals
- Hint: if the subject is a proper noun (Sam, Tom, Kebede), use 'fact' with 'isa'
- Hint: if the subject is a common noun (frogs, cats, dogs), use 'inheritance'
```

---

### 7. Parallel Mode Loses Cross-Chunk Predicate Consistency

**Problem:**  
In parallel mode (`max_workers > 1`), all chunks are extracted independently — there's no predicate vocabulary sharing between them. The same concept can get different predicate names:

```
Chunk 1: "Bob eats flies."     → (eats-flies Bob)
Chunk 2: "Sam consumes flies." → (consumes-flies Sam)  ← different predicate!
```

A rule about `(eats-flies $x)` would never match Sam.

Your sequential mode already fixes this (predicate hints flow from chunk to chunk), but the UI defaults to `max_workers=5`.

**Fix options:**
1. Default to sequential mode for documents under ~10 chunks (where latency is acceptable)
2. Add a post-extraction dedup pass that detects predicate synonyms (same arguments, similar head names) and merges them
3. Show a warning in the UI when parallel mode is used with multiple chunks

---

### 8. Add Automated Extraction Quality Tests

**Problem:**  
There is no test suite. Changes to the extraction prompt, translator logic, or canonicalization can silently break extraction quality.

**Fix:**  
Create a `tests/test_extraction.py` with deterministic assertions:

```python
import pytest
from langextract_atomspace import text_to_metta_str

def test_basic_facts():
    atoms = text_to_metta_str("Sam is a frog. Sam croaks.")
    assert "(isa Sam frog)" in atoms
    assert "(croaks Sam)" in atoms

def test_inheritance():
    atoms = text_to_metta_str("A frog is a kind of animal.")
    assert "(Inheritance frog animal)" in atoms

def test_rule_generation():
    atoms = text_to_metta_str("Every frog that croaks is green.")
    assert "(=" in atoms  # contains a rule
    assert "frog" in atoms
    assert "green" in atoms

def test_negation():
    atoms = text_to_metta_str("Tom is not a frog.")
    assert "(not" in atoms

def test_hyphenated_ids():
    atoms = text_to_metta_str("Object-77 is fragile.")
    assert "Object-77" in atoms

def test_canonicalization_dedup():
    atoms = text_to_metta_str("Cats are mammals. A cat is a mammal.")
    # Should produce one atom, not two
    assert atoms.count("Inheritance") == 1
```

Even 5-10 tests like these catch regressions when you change the prompt or translator.

> **Note:** LLM-based extraction is non-deterministic. Use `temperature=0` and run `n=3` times, asserting the majority result passes.

---

## 🟡 Lower Priority Improvements

### 9. `validate_metta_atoms` Runs After `populate_space`

Currently, bad atoms are added to the space first, then validation reports errors after the fact. A truly bad atom is already in the `GroundingSpaceRef` and could corrupt queries.

**Fix:** Validate each atom *before* adding it to the space inside `populate_space()`, or do a validate-then-add single pass.

### 10. URL Mode Limitations

- Only handles HTML — no PDF, DOCX, or other document formats
- No JavaScript rendering — SPAs and dynamically-loaded pages return empty content
- The 8,000-char / 12-paragraph trim is aggressive for long articles

Consider adding a note in the UI: *"Works best with static HTML pages. JS-heavy sites may return incomplete content."*

### 11. No Confidence or Uncertainty

Every extracted atom is treated as equally certain. PLN-RAG's `STV` (Simple Truth Value) with strength and confidence is more expressive:
```lisp
;; PLN-RAG can express uncertainty:
(: maybe_fact (Eats kebede fish) (STV 0.8 0.6))
```

For a future version, consider adding a confidence field to extractions based on alignment status (exact match = high confidence, fuzzy = lower).

### 12. The Comma-Based Conjunction in Rule Bodies

Rule bodies with multiple conjuncts use:
```lisp
(match &self (, (isa $x frog) (croaks $x)) True)
```

The `,` (comma) conjunction operator may not work consistently across all Hyperon versions. Consider using nested `(and ...)` as a safer alternative if you encounter issues.

---

## 📊 Summary: Impact vs Effort Matrix

| Improvement | Impact | Effort | Priority |
|---|---|---|---|
| Forward-chaining saturation | 🔴 Critical | ~30 lines | **#1** |
| NL question → MeTTa query | 🔴 High | ~50 lines | **#2** |
| NL answer generation | 🔴 High | ~40 lines | **#3** |
| Irregular noun table | 🟡 Medium | ~15 lines | **#4** |
| Basic persistence | 🟡 Medium | ~20 lines | **#5** |
| Prompt disambiguation | 🟡 Medium | ~5 lines | **#6** |
| Parallel mode warning | 🟡 Medium | ~10 lines | **#7** |
| Automated tests | 🟡 Medium | ~50 lines | **#8** |
| Validate-before-add | 🟢 Low | ~10 lines | **#9** |
| URL mode notes | 🟢 Low | ~2 lines | **#10** |

---

## 🎯 Recommended Next Steps

**Phase 1 — Reasoning (biggest win):**
1. Implement forward-chaining saturation loop
2. Add NL question input + NL answer output
3. Test with multi-hop examples (freezing → decreasing heat → decreasing temp → molecules slow)

**Phase 2 — Robustness:**
4. Add irregular noun table to `_singularize()`
5. Add `isa` vs `Inheritance` disambiguation to prompt
6. Create 10 basic extraction tests

**Phase 3 — Production readiness:**
7. Add persistence (save/load `.metta` files)
8. Default to sequential mode for small documents
9. Validate atoms before adding to space
