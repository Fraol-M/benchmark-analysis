# Future Plan — patterns to port from PLN-RAG

Sister project `../PLN-RAG/` solves the same NL→symbolic problem against PeTTaChainer/PLN. It hit real reasoning failures and engineered fixes for them. Most of those fixes apply to this demo too.

Each section: **problem → concrete failure example → fix → PLN-RAG analog**.

---

## 1. Symbol canonicalization (high priority)

**Problem.** `translator._normalize_text` only lowercases predicate heads and hyphenates spaces. Entity/class slots keep raw casing and plurality.

**Failure example.** Two chunks of one document:

```
chunk 1: "Cats are mammals."
chunk 2: "A cat is a mammal."
```

Current output:

```lisp
(Inheritance Cats mammals)
(Inheritance cat mammal)
```

Two atoms for one fact. Query `(match &self (Inheritance cat $p) $p)` returns `mammal` only, misses `Cats`. Reasoner case-sensitive, not semantics-aware — different symbol = different concept.

**Fix.** Lemmatize + lowercase common nouns (`Cats`/`cats` → `cat`, `mammals` → `mammal`). Protect proper nouns extracted from source capitalization. Mirrors PLN-RAG `CanonicalPLNParser._canonical_symbol` + `_extract_proper_name_map`.

After fix:

```lisp
(Inheritance cat mammal)   ; both chunks collapse to one atom
```

---

## 2. Cross-chunk predicate reuse (high priority)

**Problem.** Each chunk runs `lx.extract` in isolation. LLM picks predicate names per-call. Across chunks, same concept gets different heads.

**Failure example.** Long doc:

```
chunk 1: "Bob eats flies."                       → (eats-flies Bob)
chunk 2: "Sam consumes flies."                   → (consumes-flies Sam)
chunk 3: "Anything that eats flies is hungry."   → rule on (eats-flies $x)
```

Rule body uses `(eats-flies $x)` only — Sam never matches. Query `(match &self (hungry $x) $x)` returns Bob, never Sam.

**Fix.** After each completed chunk, collect predicate heads from emitted atoms. Append to next chunk's prompt as hints (`"preferred predicate heads: eats-flies, isa, croaks"`). Mirrors PLN-RAG `_extract_context_predicates` + `_build_parser_inputs`.

After fix:

```lisp
(eats-flies Bob)
(eats-flies Sam)            ; chunk 2 reuses head from chunk 1
(= (hungry $x) (match &self (eats-flies $x) True))
```

Query returns both.

---

## 3. Statement safety filter (high priority)

**Problem.** `pipeline.validate_metta_atoms` *reports* parse/load errors but `populate_space` already added everything. Bad atoms pollute AtomSpace.

**Failure example.** LLM emits free-variable fact:

```lisp
(eats $x flies)             ; should have been wrapped in a rule
```

Loaded as standalone fact. Query `(match &self (eats $who flies) $who)` returns `$x` — meaningless variable binding. Worse, shadows real grounded facts in some matchers.

**Fix.** Filter pre-load:
- Drop facts containing free variables outside `rule` context.
- Drop rules whose body fails `parse_single`.
- Surface rejects in UI as "Filtered" panel, not silent.

Mirrors PLN-RAG `_filter_statements` + `_has_valid_implication_shape`.

---

## 4. NL answer generator for Query REPL (medium)

**Problem.** Query tab returns raw MeTTa atoms. End-user has to read `(isa Sam frog)` to answer "Is Sam a frog?".

**Failure example.**
- User wants: "Who eats flies?"
- User must translate to: `! (match &self (eats-flies $who) $who)`
- Result: `Bob`, `Sam` — bare symbols, no sentence.

**Fix.** Second Gemini call: `(question, atoms_returned) → NL answer`. Mirrors PLN-RAG `AnswerGenerator._ProofToAnswer`. Same API key already configured.

After fix, user types "Who eats flies?" in plain English. UI generates the MeTTa query, runs it, then translates atoms back: *"Bob and Sam eat flies."*

---

## 5. Source reverse-lookup (almost free — data already exists)

**Problem.** Each `Extraction` already carries `char_interval` and `extraction_text`. Currently shown only in Raw JSON tab. MeTTa and Query tabs lose link to source.

**Failure example.** Query returns `(isa Sam frog)`. User: "where did this come from?" Currently: scroll Raw JSON tab, eyeball.

**Fix.** Build `{atom_str: extraction_text}` map in `populate_space`. Render atoms as expandable: click atom → show source span + char range. Mirrors PLN-RAG `_extract_sources` (which has to do extra Qdrant lookups; we get it free from langextract's alignment).

---

## 6. Atomspace persistence (medium)

**Problem.** Session-only state. Re-run extraction on same doc = re-pay Gemini cost.

**Fix.** Append-only `data/atomspace.metta`. Save button + auto-load on startup. Mirrors PLN-RAG `Reasoner._load_from_disk` + `add_statements` file write.

---

## 7. Sentence-merge chunker (medium)

**Problem.** `_split_text` splits on paragraphs/sentences. Pronouns lose referents across boundaries.

**Failure example.** Text:

```
Sam is a frog. He croaks.
```

Large doc + small `chunk_size` → chunker splits between sentences. Chunk 2 = "He croaks." LLM has no idea who "He" is. Output: nothing, or wrong subject.

**Fix.** Before size-based split, merge a sentence into the prior chunk when it starts with a coreference cue (`it`, `this`, `they`, `he`, `she`, `therefore`, `thus`, `however`). Mirrors PLN-RAG `Chunker._should_merge_with_previous` + `_merge_cues`.

After fix: "Sam is a frog. He croaks." stays in one chunk → `(isa Sam frog)` + `(croaks Sam)`.

---

## 8. Pluggable extractor backend (lower priority)

**Problem.** `pipeline.run_extraction` hardcodes `lx.extract` with Gemini defaults. Can't A/B Ollama vs Gemini vs OpenAI without code edits.

**Fix.** Define `SemanticExtractor` ABC mirroring PLN-RAG's `SemanticParser`. Factory in `pipeline.py` switches by env var. Lets you compare extraction quality across LLMs on the same examples.

---

## 9. Query planner with fallback chain (lower priority)

**Problem.** Query REPL runs exactly one query. If user's shape doesn't match what KB derives, they get zero results with no hint why.

**Failure example.** User asks `(match &self (Inheritance frog $p) $p)` but KB stored `(IsA frog animal)`. Empty result.

**Fix.** Suggest alternative shapes: scan KB for predicates of matching arity; offer "did you mean `(IsA frog $p)`?". Mirrors PLN-RAG `_plan_queries` + `_build_grounded_yes_no_fallbacks`.

---

## 10. REST API wrapper (lower priority)

**Problem.** Streamlit only — can't script extractions or integrate from another service.

**Fix.** Thin FastAPI layer over `run_extraction` / `text_to_atomspace`. Direct mirror of PLN-RAG `api/main.py`.

---

## Why these matter together

PLN-RAG learned (the hard way) that NL→symbolic systems fail at three layers:

1. **Symbol layer** — same concept, different atoms. Fixed by canonicalization (#1) + cross-chunk reuse (#2).
2. **Structural layer** — malformed atoms slip into KB and corrupt queries. Fixed by safety filter (#3) + sentence-merge chunker (#7).
3. **UX layer** — symbolic results unreadable; no provenance back to source. Fixed by NL answer generator (#4) + source reverse-lookup (#5).

Demo currently solves layer 0 (extraction works at all). Layers 1–3 are where reasoning quality lives. Most fixes are 50–200 lines, lifted near-verbatim from PLN-RAG.

---

## Suggested order

1. **#1 canonicalization** — biggest reasoning correctness win, smallest code change.
2. **#3 safety filter** — stops bad atoms from poisoning Query REPL.
3. **#5 source reverse-lookup** — UI win, data already in hand.
4. **#2 cross-chunk reuse** — needed for any document longer than one chunk.
5. **#4 NL answer generator** — closes UX loop.
6. Rest as needed.
