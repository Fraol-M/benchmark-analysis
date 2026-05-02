# Phase 2 Reasoning And Docker

The project is now dockerized as a two-service stack:

- `app`: Streamlit UI plus LangExtract Phase 1.
- `petta-reasoner`: PeTTaChainer service for PLN reasoning.

The PeTTa service code lives inside the package at:

```text
src/langextract_atomspace/petta_reasoner_service/
```

## Full Stack

From `lang-extract/`:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8501
```

The Streamlit container talks to PeTTaChainer through:

```text
PETTA_REASONER_URL=http://petta-reasoner:8010
```

The reasoner is also exposed on the host for debugging:

```text
http://localhost:8010/health
```

## Reasoner Only

If you only want the PeTTaChainer service:

```bash
docker compose -f docker-compose.reasoning.yml up --build
```

## Phase 2 Flow

Example Phase 1 atoms:

```lisp
(eats kebede fish)
(= (smart $x) (match &self (eats $x fish) True))
```

Translated PLN statements:

```lisp
(: kebede_fish_eats_fact (Eats kebede fish) (STV 1.0 1.0))
(: x_smart_rule (Implication (Premises (Eats $x fish)) (Conclusions (Smart $x))) (STV 1.0 1.0))
```

Query:

```lisp
(: $prf (Smart kebede) $tv)
```

## UI Steps

1. Run `docker compose up --build`.
2. Open `http://localhost:8501`.
3. Extract natural-language text.
4. Open the `PLN Reasoning` tab.
5. Click `Load into PeTTaChainer`.
6. Run a PLN query.

## Notes

- The app image is lightweight and contains Hyperon, LangExtract, Streamlit, and the UI.
- The PeTTa image is heavier because it builds SWI-Prolog, `janus_swi`, PeTTa, and PeTTaChainer.
- Current truth values default to `(STV 1.0 1.0)`.
- Natural-language-to-PLN query generation is not implemented yet; the first version expects PLN query syntax.
