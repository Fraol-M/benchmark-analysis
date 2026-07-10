# Tests

Automatic tests live directly in this folder and should run without live LLM,
Qdrant, Ollama, or network services:

```bash
python -m unittest discover -s tests
```

Manual experiments live in `tests/manual/`. They are scripts for local debugging
and may require API keys, Ollama, Qdrant, or the full service stack.

