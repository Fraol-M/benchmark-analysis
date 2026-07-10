# PLN-RAG Refactoring Summary: LangExtract-Only System

## Overview
Converted PLN-RAG from a pluggable multi-parser architecture to a **LangExtract-only** system by removing parser abstraction layers and hardcoding the LangExtract pipeline.

## Changes Made

### 1. Deleted Files (Parser Abstraction)
- ❌ `core/parser.py` - Abstract `SemanticParser` base class
- ❌ `parsers/nl2pln_parser.py` - NL2PLN parser implementation
- ❌ `parsers/canonical_pln_parser.py` - Canonical PLN parser implementation
- ❌ `parsers/canonical_pln_prev_parser.py` - Previous canonical PLN parser
- ❌ `parsers/manhin_parser.py` - Manhin parser implementation
- ❌ `parsers/canonical_pln_parser_spec.md` - Canonical PLN parser specification

### 2. Modified Files

#### `parsers/__init__.py`
**Before:**
```python
from core.parser import SemanticParser
from config import get_settings

def get_parser() -> SemanticParser:
    cfg = get_settings()
    name = cfg.parser.lower()
    # ... factory logic for multiple parsers
```

**After:**
```python
from parsers.langextract_pln_parser import LangExtractPLNParser

__all__ = ["LangExtractPLNParser"]
```

#### `parsers/langextract_pln_parser.py`
**Changes:**
- Removed inheritance from `SemanticParser` abstract base
- Added `ParseResult` dataclass directly to this file (no longer imported from `core.parser`)
- Changed from `class LangExtractPLNParser(SemanticParser):` to `class LangExtractPLNParser:`
- Updated docstring to remove "phase-1" and Hyperon/MeTTa references

#### `api/main.py`
**Before:**
```python
from parsers import get_parser
from config import get_settings

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    cfg = get_settings()
    print(f"[Startup] Loading parser: {cfg.parser}")
    parser = get_parser()
    _service = PLNRAGService(parser)
```

**After:**
```python
from parsers import LangExtractPLNParser

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    print("[Startup] Initializing LangExtract parser...")
    parser = LangExtractPLNParser()
    _service = PLNRAGService(parser)
```

#### `config.py`
**Removed:**
- `parser: str = "canonical_pln"` - Parser selection config
- `nl2pln_module_path: str = "data/simba_all.json"` - NL2PLN config
- `canonical_pln_nl2pln_module_path: str = "data/simba_canonical_pln.json"` - Canonical PLN config

**Updated:**
- Changed `faiss_path` comment from "used by Manhin parser" to "no longer used"

#### `core/service.py`
**Changes:**
- Changed import from `from core.parser import SemanticParser` to `from parsers.langextract_pln_parser import LangExtractPLNParser, ParseResult`
- Updated `__init__` signature from `def __init__(self, parser: SemanticParser):` to `def __init__(self, parser: LangExtractPLNParser):`
- Removed dynamic chunker creation: `create_chunker = getattr(parser, "create_chunker", None)` → directly call `parser.create_chunker()`
- Removed `hasattr` checks for `debug_parse`, `parse_query`, `debug_parse_query`, `reset` - now directly call these methods
- Updated docstring from "Orchestrates the full pipeline" to "Orchestrates the full LangExtract pipeline"

#### `core/pln_postprocessor.py`
**Critical Bug Fix:**
- Fixed truncated `generate_universal_identity()` method
- **Before (BROKEN):**
  ```python
  if token.islower() and not re.match(r'^[0-9.]+
  ```
- **After (FIXED):**
  ```python
  if token.islower() and not re.match(r'^[0-9.]+$', token):
  ```

## Architecture Changes

### Before (Pluggable)
```
┌─────────────────────────────────────┐
│         SemanticParser (ABC)        │
│  - parse(text, context)             │
│  - Optional: parse_query()          │
│  - Optional: debug_parse()          │
│  - Optional: create_chunker()       │
└─────────────────────────────────────┘
                  ▲
                  │
    ┌─────────────┼─────────────┬─────────────┐
    │             │             │             │
┌───┴───┐   ┌────┴────┐   ┌────┴────┐   ┌────┴────┐
│NL2PLN │   │Canonical│   │ Manhin  │   │LangExt  │
│Parser │   │PLNParser│   │ Parser  │   │PLNParser│
└───────┘   └─────────┘   └─────────┘   └─────────┘
```

### After (LangExtract-Only)
```
┌─────────────────────────────────────┐
│     LangExtractPLNParser            │
│  - parse(text, context)             │
│  - parse_query(question, context)   │
│  - debug_parse(text, context)       │
│  - debug_parse_query(q, context)    │
│  - create_chunker()                 │
│  - reset()                          │
└─────────────────────────────────────┘
```

## Benefits

1. **Simplified Architecture** - No abstract base classes or factory patterns
2. **Reduced Complexity** - Removed ~2000+ lines of unused parser code
3. **Direct Integration** - Service directly instantiates `LangExtractPLNParser`
4. **Bug Fix** - Fixed critical syntax error in `pln_postprocessor.py`
5. **Clearer Intent** - System is explicitly LangExtract-focused

## Remaining LangExtract Components

The following LangExtract-specific files remain and are now the core of the system:

- ✅ `core/langextract_chunker.py` - Paragraph-first, coreference-aware chunking
- ✅ `core/langextract_examples.py` - Loads prompts/examples from JSON
- ✅ `core/langextract_pln.py` - Translates LangExtract objects → PLN
- ✅ `parsers/langextract_pln_parser.py` - Main parser orchestration
- ✅ `data/langextract_examples.json` - Few-shot examples (30+ statements, 7+ queries)

## Testing Recommendations

1. **Syntax Check** ✅ - All files compile without errors
2. **Import Test** - Verify `from parsers import LangExtractPLNParser` works
3. **Service Initialization** - Test `PLNRAGService(LangExtractPLNParser())`
4. **API Startup** - Verify FastAPI app starts without errors
5. **Ingest Test** - Test `/ingest` endpoint with sample text
6. **Query Test** - Test `/query` endpoint with sample question
7. **Debug Endpoints** - Test `/debug/ingest` and `/debug/query`

## Migration Notes

- **No `.env` changes needed** - LangExtract config variables remain unchanged
- **No API changes** - All endpoints remain the same
- **No data format changes** - PLN output format unchanged
- **Backward compatible** - Existing knowledge bases work as-is

## Files Summary

| Category | Count | Details |
|----------|-------|---------|
| **Deleted** | 6 | Parser implementations + abstract base |
| **Modified** | 5 | Parser init, API, config, service, postprocessor |
| **Unchanged** | 20+ | All other core, storage, API models, tests |
| **Bug Fixes** | 1 | Critical syntax error in postprocessor |

---

**Refactoring completed successfully!** The system is now a streamlined LangExtract-only PLN-RAG pipeline.
