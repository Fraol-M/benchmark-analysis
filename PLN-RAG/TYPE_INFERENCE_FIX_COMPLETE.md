# Type Inference Fix - Complete Solution

## Problem Summary
The query "Is Abebe consuming excessive carbohydrates?" was failing because:
1. The rule required `(IsA $x person)` premise
2. "Abebe" had no type declaration in the PLN statements
3. The reasoner couldn't match the rule without the type premise

## Root Cause
The `infer_entity_types()` method in `core/pln_postprocessor.py` was trying to find capitalized tokens using regex `[A-Z][a-z]+` in **already-lowercased** PLN statements (e.g., searching for "Abebe" in statements containing "abebe").

## Solution Implemented
Fixed `infer_entity_types()` to:
1. Use `proper_name_map` (built from source text) as the candidate list
2. Scan statements for entities appearing as **first argument** (subject/actor) in predicates
3. Infer `person` type only for entities in subject position
4. Skip entities with digits (object_77, unit_12, etc.)
5. Skip entities that already have explicit type declarations

## Code Changes

### File: `core/pln_postprocessor.py` (lines ~169-210)

**BEFORE (broken):**
```python
def infer_entity_types(self, statements: List[str], proper_name_map: dict[str, str]) -> List[str]:
    # Extract all proper names that appear in statements
    entities_in_statements = set()
    for stmt in statements:
        # Find all capitalized tokens - BROKEN: statements are already lowercased!
        for match in re.finditer(r'\b([A-Z][a-z]+(?:-[A-Z][a-z]+)*)\b', stmt):
            entity = match.group(1)
            if entity not in self.STRUCTURAL_HEADS:
                entities_in_statements.add(entity)
    # ... rest of broken logic
```

**AFTER (fixed):**
```python
def infer_entity_types(self, statements: List[str], proper_name_map: dict[str, str]) -> List[str]:
    """
    Infer type declarations for proper names that appear in statements.
    Uses proper_name_map (built from source text) to identify candidates,
    then checks if they appear as subjects/actors (first argument) in predicates.
    """
    if not proper_name_map:
        return []
    
    # Check which entities already have type declarations
    entities_with_types = set()
    for stmt in statements:
        # Match patterns like (IsA entity type) in any position
        # Statements are already lowercased, so match lowercase
        type_matches = re.findall(r'\(IsA\s+([a-z_][a-z0-9_]*)\s+\w+\)', stmt, re.IGNORECASE)
        entities_with_types.update(type_matches)
    
    # Find entities that appear as first argument (subject/actor) in predicates
    # These are candidates for "person" type
    subject_entities = set()
    for stmt in statements:
        # Match predicate patterns: (PredicateName first_arg ...)
        # Skip structural heads and IsA declarations
        # Pattern: (Word lowercase_identifier ...)
        matches = re.finditer(
            r'\(([A-Z][a-zA-Z0-9_]*)\s+([a-z_][a-z0-9_]*)',
            stmt
        )
        for match in matches:
            predicate = match.group(1)
            first_arg = match.group(2)
            
            # Skip structural predicates and type declarations
            if predicate in self.STRUCTURAL_HEADS or predicate == 'IsA':
                continue
            
            # Skip if it looks like an object identifier (contains digits)
            if re.search(r'\d', first_arg):
                continue
            
            # If this entity is in proper_name_map, it's a candidate
            if first_arg in proper_name_map:
                subject_entities.add(first_arg)
    
    # Infer person type for subject entities without explicit types
    inferred = []
    for entity in subject_entities:
        if entity not in entities_with_types:
            inferred.append(f"(: {entity}_is_person (IsA {entity} person) (STV 1.0 1.0))")
    
    return inferred
```

## Testing

### Unit Tests (ALL PASS ✓)
```bash
python test_type_inference.py
```

**Results:**
- ✓ Test 1: `extract_proper_name_map` finds "abebe" in source text
- ✓ Test 2: `infer_entity_types` adds `(IsA abebe person)` for subject entities
- ✓ Test 3: Full `process()` pipeline includes inferred types in output
- ✓ Test 4: Rule requiring `(IsA $x person)` has all premises satisfied

### Expected Output
After fix, the PLN Canonicalized output should include:
```
(: abebe_is_person (IsA abebe person) (STV 1.0 1.0))
(: pasta_pasta (IsA pasta pasta) (STV 1.0 1.0))
(: abebe_pasta_eats_frequently_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))
```

This allows the rule to fire:
```
(Implication 
  (Premises 
    (IsA $x person)           ← NOW SATISFIED by (IsA abebe person)
    (EatsFrequently $x $food) ← Satisfied by (EatsFrequently abebe pasta)
    (IsA $food pasta)         ← Satisfied by (IsA pasta pasta)
  )
  (Conclusions (ConsumesHighCarbs $x))
)
```

## How to Apply the Fix

### Option 1: Run Locally (Fastest for Testing)
```bash
# Activate venv
.venv\Scripts\activate

# Install dependencies (if not already done)
pip install -r requirements.txt

# Run backend locally
uvicorn api.main:app --reload --port 8000

# In another terminal, run Streamlit UI
streamlit run debug_ui/app.py --server.port 8501
```

### Option 2: Rebuild Docker (Slow - 10+ minutes)
```bash
# Stop containers
docker-compose down

# Rebuild (this compiles SWI-Prolog from source - takes 10+ minutes)
docker-compose build --no-cache pln-rag

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f pln-rag
```

### Option 3: Copy Code into Running Container (Quick Hack)
```bash
# Copy fixed file into running container
docker cp core/pln_postprocessor.py pln-rag-pln-rag-1:/app/core/pln_postprocessor.py

# Restart container to reload code
docker restart pln-rag-pln-rag-1

# Check logs
docker logs -f pln-rag-pln-rag-1
```

## Verification Steps

1. **Clear the database** (important!)
   - Click "Clear Database" button in UI
   - OR call `/reset` endpoint: `curl -X POST http://localhost:8000/reset`

2. **Re-ingest the text:**
   ```
   Abebe eats pasta almost every day, often in large portions, and prefers refined pasta over whole grain. He works a desk job and gets little physical activity. While he does not consume many sugary foods, and there is no known family history of diabetes, his weight has increased. However, he has not been clinically classified as obese.
   ```

3. **Check "PLN Canonicalized" tab** - should show:
   ```
   (: abebe_is_person (IsA abebe person) (STV 1.0 1.0))
   ```

4. **Run query:**
   ```
   Is Abebe consuming excessive carbohydrates?
   ```

5. **Expected result:** Query should succeed with reasoning chain showing the rule fired.

## Files Modified
- `core/pln_postprocessor.py` - Fixed `infer_entity_types()` method
- `test_type_inference.py` - Unit tests (all passing)
- `TYPE_INFERENCE_FIX.md` - Original documentation
- `TYPE_INFERENCE_FIX_COMPLETE.md` - This complete guide

## Key Insights
1. **Proper names are lowercased** in PLN statements by the time they reach postprocessor
2. **Source text preserves capitalization** - use `proper_name_map` built from source
3. **Subject position matters** - only infer "person" for entities appearing as first argument
4. **Universal identity** - `(IsA pasta pasta)` is generated separately for all terms
5. **Two-layer defense** - LangExtract examples + postprocessor type inference

## Status
✅ **Fix implemented and tested**
✅ **Unit tests passing**
⏳ **Waiting for Docker rebuild OR run locally to test live system**

The fix is correct and proven by unit tests. The only remaining step is getting the updated code into the running environment.
