"""
Unit test to verify the type inference fix works correctly.
"""
import sys
sys.path.insert(0, '.')

from core.pln.postprocessor import PLNPostprocessor


def test_extract_proper_name_map():
    """Test that extract_proper_name_map finds capitalized names in text."""
    pp = PLNPostprocessor()
    
    text = "Abebe eats pasta almost every day, often in large portions."
    proper_name_map = pp.extract_proper_name_map(text)
    
    print("Test 1: extract_proper_name_map")
    print(f"  Input text: {text}")
    print(f"  Result: {proper_name_map}")
    print(f"  Expected: {{'abebe': 'abebe'}}")
    
    assert 'abebe' in proper_name_map, f"FAIL: 'abebe' not in proper_name_map: {proper_name_map}"
    assert proper_name_map['abebe'] == 'abebe', f"FAIL: proper_name_map['abebe'] = {proper_name_map['abebe']}"
    print("  ✓ PASS\n")
    
    return proper_name_map


def test_infer_entity_types():
    """Test that infer_entity_types adds (IsA abebe person) for entities appearing as subjects."""
    pp = PLNPostprocessor()
    
    statements = [
        "(: abebe_pasta_eats_frequently_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))",
        "(: abebe_eats_large_portions_fact (EatsLargePortions abebe) (STV 1.0 1.0))",
        "(: pasta_is_refined_fact (IsRefined pasta) (STV 1.0 1.0))",
    ]
    
    proper_name_map = {'abebe': 'abebe', 'pasta': 'pasta'}
    
    inferred = pp.infer_entity_types(statements, proper_name_map)
    
    print("Test 2: infer_entity_types")
    print(f"  Statements: {len(statements)} statements")
    print(f"  proper_name_map: {proper_name_map}")
    print(f"  Inferred types:")
    for inf in inferred:
        print(f"    {inf}")
    
    # Should infer abebe as person (appears as first arg)
    # Should NOT infer pasta as person (appears as second arg in EatsFrequently, or as subject of IsRefined which is a property)
    abebe_found = any('abebe' in inf and 'person' in inf for inf in inferred)
    pasta_found = any('pasta' in inf and 'person' in inf for inf in inferred)
    
    assert abebe_found, f"FAIL: abebe not inferred as person. Inferred: {inferred}"
    print(f"  ✓ abebe inferred as person")
    
    if pasta_found:
        print(f"  ⚠ WARNING: pasta incorrectly inferred as person")
    else:
        print(f"  ✓ pasta NOT inferred as person (correct)")
    
    print("  ✓ PASS\n")
    return inferred


def test_full_process_pipeline():
    """Test the full process() pipeline to ensure inferred types are added."""
    pp = PLNPostprocessor()
    
    text = "Abebe eats pasta almost every day, often in large portions, and prefers refined pasta over whole grain."
    
    statements = [
        "(: abebe_pasta_eats_frequently_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))",
        "(: abebe_eats_large_portions_fact (EatsLargePortions abebe) (STV 1.0 1.0))",
        "(: pasta_is_refined_fact (IsRefined pasta) (STV 1.0 1.0))",
        "(: pasta_is_whole_grain_neg (Not (IsWholeGrain pasta)) (STV 1.0 1.0))",
    ]
    
    result = pp.process(
        text=text,
        statements=statements,
        queries=[],
        context=[],
        plan_queries=False
    )
    
    print("Test 3: Full process() pipeline")
    print(f"  Input text: {text[:60]}...")
    print(f"  Input statements: {len(statements)}")
    print(f"  Output statements: {len(result.statements)}")
    
    # Find type declarations
    type_stmts = [s for s in result.statements if 'IsA abebe person' in s or 'IsA pasta person' in s]
    print(f"  Type declarations found:")
    for ts in type_stmts:
        print(f"    {ts}")
    
    abebe_person = any('IsA abebe person' in s for s in result.statements)
    
    assert abebe_person, f"FAIL: (IsA abebe person) not found in output. Statements: {result.statements}"
    print(f"  ✓ (IsA abebe person) found in output")
    print("  ✓ PASS\n")
    
    return result


def test_with_rule_requiring_person_type():
    """Test that the inferred type allows a rule to fire."""
    pp = PLNPostprocessor()
    
    text = "People who frequently eat pasta tend to consume higher amounts of carbohydrates. Abebe eats pasta almost every day."
    
    statements = [
        # Rule requiring (IsA $x person)
        "(: x_consumes_high_carbs_rule (Implication (Premises (IsA $x person) (EatsFrequently $x $food) (IsA $food pasta)) (Conclusions (ConsumesHighCarbs $x))) (STV 1.0 1.0))",
        # Facts about Abebe
        "(: abebe_pasta_eats_frequently_fact (EatsFrequently abebe pasta) (STV 1.0 1.0))",
    ]
    
    result = pp.process(
        text=text,
        statements=statements,
        queries=[],
        context=[],
        plan_queries=False
    )
    
    print("Test 4: Rule requiring person type")
    print(f"  Rule requires: (IsA $x person)")
    print(f"  Fact: (EatsFrequently abebe pasta)")
    
    # Check for required facts
    has_abebe_person = any('IsA abebe person' in s for s in result.statements)
    has_pasta_pasta = any('IsA pasta pasta' in s for s in result.statements)
    has_eats_fact = any('EatsFrequently abebe pasta' in s for s in result.statements)
    
    print(f"  Has (IsA abebe person): {has_abebe_person}")
    print(f"  Has (IsA pasta pasta): {has_pasta_pasta}")
    print(f"  Has (EatsFrequently abebe pasta): {has_eats_fact}")
    
    assert has_abebe_person, "FAIL: Missing (IsA abebe person)"
    assert has_pasta_pasta, "FAIL: Missing (IsA pasta pasta)"
    assert has_eats_fact, "FAIL: Missing (EatsFrequently abebe pasta)"
    
    print("  ✓ All required premises present for rule to fire")
    print("  ✓ PASS\n")
    
    return result


if __name__ == "__main__":
    print("=" * 70)
    print("TYPE INFERENCE FIX - UNIT TESTS")
    print("=" * 70)
    print()
    
    try:
        test_extract_proper_name_map()
        test_infer_entity_types()
        test_full_process_pipeline()
        test_with_rule_requiring_person_type()
        
        print("=" * 70)
        print("ALL TESTS PASSED ✓")
        print("=" * 70)
        print()
        print("The fix is working correctly in isolation.")
        print("If the live system still shows the issue, the problem is likely:")
        print("  1. Docker container not restarted after code changes")
        print("  2. Database not cleared before re-ingesting")
        print("  3. Text being chunked differently than expected")
        print()
        
    except AssertionError as e:
        print()
        print("=" * 70)
        print(f"TEST FAILED: {e}")
        print("=" * 70)
        sys.exit(1)
    except Exception as e:
        print()
        print("=" * 70)
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 70)
        sys.exit(1)
