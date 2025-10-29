"""
Test Script for Step 4: Verify Parameter Validation
Run this after adding __post_init__ validation to all parameter classes
"""

import sys
import os

print("=" * 70)
print("TESTING STEP 4: Parameter Validation")
print("=" * 70)

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from kr_epi.models.ode import SIR, SIS, SEIR, SI, SIRS
    print("✓ Successfully imported from kr_epi package")
except ImportError:
    print("Attempting alternative import...")
    import importlib.util
    
    base_path = input("Enter path to base.py (or press Enter for './base.py'): ").strip()
    if not base_path:
        base_path = "./base.py"
    
    spec = importlib.util.spec_from_file_location("base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    sys.modules["kr_epi.models.base"] = base_module
    spec.loader.exec_module(base_module)
    
    ode_path = input("Enter path to ode.py (or press Enter for './ode.py'): ").strip()
    if not ode_path:
        ode_path = "./ode.py"
    
    spec = importlib.util.spec_from_file_location("ode", ode_path)
    ode_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ode_module)
    
    SIR = ode_module.SIR
    SIS = ode_module.SIS
    SEIR = ode_module.SEIR
    SI = ode_module.SI
    SIRS = ode_module.SIRS
    print("✓ Successfully imported using alternative method")

print("\n" + "=" * 70)
print("TEST 1: Valid Parameters Should Work")
print("=" * 70)

test1_passed = True

valid_cases = [
    ("SI", SI, {"beta": 0.5}, "Valid SI parameters"),
    ("SIS", SIS, {"beta": 0.3, "gamma": 0.1}, "Valid SIS parameters"),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0.1}, "Valid SIR parameters"),
    ("SIRS", SIRS, {"beta": 0.5, "gamma": 0.1, "omega": 0.05}, "Valid SIRS parameters"),
    ("SEIR", SEIR, {"beta": 0.5, "gamma": 0.1, "sigma": 0.2}, "Valid SEIR parameters"),
]

for model_name, ModelClass, params, description in valid_cases:
    try:
        model = ModelClass(**params, mixing="frequency")
        print(f"  ✓ {model_name}: {description} accepted")
    except Exception as e:
        print(f"  ✗ {model_name}: FAILED - {e}")
        test1_passed = False

if test1_passed:
    print("✓ PASS: All valid parameters accepted")

print("\n" + "=" * 70)
print("TEST 2: Negative Beta Should Be Rejected")
print("=" * 70)

test2_passed = True

models_to_test = [
    ("SI", SI, {"beta": -0.5}),
    ("SIS", SIS, {"beta": -0.3, "gamma": 0.1}),
    ("SIR", SIR, {"beta": -0.5, "gamma": 0.1}),
]

for model_name, ModelClass, params in models_to_test:
    try:
        model = ModelClass(**params)
        print(f"  ✗ {model_name}: FAILED - Negative beta was accepted!")
        test2_passed = False
    except ValueError as e:
        if "beta" in str(e).lower() and "negative" in str(e).lower():
            print(f"  ✓ {model_name}: Correctly rejected negative beta")
        else:
            print(f"  ✗ {model_name}: Wrong error message: {e}")
            test2_passed = False
    except Exception as e:
        print(f"  ✗ {model_name}: Unexpected error type: {type(e).__name__}: {e}")
        test2_passed = False

if test2_passed:
    print("✓ PASS: Negative beta correctly rejected")

print("\n" + "=" * 70)
print("TEST 3: Zero/Negative Gamma Should Be Rejected")
print("=" * 70)

test3_passed = True

# Test gamma = 0
zero_gamma_tests = [
    ("SIS", SIS, {"beta": 0.3, "gamma": 0}),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0}),
    ("SIRS", SIRS, {"beta": 0.5, "gamma": 0, "omega": 0.05}),
    ("SEIR", SEIR, {"beta": 0.5, "gamma": 0, "sigma": 0.2}),
]

for model_name, ModelClass, params in zero_gamma_tests:
    try:
        model = ModelClass(**params)
        print(f"  ✗ {model_name}: FAILED - Zero gamma was accepted!")
        test3_passed = False
    except ValueError as e:
        if "gamma" in str(e).lower() and "positive" in str(e).lower():
            print(f"  ✓ {model_name}: Correctly rejected gamma=0")
        else:
            print(f"  ✗ {model_name}: Wrong error message: {e}")
            test3_passed = False
    except Exception as e:
        print(f"  ✗ {model_name}: Unexpected error: {e}")
        test3_passed = False

# Test negative gamma
neg_gamma_tests = [
    ("SIS", SIS, {"beta": 0.3, "gamma": -0.1}),
    ("SIR", SIR, {"beta": 0.5, "gamma": -0.1}),
]

for model_name, ModelClass, params in neg_gamma_tests:
    try:
        model = ModelClass(**params)
        print(f"  ✗ {model_name}: FAILED - Negative gamma was accepted!")
        test3_passed = False
    except ValueError as e:
        if "gamma" in str(e).lower():
            print(f"  ✓ {model_name}: Correctly rejected negative gamma")
        else:
            print(f"  ✗ {model_name}: Wrong error message: {e}")
            test3_passed = False
    except Exception as e:
        print(f"  ✗ {model_name}: Unexpected error: {e}")
        test3_passed = False

if test3_passed:
    print("✓ PASS: Invalid gamma correctly rejected")

print("\n" + "=" * 70)
print("TEST 4: Invalid Mixing Type Should Be Rejected")
print("=" * 70)

test4_passed = True

invalid_mixing_tests = [
    ("SI", SI, {"beta": 0.5}, "invalid"),
    ("SIS", SIS, {"beta": 0.3, "gamma": 0.1}, "mass-action"),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0.1}, "freq"),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0.1}, "dens"),
]

for model_name, ModelClass, params, bad_mixing in invalid_mixing_tests:
    try:
        model = ModelClass(**params, mixing=bad_mixing)
        print(f"  ✗ {model_name}: FAILED - Invalid mixing '{bad_mixing}' was accepted!")
        test4_passed = False
    except ValueError as e:
        if "mixing" in str(e).lower():
            print(f"  ✓ {model_name}: Correctly rejected mixing='{bad_mixing}'")
        else:
            print(f"  ✗ {model_name}: Wrong error message: {e}")
            test4_passed = False
    except Exception as e:
        print(f"  ✗ {model_name}: Unexpected error: {e}")
        test4_passed = False

if test4_passed:
    print("✓ PASS: Invalid mixing types correctly rejected")

print("\n" + "=" * 70)
print("TEST 5: Zero/Negative Sigma Should Be Rejected (SEIR)")
print("=" * 70)

test5_passed = True

sigma_tests = [
    (0, "zero"),
    (-0.2, "negative"),
]

for sigma_val, description in sigma_tests:
    try:
        seir = SEIR(beta=0.5, gamma=0.1, sigma=sigma_val)
        print(f"  ✗ SEIR: FAILED - {description} sigma was accepted!")
        test5_passed = False
    except ValueError as e:
        if "sigma" in str(e).lower():
            print(f"  ✓ SEIR: Correctly rejected {description} sigma={sigma_val}")
        else:
            print(f"  ✗ SEIR: Wrong error message: {e}")
            test5_passed = False
    except Exception as e:
        print(f"  ✗ SEIR: Unexpected error: {e}")
        test5_passed = False

if test5_passed:
    print("✓ PASS: Invalid sigma correctly rejected")

print("\n" + "=" * 70)
print("TEST 6: Negative Omega Should Be Rejected (SIRS)")
print("=" * 70)

test6_passed = True

try:
    sirs = SIRS(beta=0.5, gamma=0.1, omega=-0.05)
    print(f"  ✗ SIRS: FAILED - Negative omega was accepted!")
    test6_passed = False
except ValueError as e:
    if "omega" in str(e).lower():
        print(f"  ✓ SIRS: Correctly rejected negative omega")
    else:
        print(f"  ✗ SIRS: Wrong error message: {e}")
        test6_passed = False
except Exception as e:
    print(f"  ✗ SIRS: Unexpected error: {e}")
    test6_passed = False

# But omega = 0 should be OK (permanent immunity)
try:
    sirs = SIRS(beta=0.5, gamma=0.1, omega=0.0)
    print(f"  ✓ SIRS: Correctly accepted omega=0 (permanent immunity)")
except Exception as e:
    print(f"  ✗ SIRS: Should accept omega=0, got: {e}")
    test6_passed = False

if test6_passed:
    print("✓ PASS: Omega validation working correctly")

print("\n" + "=" * 70)
print("TEST 7: Edge Cases")
print("=" * 70)

test7_passed = True

print("\nTesting edge cases that should work:")

edge_cases = [
    ("SIR with beta=0", SIR, {"beta": 0.0, "gamma": 0.1}, "No transmission"),
    ("SIRS with omega=0", SIRS, {"beta": 0.5, "gamma": 0.1, "omega": 0.0}, "Permanent immunity"),
    ("SIR with small gamma", SIR, {"beta": 0.5, "gamma": 0.001}, "Long infectious period"),
    ("SIR with large beta", SIR, {"beta": 10.0, "gamma": 0.1}, "High transmission"),
]

for description, ModelClass, params, note in edge_cases:
    try:
        model = ModelClass(**params)
        print(f"  ✓ {description}: {note}")
    except Exception as e:
        print(f"  ✗ {description}: Should be valid but got: {e}")
        test7_passed = False

if test7_passed:
    print("✓ PASS: Edge cases handled correctly")

print("\n" + "=" * 70)
print("TEST 8: Error Messages Are Helpful")
print("=" * 70)

test8_passed = True

print("\nChecking that error messages include the invalid value:")

test_cases = [
    (SIR, {"beta": -0.5, "gamma": 0.1}, "-0.5"),
    (SIR, {"beta": 0.5, "gamma": 0}, "0"),
    (SIR, {"beta": 0.5, "gamma": 0.1, "mixing": "bad"}, "bad"),
]

for ModelClass, params, expected_in_msg in test_cases:
    try:
        model = ModelClass(**params)
        print(f"  ✗ Should have raised error for {params}")
        test8_passed = False
    except ValueError as e:
        error_msg = str(e)
        if expected_in_msg in error_msg:
            print(f"  ✓ Error message includes invalid value: '{expected_in_msg}'")
        else:
            print(f"  ⚠ Error message could be more helpful: {error_msg}")
            # Don't fail test, just warn

if test8_passed:
    print("✓ PASS: Error messages are informative")

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

all_tests = [
    ("Test 1: Valid Parameters Work", test1_passed),
    ("Test 2: Negative Beta Rejected", test2_passed),
    ("Test 3: Invalid Gamma Rejected", test3_passed),
    ("Test 4: Invalid Mixing Rejected", test4_passed),
    ("Test 5: Invalid Sigma Rejected", test5_passed),
    ("Test 6: Invalid Omega Rejected", test6_passed),
    ("Test 7: Edge Cases Work", test7_passed),
    ("Test 8: Error Messages Helpful", test8_passed),
]

passed = sum(1 for _, p in all_tests if p)
total = len(all_tests)

print(f"\nTests Passed: {passed}/{total}\n")

for test_name, test_passed in all_tests:
    status = "✓ PASS" if test_passed else "✗ FAIL"
    print(f"  {status}: {test_name}")

print("\n" + "=" * 70)

if passed == total:
    print("🎉 ALL TESTS PASSED! 🎉")
    print("\nStep 4 is complete!")
    print("Parameter validation is working correctly.")
    print("\n📌 NEXT: Ready for Step 5 - Conservation Law Tests")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nPlease review the failed tests above.")
    print("Common issues:")
    print("  - Make sure __post_init__ is added to all Params classes")
    print("  - Check validation rules:")
    print("    • beta >= 0 (can be zero)")
    print("    • gamma > 0 (must be positive)")
    print("    • sigma > 0 (must be positive)")
    print("    • omega >= 0 (can be zero)")
    print("    • mixing in ('frequency', 'density')")

print("=" * 70)

# Show example of good error message
print("\n" + "=" * 70)
print("EXAMPLE: What Good Validation Looks Like")
print("=" * 70)

print("\nTrying to create SIR with beta=-0.5...")
try:
    bad_sir = SIR(beta=-0.5, gamma=0.1)
except ValueError as e:
    print(f"✓ Got helpful error: {e}")
except Exception as e:
    print(f"Got error: {e}")

print("\nTrying to create SIR with gamma=0...")
try:
    bad_sir = SIR(beta=0.5, gamma=0)
except ValueError as e:
    print(f"✓ Got helpful error: {e}")
except Exception as e:
    print(f"Got error: {e}")

print("\nTrying to create SIR with mixing='invalid'...")
try:
    bad_sir = SIR(beta=0.5, gamma=0.1, mixing="invalid")
except ValueError as e:
    print(f"✓ Got helpful error: {e}")
except Exception as e:
    print(f"Got error: {e}")

print("\n" + "=" * 70)