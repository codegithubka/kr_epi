"""
Test Script for Step 6: Equilibrium Analysis
Tests endemic equilibria, final sizes, and vaccination thresholds.
"""

import numpy as np
import sys
import os

print("=" * 70)
print("TESTING STEP 6: Equilibrium Analysis")
print("=" * 70)

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from kr_epi.models.ode import SIR, SIS, SIRS
    from kr_epi.analysis.equilibria import (
        sis_equilibria, sirs_equilibria, sir_final_size,
        critical_vaccination_coverage, sir_attack_rate
    )
    print("✓ Successfully imported modules")
except ImportError as e:
    print(f"Import error: {e}")
    print("Attempting alternative import...")
    
    import importlib.util
    
    # Try to import what we can
    base_path = input("Enter path to base.py (or press Enter to skip): ").strip()
    if base_path:
        spec = importlib.util.spec_from_file_location("base", base_path)
        base_module = importlib.util.module_from_spec(spec)
        sys.modules["kr_epi.models.base"] = base_module
        spec.loader.exec_module(base_module)
    
    ode_path = input("Enter path to ode.py (or press Enter to skip): ").strip()
    if ode_path:
        spec = importlib.util.spec_from_file_location("ode", ode_path)
        ode_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ode_module)
        SIR = ode_module.SIR
        SIS = ode_module.SIS
        SIRS = ode_module.SIRS

print("\n" + "=" * 70)
print("TEST 1: SIS Endemic Equilibrium (Analysis Module)")
print("=" * 70)

test1_passed = True

try:
    beta, gamma = 0.3, 0.1  # R0 = 3
    S_star, I_star = sis_equilibria(beta, gamma)
    
    expected_S = 1.0 / 3.0  # 1/R0
    expected_I = 2.0 / 3.0  # 1 - 1/R0
    
    if np.isclose(S_star, expected_S, rtol=0.01):
        print(f"  ✓ S* = {S_star:.4f} (expected {expected_S:.4f})")
    else:
        print(f"  ✗ S* = {S_star:.4f}, expected {expected_S:.4f}")
        test1_passed = False
    
    if np.isclose(I_star, expected_I, rtol=0.01):
        print(f"  ✓ I* = {I_star:.4f} (expected {expected_I:.4f})")
    else:
        print(f"  ✗ I* = {I_star:.4f}, expected {expected_I:.4f}")
        test1_passed = False
    
    # Test that R0 <= 1 gives DFE
    S_dfe, I_dfe = sis_equilibria(beta=0.05, gamma=0.1)  # R0 = 0.5
    if np.isclose(S_dfe, 1.0) and np.isclose(I_dfe, 0.0):
        print(f"  ✓ R₀ < 1 gives disease-free equilibrium")
    else:
        print(f"  ✗ R₀ < 1 should give DFE, got S={S_dfe:.3f}, I={I_dfe:.3f}")
        test1_passed = False
    
    if test1_passed:
        print("✓ PASS: SIS equilibria correct")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    test1_passed = False

print("\n" + "=" * 70)
print("TEST 2: SIR Final Size (Analysis Module)")
print("=" * 70)

test2_passed = True

try:
    # R0 = 2, s0 = 0.99
    R0 = 2.0
    s0 = 0.99
    s_inf = sir_final_size(s0, R0)
    
    # For R0=2, roughly 80% attack rate
    attack_rate = 1 - s_inf
    
    print(f"  Initial susceptible: s₀ = {s0:.4f}")
    print(f"  Final susceptible: s_∞ = {s_inf:.4f}")
    print(f"  Attack rate: {attack_rate:.1%}")
    
    # Check that s_inf satisfies the implicit equation
    # s_inf should equal s0 * exp(-R0 * (1 - s_inf))
    expected = s0 * np.exp(-R0 * (1 - s_inf))
    if np.isclose(s_inf, expected, rtol=1e-6):
        print(f"  ✓ Final size satisfies implicit equation")
    else:
        print(f"  ✗ Implicit equation not satisfied: {s_inf:.6f} vs {expected:.6f}")
        test2_passed = False
    
    # Test with different R0 values
    print("\n  Testing various R₀ values:")
    for R0_test in [1.5, 3.0, 5.0]:
        s_inf_test = sir_final_size(0.99, R0_test)
        attack_test = 1 - s_inf_test
        print(f"    R₀ = {R0_test:.1f}: attack rate = {attack_test:.1%}")
    
    if test2_passed:
        print("✓ PASS: SIR final size calculations correct")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    test2_passed = False

print("\n" + "=" * 70)
print("TEST 3: SIRS Endemic Equilibrium (Analysis Module)")
print("=" * 70)

test3_passed = True

try:
    beta, gamma, omega = 0.5, 0.1, 0.05
    S_star, I_star, R_star = sirs_equilibria(beta, gamma, omega)
    
    R0 = beta / gamma  # = 5.0
    
    print(f"  R₀ = {R0:.2f}")
    print(f"  S* = {S_star:.4f}")
    print(f"  I* = {I_star:.4f}")
    print(f"  R* = {R_star:.4f}")
    
    # Check conservation
    total = S_star + I_star + R_star
    if np.isclose(total, 1.0, atol=1e-6):
        print(f"  ✓ S* + I* + R* = {total:.6f}")
    else:
        print(f"  ✗ Not conserved: S* + I* + R* = {total:.6f}")
        test3_passed = False
    
    # Check that S* = 1/R0
    expected_S = 1.0 / R0
    if np.isclose(S_star, expected_S, rtol=0.01):
        print(f"  ✓ S* = 1/R₀")
    else:
        print(f"  ✗ S* ≠ 1/R₀: {S_star:.4f} vs {expected_S:.4f}")
        test3_passed = False
    
    if test3_passed:
        print("✓ PASS: SIRS equilibria correct")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    test3_passed = False

print("\n" + "=" * 70)
print("TEST 4: Vaccination Threshold")
print("=" * 70)

test4_passed = True

try:
    # Test different R0 values
    test_cases = [
        (2.0, 0.5),    # R0=2 -> p_c=50%
        (3.0, 0.667),  # R0=3 -> p_c=66.7%
        (5.0, 0.8),    # R0=5 -> p_c=80%
        (10.0, 0.9),   # R0=10 -> p_c=90%
    ]
    
    print("\n  Vaccination coverage needed for herd immunity:")
    for R0, expected_p_c in test_cases:
        p_c = critical_vaccination_coverage(R0)
        
        if np.isclose(p_c, expected_p_c, rtol=0.01):
            print(f"    R₀ = {R0:4.1f} → p_c = {p_c:.1%} ✓")
        else:
            print(f"    R₀ = {R0:4.1f} → p_c = {p_c:.1%} (expected {expected_p_c:.1%}) ✗")
            test4_passed = False
    
    # Test that R0 <= 1 raises error
    try:
        critical_vaccination_coverage(0.5)
        print("  ✗ Should raise error for R₀ < 1")
        test4_passed = False
    except ValueError:
        print("  ✓ Correctly raises error for R₀ < 1")
    
    if test4_passed:
        print("✓ PASS: Vaccination thresholds correct")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    test4_passed = False

print("\n" + "=" * 70)
print("TEST 5: Equilibrium Matches Simulation (SIS)")
print("=" * 70)

test5_passed = True

try:
    # Create SIS model and simulate to equilibrium
    sis = SIS(beta=0.3, gamma=0.1, mixing="frequency")
    R0 = sis.R0()
    
    # Get predicted equilibrium
    S_pred, I_pred = sis_equilibria(beta=0.3, gamma=0.1)
    
    # Simulate
    y0 = {"S": 0.99, "I": 0.01}
    t, y = sis.integrate(y0, t_span=(0, 500))
    
    S_sim = y[0, -1]
    I_sim = y[1, -1]
    
    print(f"  Predicted: S* = {S_pred:.4f}, I* = {I_pred:.4f}")
    print(f"  Simulated: S  = {S_sim:.4f}, I  = {I_sim:.4f}")
    
    if np.isclose(S_sim, S_pred, rtol=0.02) and np.isclose(I_sim, I_pred, rtol=0.02):
        print("  ✓ Simulation converges to predicted equilibrium")
    else:
        print("  ✗ Simulation does not match prediction")
        test5_passed = False
    
    if test5_passed:
        print("✓ PASS: SIS equilibrium matches simulation")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test5_passed = False

print("\n" + "=" * 70)
print("TEST 6: Final Size Matches Simulation (SIR)")
print("=" * 70)

test6_passed = True

try:
    # Create SIR model
    sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")
    R0 = sir.R0()
    
    # Get predicted final size
    s0 = 0.99
    s_inf_pred = sir_final_size(s0, R0)
    
    # Simulate
    y0 = {"S": s0, "I": 0.01, "R": 0.0}
    t, y = sir.integrate(y0, t_span=(0, 500))
    
    s_inf_sim = y[0, -1]
    
    print(f"  R₀ = {R0:.2f}")
    print(f"  Predicted: s_∞ = {s_inf_pred:.4f}")
    print(f"  Simulated: s   = {s_inf_sim:.4f}")
    print(f"  Difference: {abs(s_inf_pred - s_inf_sim):.6f}")
    
    if np.isclose(s_inf_sim, s_inf_pred, rtol=0.01):
        print("  ✓ Simulation matches predicted final size")
    else:
        print("  ✗ Simulation does not match prediction")
        test6_passed = False
    
    if test6_passed:
        print("✓ PASS: SIR final size matches simulation")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    test6_passed = False

print("\n" + "=" * 70)
print("TEST 7: Attack Rate Analysis")
print("=" * 70)

test7_passed = True

try:
    print("\n  Attack rates for different R₀:")
    for R0 in [1.2, 2.0, 3.0, 5.0]:
        attack = sir_attack_rate(s0=0.99, R0=R0)
        print(f"    R₀ = {R0:.1f}: {attack:.1%} infected")
    
    # Test that higher R0 -> higher attack rate
    attack_low = sir_attack_rate(0.99, 2.0)
    attack_high = sir_attack_rate(0.99, 5.0)
    
    if attack_high > attack_low:
        print("  ✓ Higher R₀ gives higher attack rate")
    else:
        print("  ✗ Higher R₀ should give higher attack rate")
        test7_passed = False
    
    if test7_passed:
        print("✓ PASS: Attack rate analysis correct")
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    test7_passed = False

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

all_tests = [
    ("Test 1: SIS Equilibria", test1_passed),
    ("Test 2: SIR Final Size", test2_passed),
    ("Test 3: SIRS Equilibria", test3_passed),
    ("Test 4: Vaccination Threshold", test4_passed),
    ("Test 5: SIS Simulation Match", test5_passed),
    ("Test 6: SIR Simulation Match", test6_passed),
    ("Test 7: Attack Rate Analysis", test7_passed),
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
    print("\nStep 6 is complete!")
    print("Equilibrium analysis is working correctly.")
    print("\n📌 NEXT: Ready for Step 7 - Documentation")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nThis is expected if you haven't updated your model classes yet.")
    print("The analysis module functions should work independently.")

print("=" * 70)