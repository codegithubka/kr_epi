"""
Test Script for Steps 1 & 2: Verify ode.py Fixes
Run this after fixing the typo and FOI function in ode.py
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

print("=" * 70)
print("TESTING STEPS 1 & 2: ode.py Critical Fixes")
print("=" * 70)

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    # Try to import from your kr_epi package structure
    from kr_epi.models.ode import SIR, SIS, SEIR, SI, SIRS, _foi
    from kr_epi.models.base import ODEBase
    print("✓ Successfully imported from kr_epi package")
except ImportError as e:
    print(f"Note: Could not import from kr_epi package: {e}")
    print("Attempting alternative import method...")
    
    # Alternative: import directly from files
    import importlib.util
    
    # Load base
    base_path = input("Enter path to base.py (or press Enter for './base.py'): ").strip()
    if not base_path:
        base_path = "./base.py"
    
    spec = importlib.util.spec_from_file_location("base", base_path)
    base_module = importlib.util.module_from_spec(spec)
    sys.modules["kr_epi.models.base"] = base_module
    spec.loader.exec_module(base_module)
    ODEBase = base_module.ODEBase
    
    # Load ode
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
    _foi = ode_module._foi
    
    print("✓ Successfully imported using alternative method")

print("\n" + "=" * 70)
print("TEST 1: Typo Fix - Mixing Type Spelling")
print("=" * 70)

test1_passed = False
try:
    # Test that "frequency" works (not "frequnecy")
    sir_freq = SIR(beta=0.5, gamma=0.1, mixing="frequency")
    sir_dens = SIR(beta=0.5, gamma=0.1, mixing="density")
    
    assert sir_freq.params.mixing == "frequency", "Frequency mixing not set correctly"
    assert sir_dens.params.mixing == "density", "Density mixing not set correctly"
    
    print("✓ PASS: Both 'frequency' and 'density' mixing types work")
    print(f"  - Frequency mixing: {sir_freq.params.mixing}")
    print(f"  - Density mixing: {sir_dens.params.mixing}")
    test1_passed = True
    
except Exception as e:
    print(f"✗ FAIL: {e}")
    print("  Make sure line 6 is: Mixing = Literal['frequency', 'density']")

print("\n" + "=" * 70)
print("TEST 2: FOI Function Exists and Has Correct Signature")
print("=" * 70)

test2_passed = False
try:
    # Check if _foi function exists and has correct parameters
    import inspect
    sig = inspect.signature(_foi)
    params = list(sig.parameters.keys())
    
    expected_params = ['beta', 'I', 'N', 'mixing']
    assert params == expected_params, f"Expected parameters {expected_params}, got {params}"
    
    print("✓ PASS: _foi function has correct signature")
    print(f"  Parameters: {params}")
    test2_passed = True
    
except NameError:
    print("✗ FAIL: _foi function not found")
    print("  Make sure you renamed _foi_frac to _foi")
except Exception as e:
    print(f"✗ FAIL: {e}")

print("\n" + "=" * 70)
print("TEST 3: FOI Calculation - Frequency-Dependent")
print("=" * 70)

test3_passed = False
try:
    # Test frequency-dependent: lambda = beta * I / N
    beta, I, N = 0.5, 0.1, 1.0
    lam_freq = _foi(beta=beta, I=I, N=N, mixing="frequency")
    expected = beta * I / N  # = 0.05
    
    assert np.isclose(lam_freq, expected), f"Expected {expected}, got {lam_freq}"
    print(f"✓ PASS: Frequency-dependent FOI correct")
    print(f"  λ = β×I/N = {beta}×{I}/{N} = {lam_freq}")
    
    # Test that it scales with N
    lam_freq2 = _foi(beta=beta, I=I, N=2.0, mixing="frequency")
    expected2 = beta * I / 2.0  # = 0.025
    
    assert np.isclose(lam_freq2, expected2), f"Expected {expected2}, got {lam_freq2}"
    print(f"  λ = β×I/N = {beta}×{I}/2.0 = {lam_freq2}")
    print(f"✓ PASS: Frequency-dependent FOI scales with N")
    test3_passed = True
    
except Exception as e:
    print(f"✗ FAIL: {e}")

print("\n" + "=" * 70)
print("TEST 4: FOI Calculation - Density-Dependent")
print("=" * 70)

test4_passed = False
try:
    # Test density-dependent: lambda = beta * I
    beta, I, N = 0.5, 0.1, 1.0
    lam_dens = _foi(beta=beta, I=I, N=N, mixing="density")
    expected = beta * I  # = 0.05
    
    assert np.isclose(lam_dens, expected), f"Expected {expected}, got {lam_dens}"
    print(f"✓ PASS: Density-dependent FOI correct")
    print(f"  λ = β×I = {beta}×{I} = {lam_dens}")
    
    # Test that it does NOT scale with N
    lam_dens2 = _foi(beta=beta, I=I, N=2.0, mixing="density")
    
    assert np.isclose(lam_dens2, expected), f"Expected {expected}, got {lam_dens2}"
    print(f"  λ = β×I = {beta}×{I} = {lam_dens2} (N=2.0)")
    print(f"✓ PASS: Density-dependent FOI independent of N")
    test4_passed = True
    
except Exception as e:
    print(f"✗ FAIL: {e}")

print("\n" + "=" * 70)
print("TEST 5: SIR Model Integration - Frequency vs Density")
print("=" * 70)

test5_passed = False
try:
    sir_freq = SIR(beta=0.5, gamma=0.1, mixing="frequency")
    sir_dens = SIR(beta=0.5, gamma=0.1, mixing="density")
    
    y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
    
    t_freq, y_freq = sir_freq.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
    t_dens, y_dens = sir_dens.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
    
    # They should give different results
    if np.allclose(y_freq, y_dens, rtol=0.01):
        print("✗ FAIL: Frequency and density mixing give identical results!")
        print("  This means the FOI function is not being used correctly.")
    else:
        peak_I_freq = y_freq[1].max()
        peak_I_dens = y_dens[1].max()
        
        print("✓ PASS: Frequency and density mixing give different results")
        print(f"  Frequency-dependent peak I: {peak_I_freq:.4f}")
        print(f"  Density-dependent peak I: {peak_I_dens:.4f}")
        print(f"  Relative difference: {abs(peak_I_freq - peak_I_dens)/peak_I_freq * 100:.1f}%")
        test5_passed = True
        
except Exception as e:
    print(f"✗ FAIL: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("TEST 6: Conservation Law - S + I + R = Constant")
print("=" * 70)

test6_passed = False
try:
    sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")
    y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
    t, y = sir.integrate(y0, t_span=(0, 100))
    
    N = y.sum(axis=0)  # Sum over compartments at each time point
    
    if np.allclose(N, 1.0, atol=1e-6):
        print("✓ PASS: Population is conserved (S + I + R = 1)")
        print(f"  Initial: N = {N[0]:.8f}")
        print(f"  Final:   N = {N[-1]:.8f}")
        print(f"  Max deviation: {abs(N - 1.0).max():.2e}")
        test6_passed = True
    else:
        print(f"✗ FAIL: Population not conserved")
        print(f"  Initial: N = {N[0]:.8f}")
        print(f"  Final:   N = {N[-1]:.8f}")
        print(f"  Max deviation: {abs(N - 1.0).max():.2e}")
        
except Exception as e:
    print(f"✗ FAIL: {e}")

print("\n" + "=" * 70)
print("TEST 7: All Model Classes Work")
print("=" * 70)

test7_passed = True
models_to_test = [
    ("SI", SI, {"beta": 0.5}, {"S": 0.99, "I": 0.01}),
    ("SIS", SIS, {"beta": 0.3, "gamma": 0.1}, {"S": 0.99, "I": 0.01}),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0.1}, {"S": 0.99, "I": 0.01, "R": 0.0}),
    ("SIRS", SIRS, {"beta": 0.5, "gamma": 0.1, "omega": 0.05}, {"S": 0.99, "I": 0.01, "R": 0.0}),
    ("SEIR", SEIR, {"beta": 0.5, "gamma": 0.1, "sigma": 0.2}, {"S": 0.99, "E": 0.005, "I": 0.005, "R": 0.0}),
]

for model_name, ModelClass, params, y0 in models_to_test:
    try:
        model = ModelClass(**params, mixing="frequency")
        t, y = model.integrate(y0, t_span=(0, 50))
        print(f"  ✓ {model_name}: Integrated successfully ({len(t)} time points)")
    except Exception as e:
        print(f"  ✗ {model_name}: FAILED - {e}")
        test7_passed = False

if test7_passed:
    print("✓ PASS: All models work correctly")

print("\n" + "=" * 70)
print("OPTIONAL: Create Visualization")
print("=" * 70)

create_plot = input("Create comparison plot? (y/n): ").strip().lower()

if create_plot == 'y':
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Frequency vs Density SIR
        sir_freq = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        sir_dens = SIR(beta=0.5, gamma=0.1, mixing="density")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        
        t_freq, y_freq = sir_freq.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        t_dens, y_dens = sir_dens.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        
        ax = axes[0, 0]
        ax.plot(t_freq, y_freq[1], 'b-', linewidth=2, label='Frequency')
        ax.plot(t_dens, y_dens[1], 'r--', linewidth=2, label='Density')
        ax.set_xlabel('Time')
        ax.set_ylabel('Infected (I)')
        ax.set_title('SIR: Frequency vs Density Mixing')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Full SIR dynamics (frequency)
        ax = axes[0, 1]
        ax.plot(t_freq, y_freq[0], 'b-', linewidth=2, label='S')
        ax.plot(t_freq, y_freq[1], 'r-', linewidth=2, label='I')
        ax.plot(t_freq, y_freq[2], 'g-', linewidth=2, label='R')
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_title('SIR Dynamics (Frequency-Dependent)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: SIS endemic equilibrium
        sis = SIS(beta=0.3, gamma=0.1, mixing="frequency")
        y0_sis = {"S": 0.99, "I": 0.01}
        t_sis, y_sis = sis.integrate(y0_sis, t_span=(0, 200), t_eval=np.linspace(0, 200, 300))
        
        ax = axes[1, 0]
        ax.plot(t_sis, y_sis[0], 'b-', linewidth=2, label='S')
        ax.plot(t_sis, y_sis[1], 'r-', linewidth=2, label='I')
        ax.set_xlabel('Time')
        ax.set_ylabel('Fraction')
        ax.set_title('SIS: Approach to Endemic Equilibrium')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: SEIR vs SIR comparison
        seir = SEIR(beta=0.5, gamma=0.1, sigma=0.2, mixing="frequency")
        sir_comp = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        
        y0_seir = {"S": 0.99, "E": 0.005, "I": 0.005, "R": 0.0}
        y0_sir = {"S": 0.99, "I": 0.01, "R": 0.0}
        
        t_seir, y_seir = seir.integrate(y0_seir, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        t_sir, y_sir = sir_comp.integrate(y0_sir, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        
        ax = axes[1, 1]
        ax.plot(t_seir, y_seir[2], 'b-', linewidth=2, label='SEIR (I)')
        ax.plot(t_sir, y_sir[1], 'r--', linewidth=2, label='SIR (I)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Infected (I)')
        ax.set_title('SEIR vs SIR: Effect of Latent Period')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = input("Save plot to (press Enter for './test_results.png'): ").strip()
        if not output_path:
            output_path = './test_results.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Could not create plot: {e}")

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

all_tests = [
    ("Test 1: Typo Fix", test1_passed),
    ("Test 2: FOI Function Signature", test2_passed),
    ("Test 3: Frequency-Dependent FOI", test3_passed),
    ("Test 4: Density-Dependent FOI", test4_passed),
    ("Test 5: Different Mixing Results", test5_passed),
    ("Test 6: Conservation Law", test6_passed),
    ("Test 7: All Models Work", test7_passed),
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
    print("\nSteps 1 & 2 are complete!")
    print("Your ode.py file is now correctly fixed.")
    print("\n📌 NEXT: Ready for Step 3 - Adding R₀ methods")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nPlease review the failed tests above and make corrections.")
    print("Common issues:")
    print("  - Make sure the typo is fixed on line 6")
    print("  - Make sure _foi_frac is renamed to _foi with 4 parameters")
    print("  - Make sure all models calculate N before calling _foi")
    print("  - Make sure all models pass N to _foi")

print("=" * 70)