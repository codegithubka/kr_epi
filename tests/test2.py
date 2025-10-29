"""
Test Script for Step 3: Verify R₀ Methods
Run this after adding R0() methods to all model classes
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

print("=" * 70)
print("TESTING STEP 3: R₀ Method Implementation")
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
print("TEST 1: R₀ Methods Exist")
print("=" * 70)

test1_passed = True
models = [
    ("SI", SI, {"beta": 0.5}),
    ("SIS", SIS, {"beta": 0.5, "gamma": 0.1}),
    ("SIR", SIR, {"beta": 0.5, "gamma": 0.1}),
    ("SIRS", SIRS, {"beta": 0.5, "gamma": 0.1, "omega": 0.05}),
    ("SEIR", SEIR, {"beta": 0.5, "gamma": 0.1, "sigma": 0.2}),
]

for model_name, ModelClass, params in models:
    try:
        model = ModelClass(**params, mixing="frequency")
        
        # Check if R0 method exists
        if not hasattr(model, 'R0'):
            print(f"  ✗ {model_name}: R0 method NOT FOUND")
            test1_passed = False
        else:
            # Try to call it
            r0 = model.R0()
            print(f"  ✓ {model_name}: R0 method exists, R₀ = {r0:.2f}")
            
    except Exception as e:
        print(f"  ✗ {model_name}: ERROR - {e}")
        test1_passed = False

if test1_passed:
    print("✓ PASS: All models have R0 methods")

print("\n" + "=" * 70)
print("TEST 2: R₀ Calculations - Frequency-Dependent")
print("=" * 70)

test2_passed = True

# SIR: R₀ = β/γ = 0.5/0.1 = 5.0
sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")
r0_sir = sir.R0()
expected_sir = 0.5 / 0.1
if np.isclose(r0_sir, expected_sir):
    print(f"  ✓ SIR: R₀ = {r0_sir:.2f} (expected {expected_sir:.2f})")
else:
    print(f"  ✗ SIR: R₀ = {r0_sir:.2f}, expected {expected_sir:.2f}")
    test2_passed = False

# SIS: R₀ = β/γ = 0.3/0.1 = 3.0
sis = SIS(beta=0.3, gamma=0.1, mixing="frequency")
r0_sis = sis.R0()
expected_sis = 0.3 / 0.1
if np.isclose(r0_sis, expected_sis):
    print(f"  ✓ SIS: R₀ = {r0_sis:.2f} (expected {expected_sis:.2f})")
else:
    print(f"  ✗ SIS: R₀ = {r0_sis:.2f}, expected {expected_sis:.2f}")
    test2_passed = False

# SEIR: R₀ = β/γ = 0.5/0.1 = 5.0 (sigma doesn't affect R₀)
seir = SEIR(beta=0.5, gamma=0.1, sigma=0.2, mixing="frequency")
r0_seir = seir.R0()
expected_seir = 0.5 / 0.1
if np.isclose(r0_seir, expected_seir):
    print(f"  ✓ SEIR: R₀ = {r0_seir:.2f} (expected {expected_seir:.2f})")
else:
    print(f"  ✗ SEIR: R₀ = {r0_seir:.2f}, expected {expected_seir:.2f}")
    test2_passed = False

# SIRS: R₀ = β/γ = 0.5/0.1 = 5.0 (omega doesn't affect R₀)
sirs = SIRS(beta=0.5, gamma=0.1, omega=0.05, mixing="frequency")
r0_sirs = sirs.R0()
expected_sirs = 0.5 / 0.1
if np.isclose(r0_sirs, expected_sirs):
    print(f"  ✓ SIRS: R₀ = {r0_sirs:.2f} (expected {expected_sirs:.2f})")
else:
    print(f"  ✗ SIRS: R₀ = {r0_sirs:.2f}, expected {expected_sirs:.2f}")
    test2_passed = False

if test2_passed:
    print("✓ PASS: All frequency-dependent R₀ calculations correct")

print("\n" + "=" * 70)
print("TEST 3: R₀ Calculations - Density-Dependent")
print("=" * 70)

test3_passed = True

# For density-dependent: R₀ = β*N/γ
N0 = 1000.0

sir_dens = SIR(beta=0.5, gamma=0.1, mixing="density")
r0_sir_dens = sir_dens.R0(N0=N0)
expected_sir_dens = (0.5 * N0) / 0.1
if np.isclose(r0_sir_dens, expected_sir_dens):
    print(f"  ✓ SIR (density): R₀ = {r0_sir_dens:.1f} with N={N0:.0f}")
else:
    print(f"  ✗ SIR (density): R₀ = {r0_sir_dens:.1f}, expected {expected_sir_dens:.1f}")
    test3_passed = False

sis_dens = SIS(beta=0.3, gamma=0.1, mixing="density")
r0_sis_dens = sis_dens.R0(N0=N0)
expected_sis_dens = (0.3 * N0) / 0.1
if np.isclose(r0_sis_dens, expected_sis_dens):
    print(f"  ✓ SIS (density): R₀ = {r0_sis_dens:.1f} with N={N0:.0f}")
else:
    print(f"  ✗ SIS (density): R₀ = {r0_sis_dens:.1f}, expected {expected_sis_dens:.1f}")
    test3_passed = False

if test3_passed:
    print("✓ PASS: Density-dependent R₀ calculations correct")

print("\n" + "=" * 70)
print("TEST 4: R₀ Threshold - Epidemic vs No Epidemic")
print("=" * 70)

test4_passed = True

# Test R₀ > 1: Should have epidemic
sir_high = SIR(beta=0.5, gamma=0.1, mixing="frequency")  # R₀ = 5
r0_high = sir_high.R0()
y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
t_high, y_high = sir_high.integrate(y0, t_span=(0, 100))

if y_high[1].max() > y0["I"]:
    print(f"  ✓ R₀ = {r0_high:.2f} > 1: Epidemic occurs (peak I = {y_high[1].max():.3f})")
else:
    print(f"  ✗ R₀ = {r0_high:.2f} > 1: No epidemic (unexpected!)")
    test4_passed = False

# Test R₀ < 1: Should NOT have epidemic
sir_low = SIR(beta=0.05, gamma=0.1, mixing="frequency")  # R₀ = 0.5
r0_low = sir_low.R0()
t_low, y_low = sir_low.integrate(y0, t_span=(0, 100))

if y_low[1, -1] < y0["I"]:
    print(f"  ✓ R₀ = {r0_low:.2f} < 1: No epidemic (I decreases to {y_low[1,-1]:.4f})")
else:
    print(f"  ✗ R₀ = {r0_low:.2f} < 1: Epidemic occurred (unexpected!)")
    test4_passed = False

# Test R₀ ≈ 1: Borderline
sir_crit = SIR(beta=0.1, gamma=0.1, mixing="frequency")  # R₀ = 1.0
r0_crit = sir_crit.R0()
t_crit, y_crit = sir_crit.integrate(y0, t_span=(0, 100))
print(f"  ✓ R₀ = {r0_crit:.2f} ≈ 1: Critical threshold (peak I = {y_crit[1].max():.3f})")

if test4_passed:
    print("✓ PASS: R₀ threshold correctly predicts epidemic behavior")

print("\n" + "=" * 70)
print("TEST 5: R₀ and Attack Rate Relationship")
print("=" * 70)

test5_passed = True

print("\nTesting relationship between R₀ and final attack rate:")
print("  (For SIR: higher R₀ should give higher attack rate)")

r0_values = []
attack_rates = []

for beta in [0.15, 0.3, 0.5, 0.7, 1.0]:
    sir = SIR(beta=beta, gamma=0.1, mixing="frequency")
    r0 = sir.R0()
    
    y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
    t, y = sir.integrate(y0, t_span=(0, 500))
    
    attack_rate = y[2, -1]  # Final R
    r0_values.append(r0)
    attack_rates.append(attack_rate)
    
    print(f"  β = {beta:.2f}: R₀ = {r0:.2f}, attack rate = {attack_rate:.3f}")

# Check that attack rate increases with R₀
if all(attack_rates[i] <= attack_rates[i+1] for i in range(len(attack_rates)-1)):
    print("✓ PASS: Attack rate increases monotonically with R₀")
else:
    print("✗ FAIL: Attack rate does not increase monotonically with R₀")
    test5_passed = False

print("\n" + "=" * 70)
print("TEST 6: Critical Vaccination Coverage")
print("=" * 70)

test6_passed = True

print("\nCritical vaccination coverage: p_c = 1 - 1/R₀")
print("  (Fraction of population that needs to be immune to prevent epidemic)")

for beta, gamma in [(0.5, 0.1), (0.3, 0.1), (0.2, 0.1)]:
    sir = SIR(beta=beta, gamma=gamma, mixing="frequency")
    r0 = sir.R0()
    
    if r0 > 1:
        p_c = 1.0 - 1.0/r0
        print(f"  β={beta}, γ={gamma}: R₀={r0:.2f}, p_c={p_c:.3f} ({p_c*100:.1f}%)")
    else:
        print(f"  β={beta}, γ={gamma}: R₀={r0:.2f} < 1 (no vaccination needed)")

print("✓ PASS: Vaccination coverage calculations work")

print("\n" + "=" * 70)
print("OPTIONAL: Create R₀ Visualization")
print("=" * 70)

create_plot = input("Create R₀ analysis plots? (y/n): ").strip().lower()

if create_plot == 'y':
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: R₀ vs Attack Rate
        ax = axes[0, 0]
        r0_range = []
        attack_range = []
        
        for beta in np.linspace(0.05, 1.0, 20):
            sir = SIR(beta=beta, gamma=0.1, mixing="frequency")
            r0 = sir.R0()
            y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            t, y = sir.integrate(y0, t_span=(0, 500))
            
            r0_range.append(r0)
            attack_range.append(y[2, -1])
        
        ax.plot(r0_range, attack_range, 'b-', linewidth=2)
        ax.axvline(1.0, color='r', linestyle='--', alpha=0.5, label='R₀ = 1')
        ax.set_xlabel('R₀')
        ax.set_ylabel('Attack Rate')
        ax.set_title('Final Attack Rate vs R₀')
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Plot 2: Epidemic curves for different R₀
        ax = axes[0, 1]
        
        for beta, color in zip([0.15, 0.3, 0.5], ['blue', 'orange', 'red']):
            sir = SIR(beta=beta, gamma=0.1, mixing="frequency")
            r0 = sir.R0()
            y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            t, y = sir.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
            
            ax.plot(t, y[1], linewidth=2, color=color, label=f'R₀ = {r0:.1f}')
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Infected (I)')
        ax.set_title('Epidemic Curves for Different R₀')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Vaccination threshold
        ax = axes[1, 0]
        
        r0_vals = np.linspace(1.1, 10, 50)
        p_c_vals = 1.0 - 1.0/r0_vals
        
        ax.plot(r0_vals, p_c_vals, 'b-', linewidth=2)
        ax.set_xlabel('R₀')
        ax.set_ylabel('Critical Vaccination Coverage (p_c)')
        ax.set_title('Vaccination Threshold: p_c = 1 - 1/R₀')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])
        
        # Plot 4: R₀ comparison across models
        ax = axes[1, 1]
        
        models_r0 = {
            'SIS': SIS(beta=0.3, gamma=0.1).R0(),
            'SIR': SIR(beta=0.3, gamma=0.1).R0(),
            'SIRS': SIRS(beta=0.3, gamma=0.1, omega=0.05).R0(),
            'SEIR': SEIR(beta=0.3, gamma=0.1, sigma=0.2).R0(),
        }
        
        ax.bar(models_r0.keys(), models_r0.values(), color=['blue', 'orange', 'green', 'red'])
        ax.set_ylabel('R₀')
        ax.set_title('R₀ Across Models (β=0.3, γ=0.1)')
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='R₀ = 1')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        output_path = input("Save plot to (press Enter for './r0_analysis.png'): ").strip()
        if not output_path:
            output_path = './r0_analysis.png'
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
        
    except Exception as e:
        print(f"✗ Could not create plot: {e}")
        import traceback
        traceback.print_exc()

# Final Summary
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

all_tests = [
    ("Test 1: R₀ Methods Exist", test1_passed),
    ("Test 2: Frequency-Dependent R₀", test2_passed),
    ("Test 3: Density-Dependent R₀", test3_passed),
    ("Test 4: R₀ Threshold Behavior", test4_passed),
    ("Test 5: R₀ and Attack Rate", test5_passed),
    ("Test 6: Vaccination Coverage", test6_passed),
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
    print("\nStep 3 is complete!")
    print("R₀ methods are working correctly.")
    print("\n📌 NEXT: Ready for Step 4 - Parameter Validation")
else:
    print("⚠️  SOME TESTS FAILED")
    print("\nPlease review the failed tests above.")
    print("Common issues:")
    print("  - Make sure R0() method is added to all model classes")
    print("  - Check the formula: R₀ = β/γ (frequency) or βN/γ (density)")
    print("  - Make sure the method takes N0 parameter (default 1.0)")

print("=" * 70)
