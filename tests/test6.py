"""
Test script for improved counts models with endemic equilibrium and conservation checks.

Place this file in: kr_epi/tests/test_counts_improvements.py
"""
import numpy as np

from kr_epi.models.ode_counts import (
    SIRDemographyCounts, 
    SISDemographyCounts,
    SEIRDemographyCounts
)
from kr_epi.models.ode_counts_mortality import SIRDemographyCountsMortality

def test_sir_demographics():
    """Test SIR with demographics"""
    print("="*60)
    print("TEST 1: SIR with Demographics")
    print("="*60)
    
    # Create model
    sir = SIRDemographyCounts(
        beta=0.5,
        gamma=0.1,
        v=0.02,   # 2% birth rate
        mu=0.02,  # 2% death rate (balanced)
        mixing="frequency"
    )
    
    # Test R0
    R0 = sir.R0()
    print(f"R₀ = {R0:.2f}")
    assert R0 == 0.5 / (0.1 + 0.02), "R0 calculation error"
    
    # Test endemic equilibrium
    eq = sir.endemic_equilibrium()
    print(f"Endemic equilibrium:")
    print(f"  X* = {eq['X']:.2f}")
    print(f"  Y* = {eq['Y']:.2f}")
    print(f"  Z* = {eq['Z']:.2f}")
    print(f"  N* = {eq['N']:.2f}")
    
    if R0 > 1:
        # Test that equilibrium point has zero derivatives
        y_eq = np.array([eq['X'], eq['Y'], eq['Z']])
        dy = sir.rhs(0, y_eq)
        print(f"Derivatives at equilibrium: {dy}")
        assert np.allclose(dy, 0, atol=1e-6), "Equilibrium not stable"
        
        # Test conservation
        cons = sir.check_conservation(y_eq)
        print(f"Conservation error: {cons:.2e}")
        assert cons < 1e-10, "Conservation violated"
    
    print("✓ SIR demographics test passed!\n")

def test_sis_demographics():
    """Test SIS with demographics"""
    print("="*60)
    print("TEST 2: SIS with Demographics")
    print("="*60)
    
    sis = SISDemographyCounts(
        beta=0.8,
        gamma=0.1,
        v=0.02,
        mu=0.02,
        mixing="frequency"
    )
    
    R0 = sis.R0()
    print(f"R₀ = {R0:.2f}")
    
    eq = sis.endemic_equilibrium()
    print(f"Endemic equilibrium:")
    print(f"  X* = {eq['X']:.2f}")
    print(f"  Y* = {eq['Y']:.2f}")
    print(f"  N* = {eq['N']:.2f}")
    
    if R0 > 1:
        y_eq = np.array([eq['X'], eq['Y']])
        dy = sis.rhs(0, y_eq)
        print(f"Derivatives at equilibrium: {dy}")
        assert np.allclose(dy, 0, atol=1e-6), "Equilibrium not stable"
        
        cons = sis.check_conservation(y_eq)
        print(f"Conservation error: {cons:.2e}")
        assert cons < 1e-10, "Conservation violated"
    
    print("✓ SIS demographics test passed!\n")

def test_seir_demographics():
    """Test SEIR with demographics"""
    print("="*60)
    print("TEST 3: SEIR with Demographics")
    print("="*60)
    
    seir = SEIRDemographyCounts(
        beta=0.5,
        sigma=0.2,
        gamma=0.1,
        v=0.02,
        mu=0.02,
        mixing="frequency"
    )
    
    R0 = seir.R0()
    print(f"R₀ = {R0:.2f}")
    
    # Test conservation on arbitrary state
    y_test = np.array([900, 50, 40, 10])
    cons = seir.check_conservation(y_test)
    print(f"Conservation error: {cons:.2e}")
    assert cons < 1e-10, "Conservation violated"
    
    print("✓ SEIR demographics test passed!\n")

def test_mortality_model():
    """Test SIR with disease-induced mortality"""
    print("="*60)
    print("TEST 4: SIR with Disease-Induced Mortality")
    print("="*60)
    
    # Test CFR to delta conversion
    cfr = 0.1  # 10% case fatality ratio
    gamma = 0.1
    mu = 0.02
    delta = SIRDemographyCountsMortality.cfr_to_delta(cfr, gamma, mu)
    print(f"CFR = {cfr:.1%} → delta = {delta:.4f}")
    
    # Create model
    sir_mort = SIRDemographyCountsMortality(
        beta=0.6,
        gamma=0.1,
        v=0.02,
        mu=0.02,
        delta=delta,
        mixing="frequency"
    )
    
    R0 = sir_mort.R0()
    print(f"R₀ = {R0:.2f}")
    
    # Endemic equilibrium
    eq = sir_mort.endemic_equilibrium()
    print(f"Endemic equilibrium:")
    print(f"  X* = {eq['X']:.2f}")
    print(f"  Y* = {eq['Y']:.2f}")
    print(f"  Z* = {eq['Z']:.2f}")
    print(f"  N* = {eq['N']:.2f} (reduced by mortality)")
    print(f"  CFR = {eq['CFR']:.1%}")
    
    if R0 > 1:
        # Test conservation (note: different formula with mortality)
        y_eq = np.array([eq['X'], eq['Y'], eq['Z']])
        cons = sir_mort.check_conservation(y_eq)
        print(f"Conservation error: {cons:.2e}")
        assert cons < 1e-8, "Conservation violated"
    
    print("✓ Mortality model test passed!\n")

def test_parameter_validation():
    """Test parameter validation"""
    print("="*60)
    print("TEST 5: Parameter Validation")
    print("="*60)
    
    try:
        # Should raise error: negative beta
        sir = SIRDemographyCounts(beta=-0.5, gamma=0.1, v=0.02, mu=0.02)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for negative beta: {e}")
    
    try:
        # Should raise error: vacc_p >= 1
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02, vacc_p=1.5)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for invalid vacc_p: {e}")
    
    try:
        # Should raise error: negative delta
        sir_mort = SIRDemographyCountsMortality(
            beta=0.5, gamma=0.1, v=0.02, mu=0.02, delta=-0.1
        )
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"✓ Caught error for negative delta: {e}")
    
    print("✓ Parameter validation test passed!\n")

def test_density_vs_frequency():
    """Test difference between density and frequency dependent transmission"""
    print("="*60)
    print("TEST 6: Density vs Frequency Dependent Transmission")
    print("="*60)
    
    N0 = 1000
    
    # Frequency-dependent
    sir_freq = SIRDemographyCounts(
        beta=0.5, gamma=0.1, v=0.02, mu=0.02, mixing="frequency"
    )
    R0_freq = sir_freq.R0()
    
    # Density-dependent
    sir_dens = SIRDemographyCounts(
        beta=0.5/N0, gamma=0.1, v=0.02, mu=0.02, mixing="density"
    )
    R0_dens = sir_dens.R0(N0)
    
    print(f"Frequency-dependent R₀ = {R0_freq:.2f}")
    print(f"Density-dependent R₀ = {R0_dens:.2f}")
    print(f"(Adjusted beta for density to give same R₀)")
    
    assert np.isclose(R0_freq, R0_dens), "R0 should be similar with adjusted beta"
    
    print("✓ Density vs frequency test passed!\n")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("TESTING IMPROVED COUNTS MODELS")
    print("="*60 + "\n")
    
    test_sir_demographics()
    test_sis_demographics()
    test_seir_demographics()
    test_mortality_model()
    test_parameter_validation()
    test_density_vs_frequency()
    
    print("="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)