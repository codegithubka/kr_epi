"""
Test conservation laws for epidemic models.
Ensures population totals remain constant over time.
"""
import numpy as np
import pytest
from kr_epi.models.ode import SI, SIS, SIR, SIRS, SEIR


class TestConservationLaws:
    """Test that population sizes are conserved."""
    
    def test_si_conservation(self):
        """SI: S + I should be constant."""
        model = SI(beta=0.5, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6), \
            f"Population not conserved in SI: {N[0]:.8f} -> {N[-1]:.8f}"
    
    def test_sis_conservation(self):
        """SIS: S + I should be constant."""
        model = SIS(beta=0.3, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6), \
            f"Population not conserved in SIS: {N[0]:.8f} -> {N[-1]:.8f}"
    
    def test_sir_conservation(self):
        """SIR: S + I + R should be constant."""
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6), \
            f"Population not conserved in SIR: {N[0]:.8f} -> {N[-1]:.8f}"
    
    def test_sirs_conservation(self):
        """SIRS: S + I + R should be constant."""
        model = SIRS(beta=0.5, gamma=0.1, omega=0.05, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 200))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6), \
            f"Population not conserved in SIRS: {N[0]:.8f} -> {N[-1]:.8f}"
    
    def test_seir_conservation(self):
        """SEIR: S + E + I + R should be constant."""
        model = SEIR(beta=0.5, sigma=0.2, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "E": 0.005, "I": 0.005, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6), \
            f"Population not conserved in SEIR: {N[0]:.8f} -> {N[-1]:.8f}"


class TestNonNegativity:
    """Test that all compartments remain non-negative."""
    
    def test_sir_nonnegative(self):
        """All compartments should remain >= 0."""
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        assert np.all(y >= -1e-10), \
            f"Negative values detected: min = {y.min():.2e}"
    
    def test_seir_nonnegative(self):
        """All compartments should remain >= 0."""
        model = SEIR(beta=0.5, sigma=0.2, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "E": 0.005, "I": 0.005, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        assert np.all(y >= -1e-10), \
            f"Negative values detected: min = {y.min():.2e}"


class TestInitialConditions:
    """Test that initial conditions are preserved at t=0."""
    
    def test_sir_initial(self):
        """First time point should equal initial conditions."""
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100))
        
        assert np.allclose(y[:, 0], [0.99, 0.01, 0.0], atol=1e-10), \
            f"Initial conditions not preserved: {y[:, 0]}"


class TestMonotonicity:
    """Test monotonic properties (S decreases, R increases, etc.)."""
    
    def test_sir_s_decreases(self):
        """S should never increase in SIR model."""
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        
        S = y[0, :]
        # Check that S is monotonically decreasing (or constant if I=0)
        dS = np.diff(S)
        assert np.all(dS <= 1e-10), \
            f"S increased somewhere: max increase = {dS.max():.2e}"
    
    def test_sir_r_increases(self):
        """R should never decrease in SIR model."""
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = model.integrate(y0, t_span=(0, 100), t_eval=np.linspace(0, 100, 200))
        
        R = y[2, :]
        # Check that R is monotonically increasing
        dR = np.diff(R)
        assert np.all(dR >= -1e-10), \
            f"R decreased somewhere: min change = {dR.min():.2e}"


class TestSymmetry:
    """Test symmetry properties."""
    
    def test_sir_final_state(self):
        """Final S should not depend on intermediate dynamics."""
        # Two simulations with same parameters but different I0
        model = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        
        # Small initial infection
        y0_small = {"S": 0.99, "I": 0.01, "R": 0.0}
        t1, y1 = model.integrate(y0_small, t_span=(0, 500))
        
        # Larger initial infection
        y0_large = {"S": 0.95, "I": 0.05, "R": 0.0}
        t2, y2 = model.integrate(y0_large, t_span=(0, 500))
        
        # Final attack rates should be different
        attack1 = y1[2, -1]  # Final R
        attack2 = y2[2, -1]
        
        # Larger initial infection -> larger attack rate
        assert attack2 > attack1, \
            f"Larger I0 should give larger attack rate: {attack1:.3f} vs {attack2:.3f}"


if __name__ == "__main__":
    # Run tests manually
    print("Running Conservation Law Tests...")
    print("=" * 60)
    
    # Conservation tests
    print("\n1. Testing Conservation Laws:")
    tests = TestConservationLaws()
    try:
        tests.test_si_conservation()
        print("  ✓ SI conservation")
    except AssertionError as e:
        print(f"  ✗ SI conservation: {e}")
    
    try:
        tests.test_sis_conservation()
        print("  ✓ SIS conservation")
    except AssertionError as e:
        print(f"  ✗ SIS conservation: {e}")
    
    try:
        tests.test_sir_conservation()
        print("  ✓ SIR conservation")
    except AssertionError as e:
        print(f"  ✗ SIR conservation: {e}")
    
    try:
        tests.test_sirs_conservation()
        print("  ✓ SIRS conservation")
    except AssertionError as e:
        print(f"  ✗ SIRS conservation: {e}")
    
    try:
        tests.test_seir_conservation()
        print("  ✓ SEIR conservation")
    except AssertionError as e:
        print(f"  ✗ SEIR conservation: {e}")
    
    # Non-negativity tests
    print("\n2. Testing Non-Negativity:")
    tests_nn = TestNonNegativity()
    try:
        tests_nn.test_sir_nonnegative()
        print("  ✓ SIR non-negative")
    except AssertionError as e:
        print(f"  ✗ SIR non-negative: {e}")
    
    try:
        tests_nn.test_seir_nonnegative()
        print("  ✓ SEIR non-negative")
    except AssertionError as e:
        print(f"  ✗ SEIR non-negative: {e}")
    
    # Monotonicity tests
    print("\n3. Testing Monotonicity:")
    tests_mono = TestMonotonicity()
    try:
        tests_mono.test_sir_s_decreases()
        print("  ✓ SIR: S decreases")
    except AssertionError as e:
        print(f"  ✗ SIR: S decreases: {e}")
    
    try:
        tests_mono.test_sir_r_increases()
        print("  ✓ SIR: R increases")
    except AssertionError as e:
        print(f"  ✗ SIR: R increases: {e}")
    
    print("\n" + "=" * 60)
    print("Tests complete! Run with pytest for detailed output.")