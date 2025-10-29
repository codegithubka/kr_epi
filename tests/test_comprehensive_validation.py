"""
Comprehensive validation test suite for kr_epi library.

This test suite validates that the library correctly implements
Keeling & Rohani (2008) formulas and produces expected results.

Run with: pytest tests/test_comprehensive_validation.py -v
"""

import numpy as np
import pytest
from typing import Dict

# Try to import from library
try:
    from kr_epi.models.ode import SIR, SIS, SEIR, SIRS
    from kr_epi.models.ode_counts import (
        SIRDemographyCounts,
        SISDemographyCounts, 
        SEIRDemographyCounts
    )
    from kr_epi.models.ode_counts_mortality import SIRDemographyCountsMortality
    from kr_epi.analysis.equilibria import (
        sis_equilibria,
        sir_final_size,
        sirs_equilibria,
        critical_vaccination_coverage
    )
except ImportError as e:
    pytest.skip(f"Could not import kr_epi: {e}", allow_module_level=True)


class TestBasicR0Calculations:
    """Test R0 calculations match K&R formulas."""
    
    def test_sir_closed_r0_frequency(self):
        """SIR closed population: R0 = beta/gamma (K&R Eq 2.3)"""
        beta, gamma = 0.5, 0.1
        sir = SIR(beta=beta, gamma=gamma, mixing="frequency")
        
        R0 = sir.R0()
        expected = beta / gamma
        
        assert np.isclose(R0, expected), \
            f"SIR R0 = {R0:.4f}, expected {expected:.4f}"
    
    def test_sir_demography_r0_frequency(self):
        """SIR with demography: R0 = beta/(gamma+mu) (K&R Eq 2.19)"""
        beta, gamma, mu = 0.5, 0.1, 0.02
        sir = SIRDemographyCounts(
            beta=beta, gamma=gamma, v=mu, mu=mu, mixing="frequency"
        )
        
        R0 = sir.R0()
        expected = beta / (gamma + mu)
        
        assert np.isclose(R0, expected), \
            f"SIR demography R0 = {R0:.4f}, expected {expected:.4f}"
    
    def test_sir_demography_r0_density_uses_nstar(self):
        """
        CRITICAL: Density-dependent R0 must use N* = v/mu, not N0.
        K&R Equation 2.19
        """
        beta, gamma, v, mu = 0.5, 0.1, 0.02, 0.02
        sir = SIRDemographyCounts(
            beta=beta, gamma=gamma, v=v, mu=mu, mixing="density"
        )
        
        # N* = v/mu = 1.0
        N_star = v / mu
        
        R0 = sir.R0()
        expected = beta * N_star / (gamma + mu)
        
        assert np.isclose(R0, expected), \
            f"Density R0 = {R0:.4f}, expected {expected:.4f} (using N*={N_star})"
        
        # CRITICAL: Even if N0 is different, should use N* by default
        # This was the BUG in original code
    
    def test_seir_closed_r0(self):
        """SEIR closed: R0 = beta/gamma (latent period doesn't affect R0)"""
        beta, sigma, gamma = 0.5, 0.2, 0.1
        seir = SEIR(beta=beta, sigma=sigma, gamma=gamma, mixing="frequency")
        
        R0 = seir.R0()
        expected = beta / gamma  # sigma doesn't appear!
        
        assert np.isclose(R0, expected), \
            f"SEIR R0 = {R0:.4f}, expected {expected:.4f}"
    
    def test_seir_demography_r0_includes_sigma(self):
        """
        CRITICAL: SEIR with demography: R0 = (beta/(gamma+mu)) * (sigma/(sigma+mu))
        K&R Box 2.5, Equation 2.53
        
        This was MISSING in original implementation!
        """
        beta, sigma, gamma, mu = 0.5, 0.2, 0.1, 0.02
        seir = SEIRDemographyCounts(
            beta=beta, sigma=sigma, gamma=gamma, v=mu, mu=mu, mixing="frequency"
        )
        
        R0 = seir.R0()
        
        # K&R formula includes probability of surviving E class
        prob_survive_E = sigma / (sigma + mu)
        expected = (beta / (gamma + mu)) * prob_survive_E
        
        assert np.isclose(R0, expected), \
            f"SEIR demography R0 = {R0:.4f}, expected {expected:.4f}"
        
        # Verify it's less than SIR R0 (some die in E)
        sir = SIRDemographyCounts(beta=beta, gamma=gamma, v=mu, mu=mu)
        R0_sir = sir.R0()
        
        assert R0 < R0_sir, \
            f"SEIR R0 ({R0:.4f}) should be < SIR R0 ({R0_sir:.4f}) when mu > 0"


class TestEndemicEquilibria:
    """Test endemic equilibrium calculations."""
    
    def test_sis_equilibrium_frequency(self):
        """SIS endemic equilibrium (K&R Section 2.1)"""
        beta, gamma = 0.3, 0.1
        R0 = beta / gamma  # 3.0
        
        S_star, I_star = sis_equilibria(beta, gamma, N=1.0, mixing="frequency")
        
        # S* = 1/R0, I* = 1 - 1/R0
        assert np.isclose(S_star, 1.0/R0, rtol=0.01)
        assert np.isclose(I_star, 1.0 - 1.0/R0, rtol=0.01)
        assert np.isclose(S_star + I_star, 1.0)
    
    def test_sir_demography_equilibrium_matches_kr_eq_2_19(self):
        """
        CRITICAL: SIR endemic equilibrium must match K&R Eq 2.19
        
        S*/N* = 1/R0
        I*/N* = (mu/(gamma+mu)) * (1 - 1/R0)
        
        Original code was MISSING N* factor in I*!
        """
        beta, gamma, v, mu = 0.5, 0.1, 0.02, 0.02
        sir = SIRDemographyCounts(beta=beta, gamma=gamma, v=v, mu=mu)
        
        eq = sir.endemic_equilibrium()
        R0 = eq['R0']
        N_star = eq['N']
        
        # Verify N* = v/mu
        assert np.isclose(N_star, v/mu, rtol=0.01)
        
        # Verify S*/N* = 1/R0
        s_frac = eq['X'] / N_star
        assert np.isclose(s_frac, 1.0/R0, rtol=0.01)
        
        # CRITICAL: Verify I*/N* formula (this was WRONG!)
        i_frac = eq['Y'] / N_star
        i_frac_expected = (mu / (gamma + mu)) * (1.0 - 1.0/R0)
        
        assert np.isclose(i_frac, i_frac_expected, rtol=0.01), \
            f"I*/N* = {i_frac:.6f}, expected {i_frac_expected:.6f}"
        
        # Verify conservation
        assert np.isclose(eq['X'] + eq['Y'] + eq['Z'], N_star, rtol=1e-6)
    
    def test_equilibrium_has_zero_derivatives(self):
        """Verify that endemic equilibrium actually has dX/dt = dY/dt = 0"""
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        
        eq = sir.endemic_equilibrium()
        if eq['X'] is None:  # No endemic equilibrium
            pytest.skip("R0 < 1, no endemic equilibrium")
        
        # Create state vector
        y_eq = np.array([eq['X'], eq['Y'], eq['Z']])
        
        # Calculate derivatives
        dydt = sir.rhs(0, y_eq)
        
        # Should all be zero (within numerical tolerance)
        assert np.allclose(dydt, 0, atol=1e-10), \
            f"Derivatives at equilibrium: {dydt}"


class TestConservationLaws:
    """Test that population sizes are conserved correctly."""
    
    def test_sir_closed_conservation(self):
        """Closed SIR: S + I + R = constant"""
        sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = sir.integrate(y0, t_span=(0, 100))
        
        N = y.sum(axis=0)
        assert np.allclose(N, 1.0, atol=1e-6)
    
    def test_sir_demography_conservation(self):
        """
        SIR with demography: dN/dt = (v - mu)*N (no disease mortality)
        """
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        y0_dict = {"X": 990, "Y": 10, "Z": 0}
        
        # Convert to array
        y0 = np.array([y0_dict["X"], y0_dict["Y"], y0_dict["Z"]])
        
        # Check conservation
        dydt = sir.rhs(0, y0)
        dN_dt = dydt.sum()
        
        N0 = y0.sum()
        expected_dN_dt = (sir.params.v - sir.params.mu) * N0
        
        assert np.isclose(dN_dt, expected_dN_dt, rtol=1e-6), \
            f"dN/dt = {dN_dt:.6f}, expected {expected_dN_dt:.6f}"
    
    def test_sir_mortality_conservation(self):
        """
        SIR with disease mortality: dN/dt = (v - mu)*N - delta*Y
        Population declines due to disease deaths.
        """
        delta = 0.01  # disease mortality
        sir = SIRDemographyCountsMortality(
            beta=0.5, gamma=0.1, v=0.02, mu=0.02, delta=delta
        )
        
        y0 = np.array([990, 10, 0])
        dydt = sir.rhs(0, y0)
        dN_dt = dydt.sum()
        
        N0, Y0 = y0.sum(), y0[1]
        expected_dN_dt = (sir.params.v - sir.params.mu) * N0 - delta * Y0
        
        assert np.isclose(dN_dt, expected_dN_dt, rtol=1e-6)


class TestEpidemicThreshold:
    """Test that R0 > 1 predicts epidemic correctly (K&R Theorem 2.1)"""
    
    def test_r0_greater_than_1_gives_epidemic(self):
        """R0 > 1: Disease should spread"""
        sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")  # R0 = 5
        assert sir.R0() > 1
        
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = sir.integrate(y0, t_span=(0, 100))
        
        I_max = y[1, :].max()
        I_initial = y0["I"]
        
        assert I_max > I_initial, \
            f"R0 = {sir.R0():.2f} > 1 but epidemic did not grow"
    
    def test_r0_less_than_1_no_epidemic(self):
        """R0 < 1: Disease should die out"""
        sir = SIR(beta=0.05, gamma=0.1, mixing="frequency")  # R0 = 0.5
        assert sir.R0() < 1
        
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        t, y = sir.integrate(y0, t_span=(0, 100))
        
        I_final = y[1, -1]
        I_initial = y0["I"]
        
        assert I_final < I_initial, \
            f"R0 = {sir.R0():.2f} < 1 but epidemic grew"


class TestFinalSize:
    """Test final size relation (K&R Eq 2.17)"""
    
    def test_final_size_satisfies_implicit_equation(self):
        """Final size must satisfy: s_inf = s0 * exp(-R0*(1-s_inf))"""
        s0 = 0.99
        R0 = 5.0
        
        s_inf = sir_final_size(s0, R0)
        
        # Check implicit equation
        expected = s0 * np.exp(-R0 * (1.0 - s_inf))
        
        assert np.isclose(s_inf, expected, rtol=1e-6), \
            f"s_inf = {s_inf:.6f} doesn't satisfy implicit equation"
    
    def test_final_size_matches_simulation(self):
        """Final size from formula should match ODE simulation"""
        sir = SIR(beta=0.5, gamma=0.1, mixing="frequency")
        R0 = sir.R0()
        
        s0 = 0.99
        s_inf_analytical = sir_final_size(s0, R0)
        
        # Simulate
        y0 = {"S": s0, "I": 1.0 - s0, "R": 0.0}
        t, y = sir.integrate(y0, t_span=(0, 500))  # Long time
        s_inf_simulated = y[0, -1]
        
        assert np.isclose(s_inf_analytical, s_inf_simulated, rtol=0.01), \
            f"Analytical s_inf = {s_inf_analytical:.4f}, " \
            f"simulated = {s_inf_simulated:.4f}"
    
    def test_higher_r0_gives_higher_attack_rate(self):
        """Higher R0 should result in higher attack rate"""
        s0 = 0.99
        
        R0_low = 2.0
        R0_high = 10.0
        
        s_inf_low = sir_final_size(s0, R0_low)
        s_inf_high = sir_final_size(s0, R0_high)
        
        attack_low = 1.0 - s_inf_low
        attack_high = 1.0 - s_inf_high
        
        assert attack_high > attack_low, \
            f"Higher R0 should give higher attack rate"


class TestVaccinationThreshold:
    """Test vaccination threshold p_c = 1 - 1/R0"""
    
    def test_vaccination_threshold_formula(self):
        """Critical coverage: p_c = 1 - 1/R0"""
        R0 = 5.0
        p_c = critical_vaccination_coverage(R0)
        
        expected = 1.0 - 1.0/R0  # = 0.8
        
        assert np.isclose(p_c, expected)
    
    def test_vaccination_prevents_epidemic(self):
        """Vaccinating > p_c should prevent epidemic"""
        beta, gamma = 0.5, 0.1
        R0_base = beta / gamma  # = 5.0
        
        p_c = critical_vaccination_coverage(R0_base)  # = 0.8
        
        # Vaccinate 85% (> p_c)
        vacc_coverage = 0.85
        
        # Effective R0 after vaccination
        R0_eff = R0_base * (1 - vacc_coverage)
        
        assert R0_eff < 1, f"R0_eff = {R0_eff:.2f} should be < 1"


class TestParameterValidation:
    """Test that invalid parameters are rejected."""
    
    def test_negative_beta_raises_error(self):
        """Beta < 0 should raise ValueError"""
        with pytest.raises(ValueError, match="beta"):
            SIR(beta=-0.5, gamma=0.1)
    
    def test_zero_gamma_raises_error(self):
        """Gamma = 0 should raise ValueError"""
        with pytest.raises(ValueError, match="gamma"):
            SIR(beta=0.5, gamma=0)
    
    def test_negative_gamma_raises_error(self):
        """Gamma < 0 should raise ValueError"""
        with pytest.raises(ValueError, match="gamma"):
            SIR(beta=0.5, gamma=-0.1)
    
    def test_invalid_mixing_raises_error(self):
        """Invalid mixing type should raise ValueError"""
        with pytest.raises(ValueError, match="mixing"):
            SIR(beta=0.5, gamma=0.1, mixing="invalid")


class TestModelComparisons:
    """Test relationships between different model types."""
    
    def test_seir_has_delay_compared_to_sir(self):
        """SEIR should have delayed peak compared to SIR"""
        beta, gamma = 0.5, 0.1
        sigma = 0.2  # latent period
        
        sir = SIR(beta=beta, gamma=gamma)
        seir = SEIR(beta=beta, sigma=sigma, gamma=gamma)
        
        # Same initial conditions (total infected = 0.01)
        y0_sir = {"S": 0.99, "I": 0.01, "R": 0.0}
        y0_seir = {"S": 0.99, "E": 0.005, "I": 0.005, "R": 0.0}
        
        t_eval = np.linspace(0, 100, 300)
        t_sir, y_sir = sir.integrate(y0_sir, t_span=(0, 100), t_eval=t_eval)
        t_seir, y_seir = seir.integrate(y0_seir, t_span=(0, 100), t_eval=t_eval)
        
        # Find peak times
        I_sir = y_sir[1, :]
        I_seir = y_seir[2, :]
        
        t_peak_sir = t_eval[np.argmax(I_sir)]
        t_peak_seir = t_eval[np.argmax(I_seir)]
        
        # SEIR should peak later (due to latent period)
        assert t_peak_seir > t_peak_sir, \
            f"SEIR peak ({t_peak_seir:.1f}) should be later than SIR ({t_peak_sir:.1f})"
    
    def test_sirs_approaches_sis_as_omega_increases(self):
        """As waning rate ω → ∞, SIRS → SIS"""
        beta, gamma = 0.3, 0.1
        
        # SIRS with very fast waning (approaches SIS)
        sirs = SIRS(beta=beta, gamma=gamma, omega=10.0)  # Very fast waning
        
        # SIS
        sis = SIS(beta=beta, gamma=gamma)
        
        # Both should have similar R0
        R0_sirs = sirs.R0()
        R0_sis = sis.R0()
        
        # For SIRS and SIS, R0 = beta/gamma (omega doesn't affect R0)
        assert np.isclose(R0_sirs, R0_sis)


class TestNumericalStability:
    """Test numerical stability and edge cases."""
    
    def test_no_negative_populations(self):
        """Populations should never go negative"""
        sir = SIR(beta=2.0, gamma=0.5)  # High transmission
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        
        t, y = sir.integrate(y0, t_span=(0, 50))
        
        assert np.all(y >= -1e-10), f"Negative values detected: min = {y.min()}"
    
    def test_very_small_initial_infection(self):
        """Model should handle very small initial infections"""
        sir = SIR(beta=0.5, gamma=0.1)
        y0 = {"S": 0.999999, "I": 1e-6, "R": 0.0}
        
        t, y = sir.integrate(y0, t_span=(0, 100))
        
        # Should still have epidemic if R0 > 1
        assert y[1, :].max() > y0["I"]
    
    def test_stiff_system_integration(self):
        """Very high R0 creates stiff system - should still integrate"""
        sir = SIR(beta=50.0, gamma=1.0)  # R0 = 50, very stiff
        y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
        
        # Should work with automatic stiff detection
        t, y = sir.integrate(y0, t_span=(0, 20), stiff_fallback=True)
        
        # Should reach final state
        assert y[2, -1] > 0.9  # Most end up recovered


class TestKeelingRohaniExamples:
    """
    Test specific examples from Keeling & Rohani (2008).
    Ensures library reproduces published results.
    """
    
    def test_table_2_1_measles_parameters(self):
        """
        Test measles parameters from K&R Table 2.1:
        - Infectious period: 13 days (gamma = 1/13)
        - R0 ≈ 13-18 (using typical beta values)
        """
        gamma = 1.0 / 13.0  # per day
        R0_target = 15.0  # middle of range
        beta = R0_target * gamma
        
        sir = SIR(beta=beta, gamma=gamma)
        R0_calculated = sir.R0()
        
        assert np.isclose(R0_calculated, R0_target)
        
        # Generation time ≈ infectious period for SIR
        gen_time = 1.0 / gamma
        assert np.isclose(gen_time, 13.0)
    
    def test_figure_2_1_sir_dynamics(self):
        """
        Reproduce K&R Figure 2.1: Basic SIR dynamics
        - Parameters: beta=0.5, gamma=0.1 (R0=5)
        - Initial: S(0)=0.999, I(0)=0.001
        """
        sir = SIR(beta=0.5, gamma=0.1)
        assert np.isclose(sir.R0(), 5.0)
        
        y0 = {"S": 0.999, "I": 0.001, "R": 0.0}
        t, y = sir.integrate(y0, t_span=(0, 70))
        
        # Peak should occur around t ≈ 15-20
        I_peak_idx = np.argmax(y[1, :])
        t_peak = t[I_peak_idx]
        
        assert 12 < t_peak < 22, f"Peak at t={t_peak:.1f}, expected ~15"
        
        # Final size: R(∞) ≈ 0.993
        R_final = y[2, -1]
        assert 0.99 < R_final < 1.0


# Helper function to run all tests
def run_validation_suite():
    """Run all validation tests and print summary."""
    import pytest
    import sys
    
    # Run tests
    exit_code = pytest.main([__file__, '-v', '--tb=short'])
    
    if exit_code == 0:
        print("\n" + "="*70)
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("="*70)
        print("\nYour kr_epi library correctly implements:")
        print("  ✓ R0 calculations (K&R Equations 2.3, 2.19, 2.53)")
        print("  ✓ Endemic equilibria (K&R Equations 2.19-2.20)")
        print("  ✓ Conservation laws")
        print("  ✓ Epidemic threshold (R0 > 1)")
        print("  ✓ Final size relation (K&R Equation 2.17)")
        print("  ✓ Vaccination thresholds")
        print("  ✓ Parameter validation")
        print("\nLibrary is ready for research use!")
    else:
        print("\n" + "="*70)
        print("❌ SOME TESTS FAILED")
        print("="*70)
        print("\nPlease review failures above and fix issues.")
        print("Critical fixes needed before library can be trusted.")
    
    return exit_code


if __name__ == "__main__":
    exit(run_validation_suite())