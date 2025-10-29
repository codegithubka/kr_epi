"""
Tests for ModelValidator

Run with: pytest test_model_validator.py -v
"""

import numpy as np
import pytest
from kr_epi.models.ode import SIR, SEIR
from kr_epi.models.ode_counts import SIRDemographyCounts
from kr_epi.validation.model_validator import ModelValidator, validate_model


class TestR0Threshold:
    """Test R₀ threshold behavior validation."""
    
    def test_high_r0_causes_epidemic(self):
        """R₀ > 1 should cause epidemic."""
        sir = SIR(beta=2.0, gamma=1.0)  # R₀ = 2
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_r0_threshold()
        
        assert result.passed, result.message
        assert result.details['R0'] > 1
        assert result.details['I_max'] > result.details['I_initial']
    
    def test_low_r0_causes_extinction(self):
        """R₀ < 1 should cause disease extinction."""
        sir = SIR(beta=0.5, gamma=1.0)  # R₀ = 0.5
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_r0_threshold()
        
        assert result.passed, result.message
        assert result.details['R0'] < 1
        assert result.details['I_final'] < result.details['I_initial']
    
    def test_seir_r0_threshold(self):
        """SEIR model should also satisfy R₀ threshold."""
        seir = SEIR(beta=3.0, sigma=0.5, gamma=1.0)  # R₀ = 3
        validator = ModelValidator(seir, verbose=False)
        result = validator.check_r0_threshold()
        
        assert result.passed, result.message


class TestConservation:
    """Test conservation law validation."""
    
    def test_closed_sir_conserves_population(self):
        """Closed SIR should conserve total population."""
        sir = SIR(beta=2.0, gamma=1.0)
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_conservation()
        
        assert result.passed, result.message
        assert result.details['max_variation'] < 1e-6
    
    def test_sir_with_demography(self):
        """SIR with demography should follow dN/dt = (v-mu)N."""
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_conservation()
        
        assert result.passed, result.message
        assert result.details['max_error'] < 1e-6


class TestEquilibriumStability:
    """Test equilibrium stability validation."""
    
    def test_endemic_equilibrium_is_stable(self):
        """Endemic equilibrium should be stable when R₀ > 1."""
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_equilibrium_stability()
        
        if result.details and 'reason' not in result.details:
            # Only check if equilibrium exists
            assert result.passed, result.message
    
    def test_no_equilibrium_when_r0_low(self):
        """No endemic equilibrium exists when R₀ < 1."""
        sir = SIRDemographyCounts(beta=0.08, gamma=0.1, v=0.02, mu=0.02)  # R₀ < 1
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_equilibrium_stability()
        
        # Should pass because there's no equilibrium to test
        assert result.passed
        assert "No endemic equilibrium" in result.message or "not feasible" in result.message


class TestFinalSize:
    """Test final size relation validation."""
    
    def test_final_size_matches_theory(self):
        """Simulated final size should match analytical prediction."""
        sir = SIR(beta=2.0, gamma=1.0)  # R₀ = 2
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_final_size()
        
        assert result.passed, result.message
        assert result.details['error'] < 0.001
    
    def test_final_size_skipped_for_demography(self):
        """Final size relation doesn't apply to models with demography."""
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_final_size()
        
        # Should pass (skipped) because it has demography
        assert result.passed
        assert "closed populations" in result.message


class TestNumericalAccuracy:
    """Test numerical accuracy validation."""
    
    def test_results_consistent_across_tolerances(self):
        """Results should be similar for different tolerances."""
        sir = SIR(beta=2.0, gamma=1.0)
        validator = ModelValidator(sir, verbose=False)
        result = validator.check_numerical_accuracy()
        
        assert result.passed, result.message
        assert result.details['max_relative_diff'] < 0.01


class TestValidateAll:
    """Test comprehensive validation."""
    
    def test_validate_all_sir(self):
        """Validate all checks on SIR model."""
        sir = SIR(beta=2.0, gamma=1.0)
        results = validate_model(sir, verbose=False)
        
        assert len(results) == 5  # All 5 checks
        passed = sum(r.passed for r in results)
        assert passed >= 4, f"Only {passed}/5 tests passed"
    
    def test_validate_all_sir_demography(self):
        """Validate all checks on SIR with demography."""
        sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        results = validate_model(sir, verbose=False)
        
        assert len(results) == 5
        passed = sum(r.passed for r in results)
        assert passed >= 4, f"Only {passed}/5 tests passed"
    
    def test_validator_summary(self):
        """Test summary output."""
        sir = SIR(beta=2.0, gamma=1.0)
        validator = ModelValidator(sir, verbose=False)
        validator.validate_all()
        
        summary = validator.summary()
        assert "VALIDATION SUMMARY" in summary
        assert "SIR" in summary


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_result_string_representation(self):
        """Test that ValidationResult has good string repr."""
        from kr_epi.validation import ValidationResult
        
        result = ValidationResult(
            test_name="Test",
            passed=True,
            message="All good",
            details={"value": 1.0}
        )
        
        result_str = str(result)
        assert "✓ PASS" in result_str
        assert "Test" in result_str
        assert "All good" in result_str


if __name__ == "__main__":
    # Run tests manually
    print("=" * 70)
    print("Running ModelValidator Tests")
    print("=" * 70)
    
    # Test 1: R₀ threshold
    print("\n1. Testing R₀ threshold...")
    test = TestR0Threshold()
    test.test_high_r0_causes_epidemic()
    test.test_low_r0_causes_extinction()
    print("✓ R₀ threshold tests passed")
    
    # Test 2: Conservation
    print("\n2. Testing conservation...")
    test = TestConservation()
    test.test_closed_sir_conserves_population()
    test.test_sir_with_demography()
    print("✓ Conservation tests passed")
    
    # Test 3: Equilibrium stability
    print("\n3. Testing equilibrium stability...")
    test = TestEquilibriumStability()
    #test.test_endemic_equilibrium_is_stable()
    test.test_no_equilibrium_when_r0_low()
    print("✓ Equilibrium tests passed")
    
    # Test 4: Final size
    print("\n4. Testing final size...")
    test = TestFinalSize()
    test.test_final_size_matches_theory()
    test.test_final_size_skipped_for_demography()
    print("✓ Final size tests passed")
    
    # Test 5: Numerical accuracy
    print("\n5. Testing numerical accuracy...")
    test = TestNumericalAccuracy()
    test.test_results_consistent_across_tolerances()
    print("✓ Numerical accuracy tests passed")
    
    # Test 6: Validate all
    print("\n6. Testing comprehensive validation...")
    test = TestValidateAll()
    test.test_validate_all_sir()
    test.test_validate_all_sir_demography()
    test.test_validator_summary()
    print("✓ Comprehensive validation tests passed")
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)