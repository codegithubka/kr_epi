"""
Model Validation Framework for kr_epi

Automatically verifies that epidemic models behave according to theory:
- R₀ threshold behavior (Theorem 2.1 from K&R)
- Conservation laws
- Equilibrium stability
- Numerical accuracy

Usage:
    from kr_epi.validation import ModelValidator
    from kr_epi.models.ode import SIR
    
    sir = SIR(beta=2.0, gamma=1.0)
    validator = ModelValidator(sir)
    
    # Run all checks
    results = validator.validate_all()
    
    # Or run individual checks
    validator.check_r0_threshold()
    validator.check_conservation()
    validator.check_equilibrium_stability()

Author: kr_epi development team
References: Keeling & Rohani (2008), Modeling Infectious Diseases
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import warnings
from abc import ABC

# Type aliases
Array = np.ndarray


@dataclass
class ValidationResult:
    """Result from a single validation check."""
    test_name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    
    def __str__(self) -> str:
        status = "✓ PASS" if self.passed else "✗ FAIL"
        return f"{status}: {self.test_name}\n  {self.message}"


class ModelValidator:
    """
    Validate epidemic model behavior against theoretical predictions.
    
    This class provides automated checks for:
    1. R₀ threshold behavior (K&R Theorem 2.1)
    2. Conservation laws
    3. Equilibrium stability
    4. Final size relations (for closed populations)
    5. Numerical integration accuracy
    
    Parameters
    ----------
    model : ODEBase
        Epidemic model to validate (must have integrate, R0 methods)
    tolerance : float
        Numerical tolerance for checks (default: 1e-6)
    verbose : bool
        Print detailed output (default: True)
        
    Examples
    --------
    >>> from kr_epi.models.ode import SIR
    >>> sir = SIR(beta=2.0, gamma=1.0)
    >>> validator = ModelValidator(sir)
    >>> results = validator.validate_all()
    >>> print(f"Passed: {sum(r.passed for r in results)}/{len(results)}")
    
    References
    ----------
    Keeling, M. J., & Rohani, P. (2008). Modeling infectious diseases in 
    humans and animals. Princeton University Press.
    """
    
    def __init__(self, model, tolerance: float = 1e-6, verbose: bool = True):
        self.model = model
        self.tolerance = tolerance
        self.verbose = verbose
        self.results: List[ValidationResult] = []
        
    def _log(self, message: str):
        """Print message if verbose mode is on."""
        if self.verbose:
            print(message)
            
    def _add_result(self, result: ValidationResult):
        """Add result to list and optionally print."""
        self.results.append(result)
        if self.verbose:
            print(result)
            
    # =========================================================================
    # 1. R₀ THRESHOLD BEHAVIOR (K&R Theorem 2.1)
    # =========================================================================
    
    def check_r0_threshold(
        self,
        y0_high_r0: Optional[Dict[str, float]] = None,
        y0_low_r0: Optional[Dict[str, float]] = None,
        t_max: float = 100,
        growth_threshold: float = 1.1
    ) -> ValidationResult:
        """
        Verify R₀ threshold: R₀>1 causes epidemic, R₀<1 causes extinction.
        
        From K&R Theorem 2.1: Disease can invade if and only if R₀ > 1.
        
        Parameters
        ----------
        y0_high_r0 : dict, optional
            Initial conditions for high R₀ test (default: small outbreak)
        y0_low_r0 : dict, optional
            Initial conditions for low R₀ test (default: small outbreak)
        t_max : float
            Integration time (default: 100)
        growth_threshold : float
            Ratio to determine if epidemic grew (default: 1.1)
            
        Returns
        -------
        ValidationResult
            Pass if behavior matches theory
            
        Examples
        --------
        >>> sir = SIR(beta=2.0, gamma=1.0)  # R₀ = 2
        >>> validator = ModelValidator(sir)
        >>> result = validator.check_r0_threshold()
        >>> assert result.passed
        """
        try:
            R0 = self.model.R0() if hasattr(self.model, 'R0') else None
            
            if R0 is None:
                return ValidationResult(
                    test_name="R₀ Threshold Behavior",
                    passed=False,
                    message="Model does not have R0() method",
                    details={"reason": "Missing R0 method"}
                )
            
            # Determine infected compartment name
            # <-- FIX: Changed from state_labels() to labels
            labels = self.model.labels if hasattr(self.model, 'labels') else None
            if labels:
                infected_idx = None
                for i, label in enumerate(labels):
                    if label in ['I', 'Y']:  # Common names for infected
                        infected_idx = i
                        break
                if infected_idx is None:
                    infected_idx = 1  # Default assumption
            else:
                infected_idx = 1
            
            # Default initial conditions (small outbreak in mostly susceptible pop)
            if y0_high_r0 is None:
                if labels and labels[0] in ['S', 'X']:
                    y0_high_r0 = {labels[0]: 0.99, labels[infected_idx]: 0.01}
                    # Add other compartments as 0
                    for label in labels:
                        if label not in y0_high_r0:
                            y0_high_r0[label] = 0.0
                else:
                    y0_high_r0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            
            # Integrate
            t, y = self.model.integrate(y0_high_r0, t_span=(0, t_max))
            
            I_initial = list(y0_high_r0.values())[infected_idx]
            I_max = y[infected_idx, :].max()
            I_final = y[infected_idx, -1]
            
            # Check behavior based on R₀
            if R0 > 1:
                # Should see epidemic (I grows significantly)
                epidemic_occurred = I_max > I_initial * growth_threshold
                
                if epidemic_occurred:
                    return ValidationResult(
                        test_name="R₀ Threshold Behavior",
                        passed=True,
                        message=f"R₀={R0:.2f}>1: Epidemic occurred as expected (peak I={I_max:.4f})",
                        details={
                            "R0": R0,
                            "I_initial": I_initial,
                            "I_max": I_max,
                            "I_final": I_final
                        }
                    )
                else:
                    return ValidationResult(
                        test_name="R₀ Threshold Behavior",
                        passed=False,
                        message=f"R₀={R0:.2f}>1 but epidemic died out (max I={I_max:.4f})",
                        details={
                            "R0": R0,
                            "I_initial": I_initial,
                            "I_max": I_max,
                            "expected": "epidemic growth"
                        }
                    )
            elif R0 < 1:
                # Should see extinction (I decreases)
                extinction_occurred = I_final < I_initial * 0.9
                
                if extinction_occurred:
                    return ValidationResult(
                        test_name="R₀ Threshold Behavior",
                        passed=True,
                        message=f"R₀={R0:.2f}<1: Disease died out as expected (final I={I_final:.4f})",
                        details={
                            "R0": R0,
                            "I_initial": I_initial,
                            "I_final": I_final
                        }
                    )
                else:
                    return ValidationResult(
                        test_name="R₀ Threshold Behavior",
                        passed=False,
                        message=f"R₀={R0:.2f}<1 but disease persisted (final I={I_final:.4f})",
                        details={
                            "R0": R0,
                            "I_initial": I_initial,
                            "I_final": I_final,
                            "expected": "disease extinction"
                        }
                    )
            else:  # R0 == 1
                return ValidationResult(
                    test_name="R₀ Threshold Behavior",
                    passed=True,
                    message="R₀=1.0: Critical threshold (behavior not tested)",
                    details={"R0": R0, "note": "R0=1 is boundary case"}
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="R₀ Threshold Behavior",
                passed=False,
                message=f"Test failed with error: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    # =========================================================================
    # 2. CONSERVATION LAWS
    # =========================================================================
    
    def check_conservation(
        self,
        y0: Optional[Dict[str, float]] = None,
        t_max: float = 100,
        n_checks: int = 10
    ) -> ValidationResult:
        """
        Verify population conservation laws hold during integration.
        
        For closed populations: N(t) = constant
        For open populations: dN/dt = (v-mu)*N
        
        Parameters
        ----------
        y0 : dict, optional
            Initial conditions (default: use typical outbreak)
        t_max : float
            Integration time (default: 100)
        n_checks : int
            Number of timepoints to check (default: 10)
            
        Returns
        -------
        ValidationResult
            Pass if conservation error is below tolerance
            
        Examples
        --------
        >>> sir = SIR(beta=2.0, gamma=1.0)
        >>> validator = ModelValidator(sir)
        >>> result = validator.check_conservation()
        >>> assert result.passed
        """
        try:
            # Set up initial conditions
            if y0 is None:
                # <-- FIX: Changed from state_labels() to labels
                labels = self.model.labels if hasattr(self.model, 'labels') else None
                if labels:
                    # Default: mostly susceptible, small infection
                    y0 = {labels[0]: 0.99, labels[1]: 0.01}
                    for label in labels[2:]:
                        y0[label] = 0.0
                else:
                    y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            
            # Integrate
            t, y = self.model.integrate(y0, t_span=(0, t_max))
            
            # Check if model has check_conservation method
            if hasattr(self.model, 'check_conservation'):
                # Use model's built-in check
                max_error = 0.0
                errors = []
                
                check_indices = np.linspace(0, len(t)-1, n_checks, dtype=int)
                for idx in check_indices:
                    y_i = y[:, idx]
                    error = self.model.check_conservation(y_i)
                    errors.append(error)
                    max_error = max(max_error, error)
                
                passed = max_error < self.tolerance
                
                return ValidationResult(
                    test_name="Conservation Laws",
                    passed=passed,
                    message=f"Max conservation error: {max_error:.2e} {'<' if passed else '≥'} {self.tolerance:.2e}",
                    details={
                        "max_error": max_error,
                        "tolerance": self.tolerance,
                        "errors": errors,
                        "check_points": list(t[check_indices])
                    }
                )
            else:
                # Check population sum is constant (for closed models)
                N = y.sum(axis=0)
                N_initial = N[0]
                N_variation = np.abs(N - N_initial).max()
                
                passed = N_variation < self.tolerance
                
                return ValidationResult(
                    test_name="Conservation Laws",
                    passed=passed,
                    message=f"Population variation: {N_variation:.2e} {'<' if passed else '≥'} {self.tolerance:.2e}",
                    details={
                        "N_initial": N_initial,
                        "N_final": N[-1],
                        "max_variation": N_variation,
                        "tolerance": self.tolerance
                    }
                )
                
        except Exception as e:
            return ValidationResult(
                test_name="Conservation Laws",
                passed=False,
                message=f"Test failed with error: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    # =========================================================================
    # 3. EQUILIBRIUM STABILITY
    # =========================================================================
    
    def check_equilibrium_stability(
        self,
        perturbation: float = 0.01,
        t_max: float = 200,
        settling_threshold: float = 0.001
    ) -> ValidationResult:
        """
        Verify endemic equilibrium is stable by perturbing and integrating.
        
        For R₀ > 1, endemic equilibrium should be stable.
        Small perturbations should decay back to equilibrium.
        
        Parameters
        ----------
        perturbation : float
            Size of perturbation to apply (default: 0.01)
        t_max : float
            Integration time to check stability (default: 200)
        settling_threshold : float
            Max distance from equilibrium to consider "settled" (default: 0.001)
            
        Returns
        -------
        ValidationResult
            Pass if perturbations decay to equilibrium
            
        Examples
        --------
        >>> from kr_epi.models.ode_counts import SIRDemographyCounts
        >>> sir = SIRDemographyCounts(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
        >>> validator = ModelValidator(sir)
        >>> result = validator.check_equilibrium_stability()
        """
        try:
            # Check if model has endemic_equilibrium method
            if not hasattr(self.model, 'endemic_equilibrium'):
                return ValidationResult(
                    test_name="Equilibrium Stability",
                    passed=False,
                    message="Model does not have endemic_equilibrium() method",
                    details={"reason": "Missing method"}
                )
            
            # Check if model has labels property
            # <-- FIX: Added this check for 'labels' property
            if not hasattr(self.model, 'labels'):
                return ValidationResult(
                    test_name="Equilibrium Stability",
                    passed=False,
                    message="Model does not have 'labels' property",
                    details={"reason": "Missing property"}
                )
            
            # Get equilibrium
            eq = self.model.endemic_equilibrium()
            
            if eq is None or (isinstance(eq, dict) and eq.get('X') is None):
                return ValidationResult(
                    test_name="Equilibrium Stability",
                    passed=True,
                    message="No endemic equilibrium exists (R₀ ≤ 1 or not feasible)",
                    details={"reason": "No equilibrium to test"}
                )
            
            # Extract equilibrium values
            if isinstance(eq, dict):
                labels = self.model.labels  # <-- FIX: Changed from state_labels()
                y_eq = np.array([eq[label] for label in labels if label in eq and label != 'N' and label != 'R0'])
            else:
                y_eq = eq
            
            # Apply small perturbation
            y0_perturbed = y_eq * (1 + perturbation)
            
            # Integrate from perturbed state
            t, y = self.model.integrate(
                # <-- FIX: Changed from state_labels() to labels
                {label: y0_perturbed[i] for i, label in enumerate(self.model.labels) if i < len(y0_perturbed)},
                t_span=(0, t_max)
            )
            
            # Check if system returns to equilibrium
            y_final = y[:, -1]
            distance_to_eq = np.abs(y_final - y_eq).max()
            
            passed = distance_to_eq < settling_threshold
            
            if passed:
                message = f"Equilibrium is stable: perturbation decayed to {distance_to_eq:.2e}"
            else:
                message = f"Equilibrium may be unstable: distance {distance_to_eq:.2e} > {settling_threshold:.2e}"
            
            return ValidationResult(
                test_name="Equilibrium Stability",
                passed=passed,
                message=message,
                details={
                    "equilibrium": y_eq.tolist(),
                    "final_state": y_final.tolist(),
                    "distance": distance_to_eq,
                    "threshold": settling_threshold,
                    "perturbation_applied": perturbation
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Equilibrium Stability",
                passed=False,
                message=f"Test failed with error: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    # =========================================================================
    # 4. FINAL SIZE RELATION (for closed SIR)
    # =========================================================================
    
    def check_final_size(
        self,
        y0: Optional[Dict[str, float]] = None,
        t_max: float = 200
    ) -> ValidationResult:
        """
        Verify final size relation for closed SIR models.
        
        The implicit equation: s_∞ = s₀ exp(-R₀(1 - s_∞))
        
        Parameters
        ----------
        y0 : dict, optional
            Initial conditions (default: s0=0.99, i0=0.01)
        t_max : float
            Integration time (default: 200)
            
        Returns
        -------
        ValidationResult
            Pass if simulation matches analytical final size
            
        References
        ----------
        K&R Equation 2.17
        """
        try:
            # Check if this is a closed population model (no births/deaths)
            params = self.model.params if hasattr(self.model, 'params') else None
            
            # Check for demography parameters
            has_demography = False
            if params:
                has_demography = any(hasattr(params, attr) for attr in ['v', 'mu'])
            
            if has_demography:
                return ValidationResult(
                    test_name="Final Size Relation",
                    passed=True,
                    message="Skipped: Final size relation only applies to closed populations",
                    details={"reason": "Model has demography (births/deaths)"}
                )
            
            # Check if model has R0 method
            if not hasattr(self.model, 'R0'):
                return ValidationResult(
                    test_name="Final Size Relation",
                    passed=False,
                    message="Model does not have R0() method",
                    details={"reason": "Missing R0 method"}
                )
            
            R0 = self.model.R0()
            
            # Set up initial conditions
            if y0 is None:
                y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            
            s0 = y0.get("S", 0.99)
            
            # Integrate to get final size from simulation
            t, y = self.model.integrate(y0, t_span=(0, t_max), rtol=1e-10)
            s_inf_simulated = y[0, -1]
            
            # Calculate analytical final size (solve implicit equation)
            # s_inf = s0 * exp(-R0 * (1 - s_inf))
            # Use fixed-point iteration
            s_inf_analytical = s0
            for _ in range(100):
                s_new = s0 * np.exp(-R0 * (1.0 - s_inf_analytical))
                if abs(s_new - s_inf_analytical) < 1e-10:
                    break
                s_inf_analytical = s_new
            
            # Compare
            error = abs(s_inf_simulated - s_inf_analytical)
            passed = error < 0.001  # Absolute tolerance of 0.1%
            
            return ValidationResult(
                test_name="Final Size Relation",
                passed=passed,
                message=f"Final size: simulated={s_inf_simulated:.4f}, analytical={s_inf_analytical:.4f}, error={error:.2e}",
                details={
                    "R0": R0,
                    "s0": s0,
                    "s_inf_simulated": s_inf_simulated,
                    "s_inf_analytical": s_inf_analytical,
                    "error": error,
                    "attack_rate_simulated": 1 - s_inf_simulated,
                    "attack_rate_analytical": 1 - s_inf_analytical
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Final Size Relation",
                passed=False,
                message=f"Test failed with error: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    # =========================================================================
    # 5. NUMERICAL ACCURACY
    # =========================================================================
    
    def check_numerical_accuracy(
        self,
        y0: Optional[Dict[str, float]] = None,
        t_max: float = 100,
        rtol_loose: float = 1e-6,
        rtol_tight: float = 1e-12
    ) -> ValidationResult:
        """
        Check that integration results are consistent across tolerances.
        
        Integrates with both loose and tight tolerances and compares results.
        If results differ significantly, numerical accuracy may be insufficient.
        
        Parameters
        ----------
        y0 : dict, optional
            Initial conditions
        t_max : float
            Integration time (default: 100)
        rtol_loose : float
            Loose relative tolerance (default: 1e-6)
        rtol_tight : float
            Tight relative tolerance (default: 1e-12)
            
        Returns
        -------
        ValidationResult
            Pass if results agree within tolerance
        """
        try:
            if y0 is None:
                # <-- FIX: Changed from state_labels() to labels
                labels = self.model.labels if hasattr(self.model, 'labels') else None
                if labels:
                    y0 = {labels[0]: 0.99, labels[1]: 0.01}
                    for label in labels[2:]:
                        y0[label] = 0.0
                else:
                    y0 = {"S": 0.99, "I": 0.01, "R": 0.0}
            
            # Integrate with loose tolerance
            t1, y1 = self.model.integrate(y0, t_span=(0, t_max), rtol=rtol_loose)
            
            # Integrate with tight tolerance
            t2, y2 = self.model.integrate(y0, t_span=(0, t_max), rtol=rtol_tight)
            
            # Interpolate to same timepoints (use t1)
            from scipy.interpolate import interp1d
            y2_interp = np.zeros_like(y1)
            for i in range(y2.shape[0]):
                f = interp1d(t2, y2[i, :], kind='cubic', fill_value='extrapolate')
                y2_interp[i, :] = f(t1)
            
            # Compare results
            max_diff = np.abs(y1 - y2_interp).max()
            relative_diff = max_diff / (np.abs(y1).max() + 1e-10)
            
            passed = relative_diff < 0.01  # 1% difference
            
            return ValidationResult(
                test_name="Numerical Accuracy",
                passed=passed,
                message=f"Max relative difference: {relative_diff:.2e} (loose vs tight tolerance)",
                details={
                    "rtol_loose": rtol_loose,
                    "rtol_tight": rtol_tight,
                    "max_absolute_diff": max_diff,
                    "max_relative_diff": relative_diff,
                    "tolerance": 0.01
                }
            )
            
        except Exception as e:
            return ValidationResult(
                test_name="Numerical Accuracy",
                passed=False,
                message=f"Test failed with error: {str(e)}",
                details={"error": str(e), "type": type(e).__name__}
            )
    
    # =========================================================================
    # COMPREHENSIVE VALIDATION
    # =========================================================================
    
    def validate_all(self) -> List[ValidationResult]:
        """
        Run all validation checks and return list of results.
        
        Returns
        -------
        list of ValidationResult
            Results from all checks
            
        Examples
        --------
        >>> sir = SIR(beta=2.0, gamma=1.0)
        >>> validator = ModelValidator(sir)
        >>> results = validator.validate_all()
        >>> passed = sum(r.passed for r in results)
        >>> print(f"Passed {passed}/{len(results)} tests")
        """
        self._log("=" * 70)
        self._log("MODEL VALIDATION SUITE")
        self._log("=" * 70)
        self._log(f"Model: {self.model.__class__.__name__}")
        if hasattr(self.model, 'R0'):
            self._log(f"R₀: {self.model.R0():.3f}")
        self._log("")
        
        # Reset results
        self.results = []
        
        # Run all checks
        self._add_result(self.check_r0_threshold())
        self._add_result(self.check_conservation())
        self._add_result(self.check_equilibrium_stability())
        self._add_result(self.check_final_size())
        self._add_result(self.check_numerical_accuracy())
        
        # Summary
        passed = sum(r.passed for r in self.results)
        total = len(self.results)
        
        self._log("")
        self._log("=" * 70)
        self._log(f"SUMMARY: {passed}/{total} tests passed")
        self._log("=" * 70)
        
        if passed == total:
            self._log("✓ All validations passed!")
        else:
            self._log("⚠ Some validations failed - review results above")
        
        return self.results
    
    def summary(self) -> str:
        """
        Get a formatted summary of validation results.
        
        Returns
        -------
        str
            Formatted summary string
        """
        if not self.results:
            return "No validation results yet. Run validate_all() first."
        
        passed = sum(r.passed for r in self.results)
        total = len(self.results)
        
        lines = [
            "=" * 70,
            "VALIDATION SUMMARY",
            "=" * 70,
            f"Model: {self.model.__class__.__name__}",
            f"Passed: {passed}/{total} tests",
            ""
        ]
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            lines.append(f"{status} {result.test_name}: {result.message}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================

def validate_model(model, verbose: bool = True) -> List[ValidationResult]:
    """
    Quick validation of a model with default settings.
    
    Parameters
    ----------
    model : ODEBase
        Model to validate
    verbose : bool
        Print detailed output (default: True)
        
    Returns
    -------
    list of ValidationResult
        Results from all validation checks
        
    Examples
    --------
    >>> from kr_epi.models.ode import SIR
    >>> from kr_epi.validation import validate_model
    >>> sir = SIR(beta=2.0, gamma=1.0)
    >>> results = validate_model(sir)
    """
    validator = ModelValidator(model, verbose=verbose)
    return validator.validate_all()


if __name__ == "__main__":
    print(__doc__)