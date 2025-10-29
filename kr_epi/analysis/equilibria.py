"""
Equilibrium analysis for epidemic models.

This module provides functions for calculating endemic equilibria,
final epidemic sizes, and vaccination thresholds.

Based on Keeling & Rohani (2008), Chapter 2.
"""

import numpy as np
from typing import Optional, Tuple


# ============================================================================
# SIS Model Equilibria
# ============================================================================

def sis_equilibria(beta: float, gamma: float, N: float = 1.0, 
                   mixing: str = "frequency") -> Tuple[float, float]:
    """
    Calculate endemic equilibrium for SIS model.
    
    Args:
        beta: Transmission rate
        gamma: Recovery rate
        N: Population size (default 1.0 for proportions)
        mixing: "frequency" or "density"
    
    Returns:
        (S*, I*): Endemic equilibrium values
        
    Examples:
        >>> S_star, I_star = sis_equilibria(beta=0.3, gamma=0.1)
        >>> print(f"S* = {S_star:.3f}, I* = {I_star:.3f}")
        S* = 0.333, I* = 0.667
    """
    if mixing == "frequency":
        R0 = beta / gamma
    else:  # density
        R0 = (beta * N) / gamma
    
    # Disease-free equilibrium
    if R0 <= 1.0:
        return N, 0.0
    
    # Endemic equilibrium
    if mixing == "frequency":
        S_star = N / R0
        I_star = N * (1.0 - 1.0/R0)
    else:  # density
        S_star = gamma / beta
        I_star = N - S_star
    
    return S_star, I_star


# ============================================================================
# SIR Model - Final Size
# ============================================================================

def sir_final_size(s0: float, R0: float, tol: float = 1e-10, 
                   max_iter: int = 100) -> float:
    """
    Calculate final epidemic size for SIR model using implicit equation.
    
    Solves: s_∞ = s₀ · exp(-R₀(1 - s_∞))
    
    Args:
        s0: Initial susceptible fraction
        R0: Basic reproduction number
        tol: Convergence tolerance
        max_iter: Maximum iterations
    
    Returns:
        s_inf: Final susceptible fraction
        
    Examples:
        >>> s_inf = sir_final_size(s0=0.99, R0=5.0)
        >>> attack_rate = 1 - s_inf
        >>> print(f"Attack rate: {attack_rate:.1%}")
        Attack rate: 99.3%
    """
    if R0 <= 1.0:
        # No epidemic
        return s0
    
    # Fixed-point iteration
    s_inf = s0
    for _ in range(max_iter):
        s_new = s0 * np.exp(-R0 * (1.0 - s_inf))
        
        if abs(s_new - s_inf) < tol:
            return s_new
        
        s_inf = s_new
    
    # If didn't converge, return best estimate
    return s_inf


def sir_equilibria_closed(beta: float, gamma: float) -> Tuple[float, float, float]:
    """
    Equilibrium for closed SIR model (no births/deaths).
    
    In closed SIR, the only equilibrium is disease-free with all in R.
    The exact distribution depends on initial conditions.
    
    Args:
        beta: Transmission rate
        gamma: Recovery rate
    
    Returns:
        (S*, I*, R*): Disease-free equilibrium (I*=0, S*+R*=1)
        
    Note:
        For closed SIR, use sir_final_size() for epidemic predictions.
    """
    # Only disease-free equilibrium (no births to replenish S)
    # Actual final state depends on initial conditions
    return 1.0, 0.0, 0.0


# ============================================================================
# SIRS Model Equilibria
# ============================================================================

def sirs_equilibria(beta: float, gamma: float, omega: float, 
                    N: float = 1.0, mixing: str = "frequency") -> Tuple[float, float, float]:
    """
    Calculate endemic equilibrium for SIRS model.
    
    Args:
        beta: Transmission rate
        gamma: Recovery rate
        omega: Rate of waning immunity
        N: Population size
        mixing: "frequency" or "density"
    
    Returns:
        (S*, I*, R*): Endemic equilibrium values
        
    Examples:
        >>> S, I, R = sirs_equilibria(beta=0.5, gamma=0.1, omega=0.05)
        >>> print(f"S* = {S:.3f}, I* = {I:.3f}, R* = {R:.3f}")
    """
    if mixing == "frequency":
        R0 = beta / gamma
    else:
        R0 = (beta * N) / gamma
    
    # Disease-free equilibrium
    if R0 <= 1.0:
        return N, 0.0, 0.0
    
    # Endemic equilibrium
    if mixing == "frequency":
        S_star = N / R0
        I_star = (omega / (omega + gamma)) * N * (1.0 - 1.0/R0)
        R_star = (gamma / (omega + gamma)) * N * (1.0 - 1.0/R0)
    else:  # density
        S_star = gamma / beta
        I_star = (omega / (omega + gamma)) * (N - S_star)
        R_star = (gamma / (omega + gamma)) * (N - S_star)
    
    return S_star, I_star, R_star


# ============================================================================
# SEIR Model Equilibria
# ============================================================================

def seir_equilibria_closed(beta: float, sigma: float, gamma: float) -> Tuple[float, float, float, float]:
    """
    Equilibrium for closed SEIR model (no births/deaths).
    
    Args:
        beta: Transmission rate
        sigma: Rate of progression E→I
        gamma: Recovery rate
    
    Returns:
        (S*, E*, I*, R*): Disease-free equilibrium
    """
    return 1.0, 0.0, 0.0, 0.0


# ============================================================================
# Vaccination Thresholds
# ============================================================================

def critical_vaccination_coverage(R0: float) -> float:
    """
    Calculate critical vaccination coverage for herd immunity.
    
    Formula: p_c = 1 - 1/R₀
    
    Args:
        R0: Basic reproduction number
    
    Returns:
        p_c: Minimum fraction to vaccinate (0 to 1)
    
    Raises:
        ValueError: If R₀ ≤ 1
        
    Examples:
        >>> p_c = critical_vaccination_coverage(R0=5.0)
        >>> print(f"Need to vaccinate {p_c:.1%}")
        Need to vaccinate 80.0%
    """
    if R0 <= 1.0:
        raise ValueError(
            f"R₀ = {R0:.2f} ≤ 1. Disease will not spread, "
            "so vaccination threshold is not applicable."
        )
    
    return 1.0 - 1.0/R0


def vaccination_impact(R0: float, coverage: float) -> float:
    """
    Calculate effective R₀ after vaccination.
    
    With vaccination coverage p: R₀_eff = R₀(1 - p)
    
    Args:
        R0: Basic reproduction number
        coverage: Vaccination coverage (0 to 1)
    
    Returns:
        R0_eff: Effective reproduction number after vaccination
        
    Examples:
        >>> R0_eff = vaccination_impact(R0=5.0, coverage=0.8)
        >>> print(f"R₀_eff = {R0_eff:.2f}")
        R₀_eff = 1.00
    """
    if not 0 <= coverage <= 1:
        raise ValueError(f"Coverage must be in [0,1], got {coverage}")
    
    return R0 * (1.0 - coverage)


# ============================================================================
# Stability Analysis
# ============================================================================

def sis_dfe_stability(beta: float, gamma: float) -> str:
    """
    Determine stability of disease-free equilibrium for SIS.
    
    Args:
        beta: Transmission rate
        gamma: Recovery rate
    
    Returns:
        "stable" if DFE is stable (R₀ < 1)
        "unstable" if DFE is unstable (R₀ > 1)
    """
    R0 = beta / gamma
    return "stable" if R0 < 1 else "unstable"


def endemic_exists(R0: float) -> bool:
    """
    Check if endemic equilibrium exists.
    
    Args:
        R0: Basic reproduction number
    
    Returns:
        True if R₀ > 1 (endemic equilibrium exists)
        False if R₀ ≤ 1 (only disease-free equilibrium)
    """
    return R0 > 1.0


# ============================================================================
# Attack Rate Analysis
# ============================================================================

def attack_rate_from_final_size(s0: float, s_inf: float) -> float:
    """
    Calculate attack rate from initial and final susceptible fractions.
    
    Attack rate = fraction of population that got infected during epidemic
    
    Args:
        s0: Initial susceptible fraction
        s_inf: Final susceptible fraction
    
    Returns:
        Attack rate (0 to 1)
        
    Examples:
        >>> attack_rate = attack_rate_from_final_size(s0=0.99, s_inf=0.01)
        >>> print(f"Attack rate: {attack_rate:.1%}")
        Attack rate: 98.0%
    """
    return s0 - s_inf


def sir_attack_rate(s0: float, R0: float) -> float:
    """
    Calculate attack rate for SIR epidemic.
    
    Combines final size calculation with attack rate.
    
    Args:
        s0: Initial susceptible fraction
        R0: Basic reproduction number
    
    Returns:
        Attack rate (fraction infected)
        
    Examples:
        >>> attack = sir_attack_rate(s0=0.99, R0=5.0)
        >>> print(f"Attack rate: {attack:.1%}")
        Attack rate: 99.3%
    """
    s_inf = sir_final_size(s0, R0)
    return attack_rate_from_final_size(s0, s_inf)


# ============================================================================
# Helper Functions
# ============================================================================

def R0_from_parameters(beta: float, gamma: float, N: float = 1.0, 
                       mixing: str = "frequency") -> float:
    """
    Calculate R₀ from model parameters.
    
    Args:
        beta: Transmission rate
        gamma: Recovery rate
        N: Population size
        mixing: "frequency" or "density"
    
    Returns:
        R0: Basic reproduction number
    """
    if mixing == "frequency":
        return beta / gamma
    else:
        return (beta * N) / gamma


def summarize_equilibrium(model_name: str, equilibrium: Tuple, R0: float) -> str:
    """
    Create a human-readable summary of equilibrium.
    
    Args:
        model_name: Name of model (e.g., "SIS", "SIRS")
        equilibrium: Tuple of equilibrium values
        R0: Basic reproduction number
    
    Returns:
        Formatted string summary
    """
    if model_name == "SIS":
        S, I = equilibrium
        return (
            f"SIS Equilibrium (R₀ = {R0:.2f}):\n"
            f"  S* = {S:.4f}\n"
            f"  I* = {I:.4f}\n"
            f"  Prevalence: {I*100:.2f}%"
        )
    elif model_name == "SIR":
        s_inf = equilibrium
        attack = 1 - s_inf
        return (
            f"SIR Final Size (R₀ = {R0:.2f}):\n"
            f"  s_∞ = {s_inf:.4f}\n"
            f"  Attack rate: {attack*100:.2f}%"
        )
    elif model_name == "SIRS":
        S, I, R = equilibrium
        return (
            f"SIRS Equilibrium (R₀ = {R0:.2f}):\n"
            f"  S* = {S:.4f}\n"
            f"  I* = {I:.4f}\n"
            f"  R* = {R:.4f}\n"
            f"  Prevalence: {I*100:.2f}%"
        )
    else:
        return f"{model_name} Equilibrium: {equilibrium}"