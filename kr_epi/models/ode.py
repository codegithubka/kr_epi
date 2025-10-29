"""
Improved ODE models for closed populations (no demography).
Based on Keeling & Rohani (2008) - Modeling Infectious Diseases.
"""
from dataclasses import dataclass
from typing import Callable, Optional, Literal, Sequence
import numpy as np
from kr_epi.models.base import ODEBase

Mixing = Literal["frequency", "density"]  # Fixed typo


def _foi(beta: float, I: float, N: float, mixing: Mixing) -> float:
    """
    Calculate force of infection (lambda).
    
    Args:
        beta: Transmission rate parameter
        I: Number/fraction of infectious individuals
        N: Total population (for frequency-dependent)
        mixing: "frequency" (density-independent) or "density" (mass-action)
    
    Returns:
        Force of infection lambda
    """
    if mixing == "frequency":
        # Frequency-dependent: lambda = beta * I/N
        return beta * I / N if N > 0 else 0.0
    else:
        # Density-dependent (mass-action): lambda = beta * I
        return beta * I


# ============================================================================
# SI Model
# ============================================================================

@dataclass(frozen=True)
class SIParams:
    """Parameters for the SI model."""
    beta: float
    mixing: Mixing = "frequency"
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SI(ODEBase):
    """
    Simple SI model (no recovery).
    
    dS/dt = -lambda * S
    dI/dt = +lambda * S
    
    where lambda = beta * I (density) or beta * I/N (frequency)
    
    """
    
    def __init__(self, beta: float, mixing: Mixing = "frequency"):
        self.params = SIParams(beta=beta, mixing=mixing)
        
        
    @property
    def labels(self) -> list[str]:
        return ("S", "I")
    
    def rhs(self, t: float, y: np.ndarray, 
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I = y
        N = S + I
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S
        dI = +lam * S
        return np.array([dS, dI])
    
    def R0(self, N0: float = 1.0) -> float:
        """
        Basic reproduction number for SI model.
        Note: R₀ is technically infinite for SI (no recovery).
        Returns transmission rate scaled by mixing type.
        
        Args:
            N0: Total population size (only matters for density-dependent)
        
        Returns:
            beta (frequency) or beta*N0 (density)
        """
        if self.params.mixing == "frequency":
            return self.params.beta
        else:
            return self.params.beta * N0


# ============================================================================
# SIS Model
# ============================================================================

@dataclass(frozen=True)
class SISParams:
    beta: float
    gamma: float
    mixing: Mixing = "frequency"
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SIS(ODEBase):
    """
    SIS model with recovery but no immunity.
    
    dS/dt = -lambda * S + gamma * I
    dI/dt = +lambda * S - gamma * I
    
    R0 = beta/gamma (frequency) or beta*N/gamma (density)
    """
    
    def __init__(self, beta: float, gamma: float, mixing: Mixing = "frequency"):
        self.params = SISParams(beta=beta, gamma=gamma, mixing=mixing)
        
    @property
    def labels(self) -> list[str]:
        return ("S", "I")
    
    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I = y
        N = S + I
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S + p.gamma * I
        dI = +lam * S - p.gamma * I
        return np.array([dS, dI])
    
    def R0(self, N0: float = 1.0) -> float:
        """
        Basic reproduction number: R₀ = β/γ (frequency) or βN/γ (density).
        
        The threshold for endemic persistence:
        - R₀ > 1: Disease persists at endemic equilibrium
        - R₀ ≤ 1: Disease dies out
        
        Args:
            N0: Total population size
        
        Returns:
            Basic reproduction number
        """
        if self.params.mixing == "frequency":
            return self.params.beta / self.params.gamma
        else:
            return (self.params.beta * N0) / self.params.gamma
    
    def endemic_equilibrium(self, N0: float = 1.0) -> tuple[float, float]:
        """
        Return (S*, I*) at endemic equilibrium if R0 > 1.
        Returns (N0, 0) if R0 <= 1.
        """
        R0 = self.R0(N0)
        if R0 <= 1.0:
            return (N0, 0.0)
        
        if self.params.mixing == "frequency":
            I_star = N0 * (1.0 - 1.0/R0)
            S_star = N0 / R0
        else:
            I_star = N0 * (1.0 - self.params.gamma / (self.params.beta * N0))
            S_star = self.params.gamma / self.params.beta
            
        return (S_star, I_star)


# ============================================================================
# SIR Model
# ============================================================================

@dataclass(frozen=True)
class SIRParams:
    beta: float
    gamma: float
    mixing: Mixing = "frequency"
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SIR(ODEBase):
    """
    Classic SIR model (closed population).
    
    dS/dt = -lambda * S
    dI/dt = +lambda * S - gamma * I
    dR/dt = +gamma * I
    
    R0 = beta/gamma (frequency) or beta*N/gamma (density)
    """
    
    def __init__(self, beta: float, gamma: float, mixing: Mixing = "frequency"):
        self.params = SIRParams(beta=beta, gamma=gamma, mixing=mixing)
        
    @property
    def labels(self) -> list[str]:
        return ("S", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I, R = y
        N = S + I + R
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S
        dI = +lam * S - p.gamma * I
        dR = +p.gamma * I
        return np.array([dS, dI, dR])
    
    def R0(self, N0: float = 1.0) -> float:
        """
        Basic reproduction number: R₀ = β/γ (frequency) or βN/γ (density).
        
        Interpretation:
        - Average number of secondary infections from one infected individual
        - in a completely susceptible population
        
        Epidemic threshold:
        - R₀ > 1: Epidemic occurs
        - R₀ ≤ 1: No epidemic
        
        Args:
            N0: Total population size
        
        Returns:
            Basic reproduction number
        """
        if self.params.mixing == "frequency":
            return self.params.beta / self.params.gamma
        else:
            return (self.params.beta * N0) / self.params.gamma
    
    def final_size(self, s0: float, R0: Optional[float] = None) -> float:
        """
        Implicit final size relation for SIR model (frequency-dependent).
        Given initial susceptible fraction s0, returns final susceptible s_inf.
        
        Uses: s_inf = s0 * exp(-R0 * (1 - s_inf))
        
        Solved iteratively using fixed-point iteration.
        """
        if R0 is None:
            R0 = self.R0()
        
        if self.params.mixing != "frequency":
            raise ValueError("Final size formula only valid for frequency-dependent mixing")
        
        # Fixed-point iteration
        s_inf = s0
        for _ in range(100):
            s_new = s0 * np.exp(-R0 * (1.0 - s_inf))
            if abs(s_new - s_inf) < 1e-10:
                break
            s_inf = s_new
        
        return s_inf


# ============================================================================
# SIRS Model
# ============================================================================

@dataclass(frozen=True)
class SIRSParams:
    beta: float
    gamma: float
    omega: float  # Rate of waning immunity
    mixing: Mixing = "frequency"
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.omega < 0:
            raise ValueError(f"omega must be non-negative, got {self.omega}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SIRS(ODEBase):
    """
    SIRS model with waning immunity.
    
    dS/dt = -lambda * S + omega * R
    dI/dt = +lambda * S - gamma * I
    dR/dt = +gamma * I - omega * R
    
    R0 = beta/gamma (frequency) or beta*N/gamma (density)
    """
    
    def __init__(self, beta: float, gamma: float, omega: float, 
                 mixing: Mixing = "frequency"):
        self.params = SIRSParams(beta=beta, gamma=gamma, omega=omega, mixing=mixing)
        
    @property
    def labels(self) -> list[str]:
        return ("S", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I, R = y
        N = S + I + R
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S + p.omega * R
        dI = +lam * S - p.gamma * I
        dR = +p.gamma * I - p.omega * R
        return np.array([dS, dI, dR])
    
    def R0(self, N0: float = 1.0) -> float:
        """
        Basic reproduction number: R₀ = β/γ (frequency) or βN/γ (density).
        
        Note: R₀ is the same as SIR model. The waning immunity (ω) affects
        the endemic equilibrium but not the invasion threshold.
        
        Args:
            N0: Total population size
        
        Returns:
            Basic reproduction number
        """
        if self.params.mixing == "frequency":
            return self.params.beta / self.params.gamma
        else:
            return (self.params.beta * N0) / self.params.gamma
    
    def endemic_equilibrium(self, N0: float = 1.0) -> tuple[float, float, float]:
        """
        Return (S*, I*, R*) at endemic equilibrium if R0 > 1.
        Returns (N0, 0, 0) if R0 <= 1.
        """
        R0 = self.R0(N0)
        if R0 <= 1.0:
            return (N0, 0.0, 0.0)
        
        p = self.params
        if self.params.mixing == "frequency":
            S_star = N0 / R0
            I_star = p.omega * N0 * (1.0 - 1.0/R0) / (p.gamma + p.omega)
            R_star = N0 - S_star - I_star
        else:
            S_star = p.gamma / p.beta
            I_star = p.omega * (N0 - S_star) / (p.gamma + p.omega)
            R_star = N0 - S_star - I_star
            
        return (S_star, I_star, R_star)


# ============================================================================
# SEIR Model
# ============================================================================

@dataclass(frozen=True)
class SEIRParams:
    beta: float
    sigma: float  # Rate of progression from E to I
    gamma: float
    mixing: Mixing = "frequency"
    
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SEIR(ODEBase):
    """
    SEIR model with latent period.
    
    dS/dt = -lambda * S
    dE/dt = +lambda * S - sigma * E
    dI/dt = +sigma * E - gamma * I
    dR/dt = +gamma * I
    
    R0 = beta/gamma (frequency) or beta*N/gamma (density)
    Note: Latent period doesn't affect R0, but affects dynamics.
    """
    
    def __init__(self, beta: float, sigma: float, gamma: float,
                 mixing: Mixing = "frequency"):
        self.params = SEIRParams(beta=beta, sigma=sigma, gamma=gamma, mixing=mixing)
        
    @property
    def labels(self) -> list[str]:
        return ("S", "E", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, E, I, R = y
        N = S + E + I + R
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S
        dE = +lam * S - p.sigma * E
        dI = +p.sigma * E - p.gamma * I
        dR = +p.gamma * I
        return np.array([dS, dE, dI, dR])
    
    # For closed SEIR (no demography)
    def R0(self, N0: float = 1.0) -> float:
        """
        Calculate basic reproduction number for SEIR model.
        
        For closed population (no births/deaths):
            R₀ = β/γ
        
        The latent period (1/σ) affects dynamics but not R₀ in closed models
        because all exposed individuals eventually become infectious.
        
        Parameters
        ----------
        N0 : float
            Total population size (only for density-dependent mixing)
            
        Returns
        -------
        float
            Basic reproduction number
            
        Notes
        -----
        In models with demography (μ > 0), the latent period DOES affect R₀
        because individuals can die while in E class. See SEIRDemographyCounts
        for that case.
        
        References
        ----------
        Keeling & Rohani (2008), Section 2.6, Box 2.5
        """
        if self.params.mixing == "frequency":
            return self.params.beta / self.params.gamma
        else:
            return (self.params.beta * N0) / self.params.gamma
        
    def mean_generation_time(self) -> float:
        """
        Mean generation time = latent period + infectious period.
        T_g = 1/sigma + 1/gamma
        """
        return 1.0/self.params.sigma + 1.0/self.params.gamma


# ============================================================================
# SEIRS Model (bonus)
# ============================================================================

@dataclass(frozen=True)
class SEIRSParams:
    beta: float
    sigma: float
    gamma: float
    omega: float
    mixing: Mixing = "frequency"
    
    
    def __post_init__(self):
        if self.beta < 0:
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be positive, got {self.sigma}")
        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")


class SEIRS(ODEBase):
    """
    SEIRS model with latent period and waning immunity.
    
    dS/dt = -lambda * S + omega * R
    dE/dt = +lambda * S - sigma * E
    dI/dt = +sigma * E - gamma * I
    dR/dt = +gamma * I - omega * R
    """
    
    def __init__(self, beta: float, sigma: float, gamma: float, omega: float,
                 mixing: Mixing = "frequency"):
        self.params = SEIRSParams(beta=beta, sigma=sigma, gamma=gamma, 
                                   omega=omega, mixing=mixing)
        
    @property
    def labels(self) -> list[str]:
        return ("S", "E", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, E, I, R = y
        N = S + E + I + R
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi(beta, I, N, p.mixing)
        
        dS = -lam * S + p.omega * R
        dE = +lam * S - p.sigma * E
        dI = +p.sigma * E - p.gamma * I
        dR = +p.gamma * I - p.omega * R
        return np.array([dS, dE, dI, dR])
    
    def R0(self, N0: float = 1.0) -> float:
        """Basic reproduction number."""
        if self.params.mixing == "frequency":
            return self.params.beta / self.params.gamma
        else:
            return (self.params.beta * N0) / self.params.gamma