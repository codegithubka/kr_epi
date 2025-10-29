from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable, Tuple, Dict, Sequence
import numpy as np
from kr_epi.models.base import ODEBase


Mixing = Literal["frequency", "density"]

def _incidence(beta: float, X: float, Y: float, N: float, mixing: Mixing) -> float:
    # returns λX (total new infections per unit time)
    if N <= 0:
        return 0.0
    if mixing == "frequency":
        return beta * X * Y / N
    else:  # density
        return beta * X * Y
    
    
@dataclass(frozen=True)
class SIRDemographyCountsParams:
    beta: float         # transmission parameter
    gamma: float        # recovery rate
    v: float            # per-capita birth rate
    mu: float           # per-capita death rate (all classes)
    mixing: Mixing = "frequency"
    vacc_p: float = 0.0 # vaccination at birth coverage in [0,1]
    
    
class SIRDemographyCounts(ODEBase):
    """Counts model (X,Y,Z) with births vN and deaths mu*state.

    dX = v(1-p)N - inc - mu X
    dY = inc      - (gamma + mu) Y
    dZ = v p N    + gamma Y - mu Z
    where inc = β X Y / N  (freq); inc = β X Y (density)
    
    At equilibrium with v = mu (balanced demography), N* = v/mu = constant.
    Endemic equilibrium (K&R eq 2.19): Y* = (mu/beta)*(R0 - 1) when R0 > 1.
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0):
        # Parameter validation
        if beta <= 0:
            raise ValueError("beta must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if v < 0:
            raise ValueError("v (birth rate) must be non-negative")
        if mu <= 0:
            raise ValueError("mu (death rate) must be positive")
        if not 0 <= vacc_p < 1:
            raise ValueError("vacc_p must be in [0, 1)")
            
        self.params = SIRDemographyCountsParams(beta, gamma, v, mu, mixing, vacc_p)

    def state_labels(self) -> Sequence[str]:
        return ("X", "Y", "Z")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        X, Y, Z = y
        p = self.params
        N = max(X + Y + Z, 0.0)
        beta = beta_fn(t) if beta_fn is not None else p.beta
        inc = _incidence(beta, X, Y, N, p.mixing)

        births_S = p.v * (1.0 - p.vacc_p) * N
        births_R = p.v * p.vacc_p * N

        dX = births_S - inc - p.mu * X
        dY = inc - (p.gamma + p.mu) * Y
        dZ = births_R + p.gamma * Y - p.mu * Z
        return np.array([dX, dY, dZ], dtype=float)
    
    def R0(self, N0: float = None) -> float:
        """
        Calculate basic reproduction number.
        
        For frequency-dependent: R0 = beta / (gamma + mu)
        For density-dependent: R0 = beta * N0 / (gamma + mu)
        
        Parameters
        ----------
        N0 : float, optional
            Initial population size (required for density-dependent mixing)
            
        Returns
        -------
        float
            Basic reproduction number
        """
        p = self.params
        if p.mixing == "frequency":
            return p.beta / (p.gamma + p.mu)
        else:  # density: note the dependence on N
            if N0 is None:
                raise ValueError("N0 required for density-dependent R0 calculation")
            return p.beta * N0 / (p.gamma + p.mu)
    
    def endemic_equilibrium(self, N0: float = None) -> Dict[str, float]:
        """
        Calculate endemic equilibrium (K&R Section 2.1.2.1, eq 2.19).
        
        For frequency-dependent transmission:
            X* = (gamma + mu) / beta = N* / R0
            Y* = (mu / beta) * (R0 - 1)
            Z* = N* - X* - Y*
            N* = v / mu (carrying capacity)
            
        Parameters
        ----------
        N0 : float, optional
            Population size for density-dependent (defaults to v/mu)
            
        Returns
        -------
        dict
            Equilibrium values {'X': X*, 'Y': Y*, 'Z': Z*, 'N': N*}
            Returns None values if R0 <= 1 (no endemic equilibrium)
        """
        p = self.params
        
        # Calculate carrying capacity
        if p.v > 0 and p.mu > 0:
            N_star = p.v / p.mu
        elif N0 is not None:
            N_star = N0
        else:
            N_star = 1.0
        
        # Calculate R0
        R0_val = self.R0(N_star) if p.mixing == "density" else self.R0()
        
        # No endemic equilibrium if R0 <= 1
        if R0_val <= 1:
            return {'X': None, 'Y': None, 'Z': None, 'N': None, 'R0': R0_val}
        
        # Endemic equilibrium (K&R eq 2.19)
        if p.mixing == "frequency":
            X_star = N_star / R0_val  # X* / N* = 1/R0
            Y_star = (p.mu / p.beta) * (R0_val - 1)
        else:
            X_star = (p.gamma + p.mu) / p.beta
            Y_star = (p.v - p.mu * X_star) / (p.gamma + p.mu)
        
        Z_star = N_star - X_star - Y_star
        
        return {
            'X': X_star,
            'Y': Y_star, 
            'Z': Z_star,
            'N': N_star,
            'R0': R0_val
        }
    
    def check_conservation(self, y: np.ndarray) -> float:
        """
        Check conservation: dN/dt should equal (v - mu)*N.
        
        Returns the difference between actual dN/dt and expected (v - mu)*N.
        Should be close to zero for correct implementation.
        """
        X, Y, Z = y
        N = X + Y + Z
        
        # Calculate actual dN/dt
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        
        # Expected dN/dt = (v - mu) * N
        dN_expected = (self.params.v - self.params.mu) * N
        
        return abs(dN_actual - dN_expected)
        

# ------------------------- SIS + demography (counts) -------------------------

@dataclass(frozen=True)
class SISDemographyCountsParams:
    beta: float         # transmission parameter
    gamma: float        # recovery rate
    v: float            # per-capita birth rate
    mu: float           # per-capita death rate (all classes)
    mixing: Mixing = "frequency"
    
class SISDemographyCounts(ODEBase):
    """Counts model (X,Y). No immune class; recovered return to X.

    dX = v N - inc + gamma Y - mu X
    dY = inc   - (gamma + mu) Y
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float, mixing: Mixing = "frequency"):
        # Parameter validation
        if beta <= 0:
            raise ValueError("beta must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if v < 0:
            raise ValueError("v (birth rate) must be non-negative")
        if mu <= 0:
            raise ValueError("mu (death rate) must be positive")
            
        self.params = SISDemographyCountsParams(beta, gamma, v, mu, mixing)

    def state_labels(self) -> Sequence[str]:
        return ("X", "Y")

    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        X, Y = y
        p = self.params
        N = max(X + Y, 0.0)
        beta = beta_fn(t) if beta_fn is not None else p.beta
        inc = _incidence(beta, X, Y, N, p.mixing)

        dX = p.v * N - inc + p.gamma * Y - p.mu * X
        dY = inc - (p.gamma + p.mu) * Y
        return np.array([dX, dY], dtype=float)

    def R0(self, N0: float = None) -> float:
        """Calculate basic reproduction number."""
        p = self.params
        if p.mixing == "frequency":
            return p.beta / (p.gamma + p.mu)
        else:
            if N0 is None:
                raise ValueError("N0 required for density-dependent R0 calculation")
            return p.beta * N0 / (p.gamma + p.mu)
    
    def endemic_equilibrium(self, N0: float = None) -> Dict[str, float]:
        """
        Calculate endemic equilibrium for SIS model.
        
        For frequency-dependent: Y* = (mu/beta) * (R0 - 1)
        For density-dependent: Different formula
        
        Returns
        -------
        dict
            Equilibrium values {'X': X*, 'Y': Y*, 'N': N*}
        """
        p = self.params
        
        # Calculate carrying capacity
        if p.v > 0 and p.mu > 0:
            N_star = p.v / p.mu
        elif N0 is not None:
            N_star = N0
        else:
            N_star = 1.0
        
        # Calculate R0
        R0_val = self.R0(N_star) if p.mixing == "density" else self.R0()
        
        # No endemic equilibrium if R0 <= 1
        if R0_val <= 1:
            return {'X': None, 'Y': None, 'N': None, 'R0': R0_val}
        
        # Endemic equilibrium (from setting dY/dt = 0 at equilibrium)
        # At equilibrium: beta*X*Y/N = (gamma + mu)*Y
        # => X* = (gamma + mu) * N / beta (frequency)
        # => X* = (gamma + mu) / beta (density)
        if p.mixing == "frequency":
            # X* / N* = 1/R0
            X_star = N_star / R0_val
        else:
            X_star = (p.gamma + p.mu) / p.beta
        
        Y_star = N_star - X_star
        
        return {
            'X': X_star,
            'Y': Y_star,
            'N': N_star,
            'R0': R0_val
        }
    
    def check_conservation(self, y: np.ndarray) -> float:
        """Check conservation: dN/dt should equal (v - mu)*N."""
        X, Y = y
        N = X + Y
        
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        dN_expected = (self.params.v - self.params.mu) * N
        
        return abs(dN_actual - dN_expected)

    
# ------------------------- SEIR + demography (counts) ------------------------

@dataclass(frozen=True)
class SEIRDemographyCountsParams:
    beta: float
    sigma: float
    gamma: float
    v: float
    mu: float
    mixing: Mixing = "frequency"
    vacc_p: float = 0.0

class SEIRDemographyCounts(ODEBase):
    """Counts model (X,E,Y,Z) with births and deaths.

    dX = v(1-p)N - inc           - mu X
    dE = inc      - (sigma + mu) E
    dY = sigma E  - (gamma + mu) Y
    dZ = v p N    + gamma Y      - mu Z
    """
    def __init__(self, beta: float, sigma: float, gamma: float, v: float, mu: float,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0):
        # Parameter validation
        if beta <= 0:
            raise ValueError("beta must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if gamma <= 0:
            raise ValueError("gamma must be positive")
        if v < 0:
            raise ValueError("v (birth rate) must be non-negative")
        if mu <= 0:
            raise ValueError("mu (death rate) must be positive")
        if not 0 <= vacc_p < 1:
            raise ValueError("vacc_p must be in [0, 1)")
            
        self.params = SEIRDemographyCountsParams(beta, sigma, gamma, v, mu, mixing, vacc_p)

    def state_labels(self) -> Sequence[str]:
        return ("X", "E", "Y", "Z")

    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        X, E, Y, Z = y
        p = self.params
        N = max(X + E + Y + Z, 0.0)
        beta = beta_fn(t) if beta_fn is not None else p.beta
        inc = _incidence(beta, X, Y, N, p.mixing)

        births_S = p.v * (1.0 - p.vacc_p) * N
        births_R = p.v * p.vacc_p * N

        dX = births_S - inc - p.mu * X
        dE = inc - (p.sigma + p.mu) * E
        dY = p.sigma * E - (p.gamma + p.mu) * Y
        dZ = births_R + p.gamma * Y - p.mu * Z
        return np.array([dX, dE, dY, dZ], dtype=float)

    def R0(self, N0: float = None) -> float:
        """
        Calculate basic reproduction number.
        
        R0 = (beta) * (sigma/(mu+sigma)) * (1/(mu+gamma)) for frequency
        R0 = (beta * N0) * (sigma/(mu+sigma)) * (1/(mu+gamma)) for density
        """
        p = self.params
        # R0 = (beta * X*/N*) * (1/(mu+sigma)) * (sigma/(mu+gamma)) for frequency;
        # for density, replace beta*X*/N* with beta*X* = beta*N0
        if p.mixing == "frequency":
            return (p.beta) * (p.sigma / (p.mu + p.sigma)) * (1.0 / (p.mu + p.gamma))
        else:
            if N0 is None:
                raise ValueError("N0 required for density-dependent R0 calculation")
            return (p.beta * N0) * (p.sigma / (p.mu + p.sigma)) * (1.0 / (p.mu + p.gamma))
    
    def check_conservation(self, y: np.ndarray) -> float:
        """Check conservation: dN/dt should equal (v - mu)*N."""
        N = y.sum()
        
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        dN_expected = (self.params.v - self.params.mu) * N
        
        return abs(dN_actual - dN_expected)