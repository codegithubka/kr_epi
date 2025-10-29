from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable, Tuple, Dict, Sequence
import numpy as np
from kr_epi.models.base import ODEBase
import warnings


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

    @property
    def labels(self) -> list[str]:
        return ["X", "Y", "Z"] # (or S, E, I, R)
    
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
        
        From Keeling & Rohani (2008), Equation 2.19:
        - Frequency-dependent: R₀ = β/(γ+μ)
        - Density-dependent: R₀ = βN*/(γ+μ) where N* = ν/μ
        
        Parameters
        ----------
        N0 : float, optional
            Override for equilibrium population size. If not provided,
            uses N* = v/mu from demographic equilibrium.
            
        Returns
        -------
        float
            Basic reproduction number
            
        References
        ----------
        Keeling & Rohani (2008), Section 2.1.2, Equation 2.19
        """
        p = self.params
        
        if p.mixing == "frequency":
            return p.beta / (p.gamma + p.mu)
        else:  # density-dependent
            # Use demographic equilibrium N* = v/mu
            if p.v > 0 and p.mu > 0:
                N_star = p.v / p.mu
            elif N0 is not None:
                # Allow override but warn
                N_star = N0
                import warnings
                warnings.warn(
                    f"Using N0={N0} for R0 calculation. "
                    f"Demographic equilibrium would give N*={p.v/p.mu if p.v>0 and p.mu>0 else 'undefined'}"
                )
            else:
                raise ValueError(
                    "Cannot calculate R0 for density-dependent mixing: "
                    "need v>0 and mu>0, or provide N0 explicitly"
                )
            
            return p.beta * N_star / (p.gamma + p.mu)
    
    def endemic_equilibrium(self, N0: float = None) -> Dict[str, float]:
        """
        Calculate endemic equilibrium for SIR model with demography.
        
        From Keeling & Rohani (2008), Section 2.1.2, Equations 2.19-2.20:
        
        At endemic equilibrium (frequency-dependent):
            S*/N* = 1/R₀
            I*/N* = (μ/(γ+μ)) × (1 - 1/R₀)
            R*/N* = (γ/(γ+μ)) × (1 - 1/R₀)
            N* = ν/μ (demographic equilibrium)
        
        Parameters
        ----------
        N0 : float, optional
            Override equilibrium population size
            
        Returns
        -------
        dict
            Dictionary with keys 'X', 'Y', 'Z', 'N', 'R0'
            Values are None if R₀ ≤ 1 (no endemic equilibrium)
            
        References
        ----------
        Keeling & Rohani (2008), Section 2.1.2, Box 2.1
        """
        p = self.params
        
        # Calculate carrying capacity N* = v/mu
        if p.v > 0 and p.mu > 0:
            N_star = p.v / p.mu
        elif N0 is not None:
            N_star = N0
        else:
            N_star = 1.0  # Default for closed population approximation
        
        # Calculate R0
        if p.mixing == "density":
            R0_val = self.R0(N_star)
        else:
            R0_val = self.R0()
        
        # No endemic equilibrium if R0 <= 1
        if R0_val <= 1:
            return {
                'X': None, 
                'Y': None, 
                'Z': None, 
                'N': None, 
                'R0': R0_val
            }
        
        # Endemic equilibrium (K&R Eq 2.19-2.20)
        if p.mixing == "frequency":
            # As fractions of N*
            s_frac = 1.0 / R0_val
            i_frac = (p.mu / (p.gamma + p.mu)) * (1.0 - 1.0/R0_val)
            r_frac = (p.gamma / (p.gamma + p.mu)) * (1.0 - 1.0/R0_val)
            
            # Convert to counts
            X_star = s_frac * N_star
            Y_star = i_frac * N_star  # FIX: Multiply by N_star!
            Z_star = r_frac * N_star
            
        else:  # density-dependent
            # From setting dS/dt = dI/dt = 0
            # S* = (γ+μ)/β
            X_star = (p.gamma + p.mu) / p.beta
            
            # I* from birth-death balance: v*N* = (γ+μ)*I* + μ*(S*+R*)
            # After algebra: I* = (v*N* - μ*S*) / (γ+μ)
            Y_star = (p.v * N_star - p.mu * X_star) / (p.gamma + p.mu)
            
            # R* from total: N* = S* + I* + R*  
            Z_star = N_star - X_star - Y_star
        
        # Validate equilibrium
        if Y_star < 0 or Z_star < 0:
            return {
                'X': None,
                'Y': None, 
                'Z': None,
                'N': None,
                'R0': R0_val,
                'error': 'Negative compartment at equilibrium'
            }
        
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

    @property
    def labels(self) -> list[str]:
        return ["X", "Y"] # (or S, E, I, R)

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

    @property
    def labels(self) -> list[str]:
        return ["X", "E", "Y", "Z"] # (or S, E, I, R)

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
        Calculate basic reproduction number for SEIR with demography.
        
        From Keeling & Rohani (2008), Box 2.5:
        
            R₀ = β/(γ+μ) × σ/(σ+μ)
        
        The factor σ/(σ+μ) represents the probability that an individual
        in the exposed class survives to become infectious.
        
        Parameters
        ----------
        N0 : float, optional
            Population size for density-dependent transmission
            
        Returns
        -------
        float
            Basic reproduction number
            
        References
        ----------
        Keeling & Rohani (2008), Section 2.6, Box 2.5, Equation 2.53
        """
        p = self.params
        
        # Probability of surviving E compartment
        prob_survive_E = p.sigma / (p.sigma + p.mu)
        
        # Average time infectious (accounting for death)
        avg_infectious_time = 1.0 / (p.gamma + p.mu)
        
        if p.mixing == "frequency":
            # R0 = (contacts per time) × (prob survive E) × (time infectious)
            return p.beta * prob_survive_E * avg_infectious_time
        else:  # density
            # Use N* = v/mu
            if p.v > 0 and p.mu > 0:
                N_star = p.v / p.mu
            elif N0 is not None:
                N_star = N0
            else:
                raise ValueError("Need v>0, mu>0 or provide N0 for density-dependent R0")
            
            return p.beta * N_star * prob_survive_E * avg_infectious_time
    
    def check_conservation(self, y: np.ndarray) -> float:
        """Check conservation: dN/dt should equal (v - mu)*N."""
        N = y.sum()
        
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        dN_expected = (self.params.v - self.params.mu) * N
        
        return abs(dN_actual - dN_expected)
    
    
    
@dataclass(frozen=True)
class MSIRDemographyCountsParams(SIRDemographyCountsParams):
    """Parameters for an MSIR model with births, deaths, and counts."""
    alpha: float = 1.0 / 180.0  # Rate of loss of maternal immunity (e.g., 1/180 days)

    def __post_init__(self):
        if self.alpha <= 0:
            raise ValueError(f"alpha (loss of immunity rate) must be positive, got {self.alpha}")


class MSIRDemographyCounts(ODEBase):
    """
    MSIR model with births, deaths, and counts (M-S-I-R).
    Assumes counts (X, Y, Z) and total population N.
    Follows conventions from kr_epi.models.ode_counts.SIRDemographyCounts.

    Equations:
    dN/dt = (v - mu) * N
    dM/dt = v * N * (1 - vacc_p) - (alpha + mu) * M
    dS/dt = alpha * M - lam * S - mu * S
    dI/dt = lam * S - (gamma + mu) * I
    dR/dt = v * N * vacc_p + gamma * I - mu * R
    
    where lam = _incidence(beta, I, S, N, mixing)
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float, alpha: float, *,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0):
        self.params = MSIRDemographyCountsParams(
            beta=beta, gamma=gamma, v=v, mu=mu, mixing=mixing, vacc_p=vacc_p, alpha=alpha
        )

    @property
    def labels(self) -> list[str]:
        return ["M", "S", "I", "R"]

    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        M, S, I, R = y
        N = M + S + I + R
        p = self.params

        current_beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _incidence(current_beta, I, S, N, p.mixing)

        # Births (total N = v * N)
        births_total = p.v * N
        births_to_M = births_total * (1.0 - p.vacc_p)
        births_to_R = births_total * p.vacc_p

        # ODEs
        dM = births_to_M - (p.alpha + p.mu) * M
        dS = p.alpha * M - lam * S - p.mu * S
        dI = lam * S - (p.gamma + p.mu) * I
        dR = births_to_R + p.gamma * I - p.mu * R

        return np.array([dM, dS, dI, dR])

    def R0(self) -> float:
        """
        Calculate R0 for the MSIR model with demography.
        R0 = (beta / (gamma + mu)) * (Fraction Susceptible at DFE)
        Fraction Susceptible = S* / N* = (alpha / (alpha + mu))
        """
        p = self.params
        if p.gamma + p.mu == 0 or p.alpha + p.mu == 0:
            return 0.0 # Prevent division by zero
            
        # R0 = (rate of infection) * (duration of infection) * (fraction susceptible)
        R0 = (p.beta / (p.gamma + p.mu)) * (p.alpha / (p.alpha + p.mu))
        return R0

    def endemic_equilibrium(self) -> Optional[np.ndarray]:
        """
        Calculate the endemic equilibrium state (M*, S*, I*, R*).
        Returns None if R0 <= 1 or equilibrium is not feasible.
        """
        p = self.params
        R0 = self.R0()
        if R0 <= 1.0 or p.v == 0.0 or p.mu == 0.0:
            return None # Endemic state not feasible

        M_star = p.mu * (1.0 - p.vacc_p) / (p.alpha + p.mu)
        
        # We need N=1, v=mu for this simple solution
        if not np.isclose(p.v, p.mu) or p.mixing != 'frequency':
             print("Warning: Endemic equilibrium calculation only implemented for frequency mixing with v=mu.")
             return None

        # S* / N* = 1 / R0
        S_star = 1.0 / R0 # This is S_star fraction
        S_star_abs = S_star * (p.v / p.mu) # Scale to N* (which is v/mu)
        S_star = S_star_abs # Use absolute S* for remaining calcs if N* != 1?
                           # Let's stick to fractions (N*=1, v=mu)
        S_star = 1.0 / R0 

        # I* = (alpha*M* - mu*S*) / (gamma + mu)
        I_star = (p.alpha * M_star - p.mu * S_star) / (p.gamma + p.mu)
        
        # R* = 1.0 - M* - S* - I*
        R_star = 1.0 - M_star - S_star - I_star

        if I_star < 0 or R_star < 0:
            # R0 > 1 but endemic state is not physically feasible (e.g., high vacc)
            return None

        # Scale by N* = v / mu
        N_star = p.v / p.mu
        y_eq = np.array([M_star, S_star, I_star, R_star]) * N_star
        return y_eq


    
    def check_conservation(self, y: np.ndarray) -> float:
        """
        Check population conservation law: dN/dt should equal (v - mu)*N.
        
        Parameters
        ----------
        y : np.ndarray
            State vector [M, S, I, R]
            
        Returns
        -------
        float
            Absolute conservation error (should be near zero)
            
        Warnings
        --------
        If v != mu, warns that endemic equilibrium calculation assumes v=mu.
        """
        p = self.params
        M, S, I, R = y
        N = M + S + I + R
        
        # Calculate actual dN/dt
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        
        # Expected dN/dt = (v - mu) * N
        dN_expected = (p.v - p.mu) * N
        
        # Calculate error
        error = abs(dN_actual - dN_expected)
        
        # Warn if v != mu (since endemic equilibrium assumes balanced demography)
        if not np.isclose(p.v, p.mu, rtol=1e-6):
            warnings.warn(
                f"MSIR model has v={p.v:.4f} != mu={p.mu:.4f}. "
                "Note: endemic_equilibrium() method assumes v=mu for analytical solution.",
                UserWarning
            )
        
        return error
        
            
        
   
   
@dataclass(frozen=True)
class SEIRSDemographyCountsParams(SEIRDemographyCountsParams):
    """Parameters for an SEIRS model with births, deaths, and counts."""
    omega: float = 0.0  # Rate of waning immunity (R -> S)

    def __post_init__(self):
        # Parent validation happens in SEIRDemographyCounts.__init__()
        # Don't call super() - parent doesn't have __post_init__
        if self.omega < 0:
            raise ValueError(f"omega (waning rate) must be non-negative, got {self.omega}")
        
class SEIRSDemographyCounts(SEIRDemographyCounts):
    """
    SEIRS model with births, deaths, and counts (S-E-I-R-S).
    Inherits from SEIRDemographyCounts and adds waning immunity (R -> S).

    Equations:
    dN/dt = (v - mu) * N
    dS/dt = v * N * (1 - vacc_p) - lam * S - mu * S + omega * R  <-- Added omega*R
    dE/dt = lam * S - (sigma + mu) * E
    dI/dt = sigma * E - (gamma + mu) * I
    dR/dt = v * N * vacc_p + gamma * I - mu * R - omega * R      <-- Subtracted omega*R
    
    where lam = _incidence(beta, I, S, N, mixing)
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float, sigma: float, *,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0, omega: float = 0.0):
        # We need to set the params attribute directly to the new dataclass
        self.params = SEIRSDemographyCountsParams(
            beta=beta, gamma=gamma, v=v, mu=mu, sigma=sigma,
            mixing=mixing, vacc_p=vacc_p, omega=omega
        )

    # R0 method is inherited from SEIRDemographyCounts and remains the same
    # R0 = (beta / (gamma + mu)) * (sigma / (sigma + mu))
    # Waning immunity doesn't affect the initial invasion potential.

    def rhs(self, t: float, y: np.ndarray,
            beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, E, I, R = y
        N = S + E + I + R
        # Must cast params to the correct type for mypy/type checkers
        p: SEIRSDemographyCountsParams = self.params # type: ignore

        current_beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _incidence(current_beta, I, S, N, p.mixing)

        # Births
        births_total = p.v * N
        births_to_S = births_total * (1.0 - p.vacc_p)
        births_to_R = births_total * p.vacc_p # Vaccination to R
        
        # Waning immunity
        waning_R_to_S = p.omega * R

        # ODEs
        dS = births_to_S - lam * S - p.mu * S + waning_R_to_S
        dE = lam * S - (p.sigma + p.mu) * E
        dI = p.sigma * E - (p.gamma + p.mu) * I
        dR = births_to_R + p.gamma * I - p.mu * R - waning_R_to_S

        return np.array([dS, dE, dI, dR])

    def endemic_equilibrium(self) -> Optional[np.ndarray]:
        """
        Calculate the endemic equilibrium state (S*, E*, I*, R*).
        Returns None if R0 <= 1 or equilibrium is not feasible.
        """
        # This calculation is more complex due to the R->S feedback.
        # R0 is the same as SEIR
        R0 = self.R0()
        p: SEIRSDemographyCountsParams = self.params # type: ignore
        
        if R0 <= 1.0 or not np.isclose(p.v, p.mu) or p.mixing != 'frequency':
             print("Warning: Endemic equilibrium calculation for SEIRS only implemented for R0>1, frequency mixing, and v=mu.")
             return None


        S_star = 1.0 / R0

        term_E = (p.gamma + p.mu) / p.sigma
        term_I = 1.0
        term_R = p.gamma / (p.mu + p.omega)
        
        I_star_coeff = term_E + term_I + term_R
        
        RHS = (1.0 - (1.0 / R0)) - (p.mu * p.vacc_p) / (p.mu + p.omega)
        
        if RHS <= 0: # If R0 is too low or vacc_p is too high
             return None
             
        I_star = RHS / I_star_coeff
        
        if I_star <= 0:
             return None

        # Back-calculate others
        S_star = 1.0 / R0
        E_star = term_E * I_star
        R_star = (p.mu * p.vacc_p + p.gamma * I_star) / (p.mu + p.omega)
        
        # Check conservation (S+E+I+R should be 1.0)
        N_check = S_star + E_star + I_star + R_star
        if not np.isclose(N_check, 1.0):
             # This indicates a likely math error in the derivation above
             print(f"Warning: SEIRS equilibrium check failed (N={N_check:.4f}). Returning None.")
             return None

        # Scale by N* = v / mu
        N_star = p.v / p.mu
        y_eq = np.array([S_star, E_star, I_star, R_star]) * N_star
        return y_eq     
        
