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
    
    
    
@dataclass(frozen=True)
class MSIRDemographyCountsParams(SIRDemographyCountsParams):
    """Parameters for an MSIR model with births, deaths, and counts."""
    alpha: float = 1.0 / 180.0  # Rate of loss of maternal immunity (e.g., 1/180 days)

    def __post_init__(self):
        super().__post_init__()
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

        # Assume N* = v / mu (if v=mu) or solve for N, but here we
        # solve for fractions and scale by N* if needed.
        # Let's assume v=mu for stable N=N* (though the model supports v!=mu)
        # For simplicity, let's solve assuming N=1 (fractions) if mixing='frequency'
        # N* = N* (cancels)
        # S* = (gamma + mu) / beta   (since S*/N* = 1/R0 for freq mixing)
        # I* = (mu / beta) * (R0 - 1) * (alpha + mu) / alpha  <-- This isn't quite right
        
        # Let's re-derive from steady state equations (setting d/dt=0, N=1, v=mu)
        # 0 = mu * (1 - vacc_p) - (alpha + mu) * M  => M* = mu * (1 - vacc_p) / (alpha + mu)
        # 0 = lam*S* = (gamma + mu) * I* => lam* = (gamma + mu) * I* / S*
        # 0 = alpha*M* - lam*S* - mu*S* => alpha*M* = (lam + mu) * S*
        # S* = alpha*M* / (lam + mu)
        
        # If freq mixing (lam = beta * I* / N* = beta * I*):
        # S* = (gamma + mu) / beta  (S*/N* = 1/R0)
        # S_star = (p.gamma + p.mu) / p.beta
        
        # from dI/dt=0: I* = (lam*S*) / (gamma + mu)
        # from dS/dt=0: lam*S* = alpha*M* - mu*S*
        # I* = (alpha*M* - mu*S*) / (gamma + mu)
        
        # from dM/dt=0: M* = p.v * (1 - p.vacc_p) / (p.alpha + p.mu) (assuming N=1, v=mu)
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

    def check_conservation(self, y: np.ndarray) -> tuple[bool, str]:
        p = self.params
        if np.isclose(p.v, p.mu):
            N = y.sum()
            N_expected = p.v / p.mu
            is_conserved = np.allclose(N, N_expected, rtol=1e-3)
            msg = f"N = {N:.2f}, Expected N = {N_expected:.2f}"
            return is_conserved, msg
        else:
            return True, "N/A (v != mu)"
   
   
@dataclass(frozen=True)
class SEIRSDemographyCountsParams(SEIRDemographyCountsParams):
    """Parameters for an SEIRS model with births, deaths, and counts."""
    omega: float = 0.0  # Rate of waning immunity (R -> S)

    def __post_init__(self):
        super().__post_init__()
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

        # Solve steady-state equations (N=1, v=mu)
        # S* / N* = 1 / R0
        # S_star = 1.0 / R0 # This is only true if omega=0 !
        
        # Let's solve from scratch (N=1, v=mu)
        # dE/dt=0 => lam*S = (sigma+mu)E => E* = lam*S / (sigma+mu)
        # dI/dt=0 => sigma*E = (gamma+mu)I => I* = sigma*E / (gamma+mu)
        # Substitute E* into I*:
        # I* = sigma * (lam*S / (sigma+mu)) / (gamma+mu)
        # If lam = beta*I:
        # I* = sigma * (beta*I*S / (sigma+mu)) / (gamma+mu)
        # 1 = (sigma*beta*S) / ((sigma+mu)(gamma+mu))
        # S* = ((sigma+mu)(gamma+mu)) / (sigma*beta)
        # S* = 1 / R0_eff  (where R0_eff = (beta/(gamma+mu))*(sigma/(sigma+mu)))
        # This holds: S_star = 1.0 / R0 (as a fraction)
        
        S_star = 1.0 / R0

        # dR/dt=0 => mu*vacc_p + gamma*I = (mu+omega)*R
        # R* = (mu*vacc_p + gamma*I) / (mu+omega)
        
        # dI/dt=0 => I* = sigma*E / (gamma+mu)
        # dE/dt=0 => E* = (beta*I*S) / (sigma+mu)  (using lam=beta*I, N=1)
        # Substitute S_star:
        # E* = (beta*I*S_star) / (sigma+mu)
        
        # dS/dt=0 => mu*(1-vacc_p) - beta*I*S - mu*S + omega*R = 0
        # Substitute S*, E*, R*:
        # mu(1-vacc_p) - beta*I*(1/R0) - mu*(1/R0) + omega*((mu*vacc_p + gamma*I) / (mu+omega)) = 0
        
        # Group terms with I*:
        # I * [ -beta/R0 + omega*gamma/(mu+omega) ] = -mu(1-vacc_p) + mu/R0 - omega*mu*vacc_p/(mu+omega)
        # I * [ (omega*gamma - beta*(mu+omega)/R0) / (mu+omega) ] = mu * [ -1+vacc_p + 1/R0 - omega*vacc_p/(mu+omega) ]
        
        # This is getting complex. Let's use the simpler relations:
        # E* = (gamma+mu)/sigma * I*
        # R* = (mu*p.vacc_p + p.gamma*I*) / (p.mu + p.omega)
        # S* = 1 - E* - I* - R* (since N=1)
        
        # And S* must also equal 1/R0.
        # 1/R0 = 1 - E* - I* - R*
        # 1/R0 = 1 - ((gamma+mu)/sigma)*I - I - (mu*vacc_p + gamma*I) / (mu+omega)
        
        # Solve for I*:
        # I * [ (gamma+mu)/sigma + 1 + gamma/(mu+omega) ] = 1 - 1/R0 - (mu*vacc_p)/(mu+omega)
        
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
        
