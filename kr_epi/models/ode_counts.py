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
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0):
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
    
    # Useful R0 calculators at DFE (X*≈N, Y*=0, Z*=0)
    def R0(self, N0: float) -> float:
        p = self.params
        if p.mixing == "frequency":
            return p.beta / (p.gamma + p.mu)
        else:  # density: note the dependence on N
            return p.beta * N0 / (p.gamma + p.mu)
        

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

    def R0(self, N0: float) -> float:
        p = self.params
        return (p.beta / (p.gamma + p.mu)) if p.mixing == "frequency" else (p.beta * N0 / (p.gamma + p.mu))

    
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

    def R0(self, N0: float) -> float:
        p = self.params
        # R0 = (beta * X*/N*) * (1/(mu+sigma)) * (sigma/(mu+gamma)) for frequency;
        # for density, replace beta*X*/N* with beta*X* = beta*N0
        if p.mixing == "frequency":
            return (p.beta) * (1.0 / (p.mu + p.sigma)) * (p.sigma / (p.mu + p.gamma))
        else:
            return (p.beta * N0) * (1.0 / (p.mu + p.sigma)) * (p.sigma / (p.mu + p.gamma))