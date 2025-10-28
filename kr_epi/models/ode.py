from dataclasses import dataclass
from typing import Callable, Optional, Literal
import numpy as np
from .base import ODEBase

Mixing = Literal["frequnecy", "density"]

def _foi_frac(beta: float, I:float, mixing:Mixing)->float:
    return beta * I


# SI ------------------------------------------------

@dataclass(frozen=True)

class SIParams:
    beta: float
    mixing: Mixing = "frequnecy"
    
class SI(ODEBase):
    def __init__(self, beta: float, mixing: Mixing = "frequnecy"):
        self.params = SIParams(beta=beta, mixing=mixing)
        
    def state_labels(self):
        return ("S", "I")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I = y
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi_frac(beta, I, p.mixing)
        dSdt = -lam * S
        dIdt = lam * S
        return np.array([dSdt, dIdt])
    
# SIS ------------------------------------------------

@dataclass(frozen=True)
class SISParams:
    beta: float
    gamma: float
    mixing: Mixing = "frequency"
    
class SIS(ODEBase):
    def __init__(self, beta: float, gamma: float, mixing: Mixing = "frequency"):
        self.params = SISParams(beta=beta, gamma=gamma, mixing=mixing)

    def state_labels(self):
        return ("S", "I")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I = y
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi_frac(beta, I, p.mixing)
        dSdt = -lam * S + p.gamma * I
        dIdt = lam * S - p.gamma * I
        return np.array([dSdt, dIdt])
    
# SIR (closed) ------------------------------------------------
@dataclass(frozen=True)
class SIRParams:
    beta: float
    gamma: float
    mixing: Mixing = "frequency"
    
class SIR(ODEBase):
    def __init__(self, beta: float, gamma: float, mixing: Mixing = "frequency"):
        self.params = SIRParams(beta=beta, gamma=gamma, mixing=mixing)
        
    def state_labels(self):
        return ("S", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I, R = y
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi_frac(beta, I, p.mixing)
        dSdt = -lam * S
        dIdt = lam * S - p.gamma * I
        dRdt = p.gamma * I
        return np.array([dSdt, dIdt, dRdt])
    
    # SIRS (closed) ------------------------------------------------
@dataclass(frozen=True)
class SIRSParams:
    beta: float
    gamma: float
    omega: float
    mixing: Mixing = "frequency"
    
class SIRS(ODEBase):
    def __init__(self, beta: float, gamma: float, omega: float, mixing: Mixing = "frequency"):
        self.params = SIRSParams(beta=beta, gamma=gamma, omega=omega, mixing=mixing)
        
    def state_labels(self):
        return ("S", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, I, R = y
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi_frac(beta, I, p.mixing)
        dSdt = -lam * S + p.omega * R
        dIdt = lam * S - p.gamma * I
        dRdt = p.gamma * I - p.omega * R
        return np.array([dSdt, dIdt, dRdt])
    
# SEIR (closed) ------------------------------------------------
@dataclass(frozen=True)
class SEIRParams:
    beta: float
    gamma: float
    sigma: float
    mixing: Mixing = "frequency"
    
class SEIR(ODEBase):
    def __init__(self, beta: float, gamma: float, sigma: float, mixing: Mixing = "frequency"):
        self.params = SEIRParams(beta=beta, gamma=gamma, sigma=sigma, mixing=mixing)
        
    def state_labels(self):
        return ("S", "E", "I", "R")
    
    def rhs(self, t: float, y: np.ndarray, beta_fn: Optional[Callable[[float], float]] = None) -> np.ndarray:
        S, E, I, R = y
        p = self.params
        beta = beta_fn(t) if beta_fn is not None else p.beta
        lam = _foi_frac(beta, I, p.mixing)
        dSdt = -lam * S
        dEdt = lam * S - p.sigma * E
        dIdt = p.sigma * E - p.gamma * I
        dRdt = p.gamma * I
        return np.array([dSdt, dEdt, dIdt, dRdt])