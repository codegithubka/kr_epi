from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Callable, Sequence
import numpy as np
from kr_epi.models.base import ODEBase


Mixing = Literal["frequency", "density"]

def _incidence(beta: float, X: float, Y: float, N: float, mixing: Mixing) -> float:
    if N <= 0:
        return 0.0
    return beta * X * Y / N if mixing == "frequency" else beta * X * Y

def delta_from_cfr(rho: float, gamma: float, mu: float) -> float:
    """
    For competing hazards with rates (gamma, mu, delta):
    CFR rho = delta / (gamma + mu + delta)  =>  delta = rho*(gamma+mu)/(1 - rho)
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError("rho must be in [0,1).")
    return rho * (gamma + mu) / (1.0 - rho)


@dataclass(frozen=True)
class SIRDemographyCountsMortalityParams:
    beta: float      # transmission parameter
    gamma: float     # recovery rate
    v: float         # per-capita birth rate
    mu: float        # per-capita natural death (all classes)
    delta: float     # infection-induced extra death rate for Y
    mixing: Mixing = "frequency"
    vacc_p: float = 0.0  # vaccination at birth (fraction of newborns to Z)



class SIRDemographyCountsMortality(ODEBase):
    """
    Counts model (X,Y,Z) with births vN, natural deaths mu, and extra death delta in Y.

    dX = v(1-p)N - inc                  - mu X
    dY = inc      - (gamma + mu + delta) Y
    dZ = v p N    + gamma Y             - mu Z

    N = X + Y + Z, and   dN/dt = (v - mu)N - delta Y   (extra deaths lower N)
    inc = β X Y / N (frequency) or β X Y (density).
    
    From K&R Section 2.2.1: Disease-induced mortality changes equilibrium.
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float, delta: float,
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
        if delta < 0:
            raise ValueError("delta (disease-induced death rate) must be non-negative")
        if not 0 <= vacc_p < 1:
            raise ValueError("vacc_p must be in [0, 1)")
            
        self.params = SIRDemographyCountsMortalityParams(
            beta=beta, gamma=gamma, v=v, mu=mu, delta=delta, mixing=mixing, vacc_p=vacc_p
        )

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
        dY = inc - (p.gamma + p.mu + p.delta) * Y
        dZ = births_R + p.gamma * Y - p.mu * Z
        return np.array([dX, dY, dZ], dtype=float)

    def R0(self, N0: float = None) -> float:
        """
        Calculate basic reproduction number.
        
        R0 = beta / (gamma + mu + delta) for frequency-dependent
        R0 = beta * N0 / (gamma + mu + delta) for density-dependent
        
        Note: Disease-induced mortality (delta) reduces R0.
        """
        p = self.params
        denom = p.gamma + p.mu + p.delta
        if p.mixing == "frequency":
            return p.beta / denom
        else:
            if N0 is None:
                raise ValueError("N0 required for density-dependent R0 calculation")
            return p.beta * N0 / denom
    
    def endemic_equilibrium(self, N0: float = None) -> Dict[str, float]:
        """
        Calculate endemic equilibrium with disease-induced mortality.
        
        From K&R Section 2.2.1, equation (2.35):
        For frequency-dependent transmission:
            S* = 1/R0
            I* = (mu/beta(1-rho)) * (R0 - 1)  where rho is CFR
            N* = (v/mu) * (R0(1-rho)/(R0 - rho))
            
        Population size is reduced by disease-induced mortality.
        """
        p = self.params
        
        # Calculate R0
        if p.mixing == "frequency":
            R0_val = self.R0()
        else:
            if N0 is None:
                N0 = p.v / p.mu if p.v > 0 and p.mu > 0 else 1.0
            R0_val = self.R0(N0)
        
        # No endemic equilibrium if R0 <= 1
        if R0_val <= 1:
            return {'X': None, 'Y': None, 'Z': None, 'N': None, 'R0': R0_val}
        
        # Calculate CFR (case fatality ratio)
        # CFR = delta / (gamma + mu + delta)
        cfr = p.delta / (p.gamma + p.mu + p.delta)
        
        # Endemic equilibrium for frequency-dependent (K&R eq 2.35)
        if p.mixing == "frequency":
            # N* is reduced by mortality
            N_star = (p.v / p.mu) * (R0_val * (1 - cfr) / (R0_val - cfr))
            # S* / N* = 1/R0
            S_star = 1.0 / R0_val
            X_star = S_star * N_star
            # I* / N* = (mu / (beta * (1 - rho))) * (R0 - 1) / N*
            I_star_frac = (p.mu / (p.beta * (1 - cfr))) * (R0_val - 1) / N_star
            Y_star = I_star_frac * N_star
        else:
            # Density-dependent equilibrium
            X_star = (p.gamma + p.mu + p.delta) / p.beta
            Y_star = (p.v - p.mu * X_star) / (p.gamma + p.mu + p.delta)
            N_star = X_star + Y_star / (1 - cfr)  # Approximate
        
        Z_star = N_star - X_star - Y_star
        
        return {
            'X': X_star,
            'Y': Y_star,
            'Z': Z_star,
            'N': N_star,
            'R0': R0_val,
            'CFR': cfr
        }
    
    def check_conservation(self, y: np.ndarray) -> float:
        """
        Check conservation: dN/dt should equal (v - mu)*N - delta*Y.
        
        Note: Population declines due to disease-induced mortality (delta*Y).
        """
        X, Y, Z = y
        N = X + Y + Z
        
        # Calculate actual dN/dt
        dy = self.rhs(0, y)
        dN_actual = dy.sum()
        
        # Expected dN/dt = (v - mu) * N - delta * Y
        dN_expected = (self.params.v - self.params.mu) * N - self.params.delta * Y
        
        return abs(dN_actual - dN_expected)
    
    @staticmethod
    def cfr_to_delta(cfr: float, gamma: float, mu: float) -> float:
        """
        Convert case fatality ratio to disease-induced death rate.
        
        CFR = delta / (gamma + mu + delta)
        => delta = CFR * (gamma + mu) / (1 - CFR)
        
        Parameters
        ----------
        cfr : float
            Case fatality ratio in [0, 1)
        gamma : float
            Recovery rate
        mu : float
            Natural death rate
            
        Returns
        -------
        float
            Disease-induced death rate delta
        """
        return delta_from_cfr(cfr, gamma, mu)