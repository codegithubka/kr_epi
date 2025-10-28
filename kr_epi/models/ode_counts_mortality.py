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
    """
    def __init__(self, beta: float, gamma: float, v: float, mu: float, delta: float,
                 mixing: Mixing = "frequency", vacc_p: float = 0.0):
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

    # Basic-threshold R0 at the DFE (X≈N0, Y≈0)
    def R0(self, N0: float) -> float:
        p = self.params
        denom = p.gamma + p.mu + p.delta
        if p.mixing == "frequency":
            return p.beta / denom
        else:
            return p.beta * N0 / denom
