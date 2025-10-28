# kr_epi/models/reactions.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple, Literal, Optional
import numpy as np

Array = np.ndarray
Mixing = Literal["frequency", "density"]

@dataclass(frozen=True)
class Reaction:
    """A single reaction with hazard (propensity) and state change vector."""
    name: str
    stoich: Dict[str, int]            # e.g. {"X": -1, "Y": +1}
    hazard: Callable[[float, Dict[str, float], Dict[str, float]], float]
    # hazard(t, state_as_dict, params) -> rate

class ReactionSystem:
    def __init__(self,
                 labels: Tuple[str, ...],
                 reactions: Tuple[Reaction, ...],
                 params: Dict[str, float],
                 mixing: Mixing = "frequency"):
        self.labels = labels
        self.reactions = reactions
        self.params = params
        self.mixing = mixing

    def as_vectorized(self):
        idx = {k: i for i, k in enumerate(self.labels)}
        S = np.zeros((len(self.labels), len(self.reactions)), dtype=int)
        for j, r in enumerate(self.reactions):
            for k, v in r.stoich.items():
                S[idx[k], j] = v

        def hazards(t: float, y: Array) -> Array:
            state = {k: float(y[idx[k]]) for k in self.labels}
            return np.array([max(0.0, r.hazard(t, state, self.params)) for r in self.reactions], dtype=float)

        return idx, S, hazards

# ---------- helpers ----------
def incidence(beta: float, X: float, Y: float, N: float, mixing: Mixing) -> float:
    if N <= 0 or X <= 0 or Y <= 0 or beta <= 0: 
        return 0.0
    return beta * X * Y / N if mixing == "frequency" else beta * X * Y

# ---------- factory with beta_fn ----------
def sir_demography_counts(
    beta: float, gamma: float, v: float, mu: float, *,
    mixing: Mixing = "frequency", vacc_p: float = 0.0,
    delta: float = 0.0, beta_fn: Optional[Callable[[float], float]] = None
) -> ReactionSystem:
    """
    States: X, Y, Z. Reactions:
      infection, recovery, births to X/Z, natural deaths X/Y/Z,
      (optional) infection-induced death in Y with rate delta*Y.
    If beta_fn is provided, infection hazard uses beta_fn(t); otherwise constant beta.
    """
    labels = ("X", "Y", "Z")
    params = dict(beta=beta, gamma=gamma, v=v, mu=mu, mixing=mixing, p=vacc_p, delta=delta)

    def N(s: Dict[str, float]) -> float:
        return max(0.0, s["X"] + s["Y"] + s["Z"])

    def beta_t(t: float, p: Dict[str, float]) -> float:
        return beta_fn(t) if beta_fn is not None else p["beta"]

    rxns: List[Reaction] = []

    rxns.append(Reaction(
        "infection", {"X": -1, "Y": +1},
        lambda t, s, p: incidence(beta_t(t, p), s["X"], s["Y"], N(s), p["mixing"])
    ))
    rxns.append(Reaction("recovery", {"Y": -1, "Z": +1}, lambda t, s, p: p["gamma"] * s["Y"]))
    rxns.append(Reaction("birth_to_X", {"X": +1}, lambda t, s, p: p["v"] * (1.0 - p["p"]) * N(s)))
    rxns.append(Reaction("birth_to_Z", {"Z": +1}, lambda t, s, p: p["v"] * p["p"] * N(s)))
    rxns.append(Reaction("death_X", {"X": -1}, lambda t, s, p: p["mu"] * s["X"]))
    rxns.append(Reaction("death_Y", {"Y": -1}, lambda t, s, p: p["mu"] * s["Y"]))
    rxns.append(Reaction("death_Z", {"Z": -1}, lambda t, s, p: p["mu"] * s["Z"]))
    if delta > 0:
        rxns.append(Reaction("inf_mort_Y", {"Y": -1}, lambda t, s, p: p["delta"] * s["Y"]))

    # Use positional args to be robust against older cached __init__ signatures
    return ReactionSystem(labels, tuple(rxns), params, mixing)
