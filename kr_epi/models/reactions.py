
    # kr_epi/models/reactions.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Tuple, Literal
import numpy as np

Array = np.ndarray
Mixing = Literal["frequency", "density"]

@dataclass(frozen=True)
class Reaction:
    name: str
    stoich: Dict[str, int]
    hazard: Callable[[float, Dict[str, float], Dict[str, float]], float]

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
            return np.array([max(0.0, r.hazard(t, state, self.params))
                             for r in self.reactions], dtype=float)
        return idx, S, hazards

# Helpers for incidence
    
def incidence(beta: float, X: float, Y: float, N: float, mixing: Mixing) -> float:
    if N <= 0 or X <= 0 or Y <= 0 or beta <= 0: return 0.0
    return beta * X * Y / N if mixing == "frequency" else beta * X * Y


def sir_demography_counts(beta: float, gamma: float, v: float, mu: float,
                          *, mixing: Mixing = "frequency", vacc_p: float = 0.0,
                          delta: float = 0.0) -> ReactionSystem:
    """
    States: X (susceptible), Y (infectious), Z (removed/immune).
    """
    labels = ("X", "Y", "Z")
    params = dict(beta=beta, gamma=gamma, v=v, mu=mu, mixing=mixing, p=vacc_p, delta=delta)

    def N(state: Dict[str, float]) -> float:
        return max(0.0, state["X"] + state["Y"] + state["Z"])

    rxns: List[Reaction] = []

    # Infection
    rxns.append(Reaction(
        "infection", {"X": -1, "Y": +1},
        lambda t, s, p: incidence(p["beta"], s["X"], s["Y"], N(s), p["mixing"])
    ))
    # Recovery
    rxns.append(Reaction(
        "recovery", {"Y": -1, "Z": +1},
        lambda t, s, p: p["gamma"] * s["Y"]
    ))
    # Births
    rxns.append(Reaction(
        "birth_to_X", {"X": +1},
        lambda t, s, p: p["v"] * (1.0 - p["p"]) * N(s)
    ))
    rxns.append(Reaction(
        "birth_to_Z", {"Z": +1},
        lambda t, s, p: p["v"] * p["p"] * N(s)
    ))
    # Natural deaths
    rxns.append(Reaction("death_X", {"X": -1}, lambda t, s, p: p["mu"] * s["X"]))
    rxns.append(Reaction("death_Y", {"Y": -1}, lambda t, s, p: p["mu"] * s["Y"]))
    rxns.append(Reaction("death_Z", {"Z": -1}, lambda t, s, p: p["mu"] * s["Z"]))
    # Infection-induced mortality (optional)
    if delta > 0:
        rxns.append(Reaction("inf_mort_Y", {"Y": -1}, lambda t, s, p: p["delta"] * s["Y"]))

    return ReactionSystem(labels, tuple(rxns), params, mixing)
