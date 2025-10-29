# kr_epi/models/reactions.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple, Literal, Optional, Sequence
import numpy as np

Array = np.ndarray
Mixing = Literal["frequency", "density"]

# --- Parameter Dataclasses (similar to ODE models) ---
@dataclass(frozen=True)
class ReactionParamsBase:
    """
    Base class for reaction parameters.
    Ensures subclasses will call validation.
    """
    def __post_init__(self):
        # This method will be called by any inheriting dataclasses
        pass

@dataclass(frozen=True)
class SIRDemographyCountsReactionParams(ReactionParamsBase):
    """
    Parameters for the SIR Demography Counts reaction system.
    
    Non-default fields are defined FIRST.
    Default fields are defined AFTER.
    """
    # --- NON-DEFAULT FIELDS ---
    beta: float
    gamma: float
    v: float  # Birth rate
    mu: float  # Natural death rate
    
    # --- DEFAULT FIELDS ---
    mixing: Mixing = "frequency"
    beta_fn: Optional[Callable[[float], float]] = None
    vacc_p: float = 0.0 # Vaccination at birth coverage
    delta: float = 0.0 # Infection-induced mortality rate

    def __post_init__(self):
        """Validate parameters after initialization."""
        # super().__post_init__() # Not needed if base post_init is empty

        if self.mixing not in ("frequency", "density"):
            raise ValueError(f"mixing must be 'frequency' or 'density', got '{self.mixing}'")
        if self.beta < 0: # Allow beta=0 for forcing via beta_fn
            raise ValueError(f"beta must be non-negative, got {self.beta}")
        if self.gamma <= 0:
            raise ValueError(f"gamma must be positive, got {self.gamma}")
        if self.v < 0:
            raise ValueError(f"v (birth rate) must be non-negative, got {self.v}")
        if self.mu <= 0:
            raise ValueError(f"mu (natural death rate) must be positive, got {self.mu}")
        if not 0 <= self.vacc_p < 1:
            raise ValueError(f"vacc_p must be in [0, 1), got {self.vacc_p}")
        if self.delta < 0:
            raise ValueError(f"delta (disease death rate) must be non-negative, got {self.delta}")

# --- Reaction Definition (Remains the same) ---
@dataclass(frozen=True)
class Reaction:
    """A single reaction with hazard (propensity) and state change vector."""
    name: str
    stoich: Dict[str, int]
    hazard: Callable[[float, Dict[str, float], ReactionParamsBase], float]
    # hazard(t, state_as_dict, params_dataclass) -> rate

# --- Reaction System ---
@dataclass
class ReactionSystem:
    labels: Sequence[str]
    reactions: Sequence[Reaction]
    params: ReactionParamsBase # Use the base dataclass or specific ones
    _idx: Dict[str, int] = field(init=False, repr=False)
    _S: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        # Precompute index map and stoichiometry matrix
        self._idx = {k: i for i, k in enumerate(self.labels)}
        self._S = np.zeros((len(self.labels), len(self.reactions)), dtype=int)
        for j, r in enumerate(self.reactions):
            for k, v in r.stoich.items():
                if k not in self._idx:
                    raise ValueError(f"State '{k}' in reaction '{r.name}' stoichiometry not found in labels {self.labels}")
                self._S[self._idx[k], j] = v

    def get_stoichiometry_matrix(self) -> np.ndarray:
        return self._S.copy() # Return a copy

    def calculate_hazards(self, t: float, y: np.ndarray) -> np.ndarray:
        """Calculates all reaction hazards given the current state vector y."""
        if len(y) != len(self.labels):
            raise ValueError(f"State vector y has length {len(y)}, expected {len(self.labels)}")
        # Convert state vector y to dictionary for hazard functions
        state_dict = {label: float(y[i]) for i, label in enumerate(self.labels)}
        
        hazards = np.array(
            [max(0.0, r.hazard(t, state_dict, self.params)) for r in self.reactions],
            dtype=float
        )
        return hazards

# ---------- Incidence Helper (Remains the same) ----------
def _incidence(beta: float, X: float, Y: float, N: float, mixing: Mixing) -> float:
    if N <= 0 or X <= 0 or Y <= 0 or beta <= 0:
        return 0.0
    return beta * X * Y / N if mixing == "frequency" else beta * X * Y

# ---------- Factory Function Example (SIR with Demography/Mortality) ----------
def sir_demography_counts_reactions(
    beta: float, gamma: float, v: float, mu: float, *,
    mixing: Mixing = "frequency", vacc_p: float = 0.0, delta: float = 0.0,
    beta_fn: Optional[Callable[[float], float]] = None
) -> ReactionSystem:
    """
    Creates a ReactionSystem for an SIR model with births, deaths,
    optional vaccination at birth, and optional disease-induced mortality.

    States: X (Susceptible), Y (Infectious), Z (Recovered/Immune).
    """
    labels = ("X", "Y", "Z")
    params = SIRDemographyCountsReactionParams(
        beta=beta, gamma=gamma, v=v, mu=mu, mixing=mixing,
        vacc_p=vacc_p, delta=delta, beta_fn=beta_fn
    )

    def N(s: Dict[str, float]) -> float:
        # Helper to calculate total population from state dict
        return max(0.0, s["X"] + s["Y"] + s["Z"])

    def get_beta(t: float, p: SIRDemographyCountsReactionParams) -> float:
        # Helper to get the correct beta value at time t
        return p.beta_fn(t) if p.beta_fn is not None else p.beta

    reactions: List[Reaction] = [
        Reaction(
            name="infection",
            stoich={"X": -1, "Y": +1},
            # Hazard needs params cast to the specific dataclass type if needed inside
            hazard=lambda t, s, p: _incidence(get_beta(t, p), s["X"], s["Y"], N(s), p.mixing) # type: ignore
        ),
        Reaction(
            name="recovery",
            stoich={"Y": -1, "Z": +1},
            hazard=lambda t, s, p: p.gamma * s["Y"] # type: ignore
        ),
        Reaction(
            name="birth_to_X",
            stoich={"X": +1},
            hazard=lambda t, s, p: p.v * (1.0 - p.vacc_p) * N(s) # type: ignore
        ),
        Reaction(
            name="birth_to_Z",
            stoich={"Z": +1},
            hazard=lambda t, s, p: p.v * p.vacc_p * N(s) # type: ignore
        ),
        Reaction(
            name="death_X",
            stoich={"X": -1},
            hazard=lambda t, s, p: p.mu * s["X"] # type: ignore
        ),
        Reaction(
            name="death_Y",
            stoich={"Y": -1},
            hazard=lambda t, s, p: p.mu * s["Y"] # type: ignore
        ),
        Reaction(
            name="death_Z",
            stoich={"Z": -1},
            hazard=lambda t, s, p: p.mu * s["Z"] # type: ignore
        ),
    ]

    # Add infection-induced mortality if delta > 0
    if params.delta > 0:
        reactions.append(
            Reaction(
                name="infection_death_Y",
                stoich={"Y": -1},
                hazard=lambda t, s, p: p.delta * s["Y"] # type: ignore
            )
        )

    return ReactionSystem(labels=labels, reactions=reactions, params=params)

# --- Add similar factory functions for SIS, SEIR etc. as needed ---
# Example placeholder:
# def sis_demography_counts_reactions(...) -> ReactionSystem: ...
# def seir_demography_counts_reactions(...) -> ReactionSystem: ...