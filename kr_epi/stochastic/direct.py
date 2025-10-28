from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List
import numpy as np
from kr_epi.models.reactions import ReactionSystem

Array = np.ndarray


@dataclass
class Direct:
    """Gillespie Direct (SSA) for a ReactionSystem (counts)."""
    system: ReactionSystem
    seed: Optional[int] = None

    def run(self,
            x0: Dict[str, float],
            t_max: float,
            *,
            record_times: Optional[Array] = None, # type: ignore
            max_events: int = 2_000_000) -> Tuple[Array, Array]: # type: ignore
        rng = np.random.default_rng(self.seed)
        labels = self.system.labels
        idx, S, hazards = self.system.as_vectorized()

        # state vector
        x = np.array([float(x0.get(k, 0.0)) for k in labels], dtype=float)
        # require non-negative integers (counts)
        x = np.maximum(0.0, np.floor(x + 1e-12))

        t = 0.0
        T: List[float] = [t]
        Xs: List[Array] = [x.copy()]

        # If record_times provided, we’ll do linear interpolation onto them at the end.
        n_events = 0
        while t < t_max and n_events < max_events:
            a = hazards(t, x)
            a0 = a.sum()
            if a0 <= 0.0:
                # No possible reaction; jump to t_max and record final state
                T.append(t_max)
                Xs.append(x.copy())
                break

            # Exponential waiting time
            tau = rng.exponential(1.0 / a0)
            t_next = t + tau
            if t_next > t_max:
                # Stop exactly at t_max with current state
                T.append(t_max)
                Xs.append(x.copy())
                break

            # Pick reaction j with prob a_j/a0 via inverse CDF
            r = rng.random() * a0
            cum = np.cumsum(a)
            j = int(np.searchsorted(cum, r, side="right"))

            # Apply stoichiometry: x := x + S[:, j], but guard against negative
            proposal = x + S[:, j]
            if np.any(proposal < 0):
                # if impossible (e.g., recovery with Y=0 due to numerical rounding), skip
                # but this is rare since hazards already 0 in those states
                T.append(t_next)
                Xs.append(x.copy())
                t = t_next
                n_events += 1
                continue

            x = proposal
            t = t_next
            T.append(t)
            Xs.append(x.copy())
            n_events += 1

        T = np.asarray(T, dtype=float)
        X = np.vstack(Xs).T  # shape (n_states, n_timepoints)

        # Interpolate onto record_times if requested (piecewise-constant, left-hold)
        if record_times is not None:
            rt = np.asarray(record_times, dtype=float)
            # indices of last T <= rt
            ii = np.searchsorted(T, rt, side="right") - 1
            ii = np.clip(ii, 0, len(T) - 1)
            X_interp = X[:, ii]
            return rt, X_interp

        return T, X
