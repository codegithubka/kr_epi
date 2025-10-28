from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from kr_epi.stochastic.direct import ReactionSystem

Array = np.ndarray

@dataclass
class TauLeap:
    system: ReactionSystem
    seed: Optional[int] = None
    adapt: bool = True
    eps: float = 0.03
    max_dt: float = 1.0
    min_dt: float = 1e-4
    safety: float = 0.9
    backtrack: int = 8
    clamp_negatives: bool = True
    
    def run(self,
            x0: Dict[str, float],
            t_max: float,
            *,
            dt: float = 0.2,
            record_times: Optional[Array] = None) -> Tuple[Array, Array]: # type: ignore
        rng = np.random.default_rng(self.seed)
        labels = self.system.labels
        idx, S, hazards = self.system.as_vectorized()

        x = np.array([float(x0.get(k, 0.0)) for k in labels], dtype=float)
        # integerize and non-negative
        x = np.maximum(0.0, np.floor(x + 1e-12))

        # Internal event times (adaptive) or fixed grid
        T: List[float] = [0.0]
        Xs: List[Array] = [x.copy()] # type: ignore
        t = 0.0

        while t < t_max:
            a = hazards(t, x)             # propensities a_j >= 0
            a = np.clip(a, 0.0, np.inf)
            a_sum = a.sum()
            if a_sum <= 0.0:
                # no more reactions possible: jump to t_max
                T.append(t_max)
                Xs.append(x.copy())
                break

            # choose tau
            tau = dt if not self.adapt else self._suggest_tau(x, a, S)
            tau = min(tau, self.max_dt, t_max - t)
            if tau < self.min_dt:
                tau = min(self.min_dt, t_max - t)

            # attempt leap; backtrack on negatives
            ok = False
            tau_try = float(tau)
            for _ in range(max(1, self.backtrack)):
                # Poisson draws for each reaction channel
                k = rng.poisson(a * tau_try)
                proposal = x + S @ k

                if (proposal >= -1e-9).all():
                    ok = True
                    x = np.maximum(0.0, proposal) if self.clamp_negatives else proposal
                    t += tau_try
                    T.append(t)
                    Xs.append(x.copy())
                    break

                # backtrack: shrink step
                tau_try *= 0.5
                if tau_try < self.min_dt:
                    # final guard: clamp (rare)
                    if self.clamp_negatives:
                        x = np.maximum(0.0, proposal)
                        t += tau_try
                        T.append(t)
                        Xs.append(x.copy())
                        ok = True
                    break

            if not ok:
                # If still not ok (should be extremely rare), stop to avoid loops
                T.append(t)
                Xs.append(x.copy())
                break

        T = np.asarray(T, dtype=float)
        X = np.vstack(Xs).T  # (n_states, n_times)

        if record_times is not None:
            rt = np.asarray(record_times, dtype=float)
            # left-hold interpolation
            ii = np.searchsorted(T, rt, side="right") - 1
            ii = np.clip(ii, 0, len(T) - 1)
            return rt, X[:, ii]

        return T, X

    
    def _suggest_tau(self, x: Array, a: Array, S: Array) -> float: # type: ignore
        """
        Anderson-style heuristic:
          Choose tau so that for every species i,
            |E[ΔX_i]| = |sum_j nu_ij * a_j| * tau  <= eps * max(1, x_i)
          => tau_i = eps * max(1, x_i) / (|sum_j nu_ij * a_j| + tiny)
        """
        small = 1e-12
        drift = S @ a  # expected change per unit time for each species
        scale = np.maximum(1.0, x)  # protects small populations
        denom = np.abs(drift) + small
        tau_vec = self.eps * scale / denom
        tau = float(np.min(tau_vec))
        # safety factor to be conservative
        tau *= self.safety
        # cap by max_dt, lower bound handled in caller
        return max(self.min_dt, min(tau, self.max_dt))