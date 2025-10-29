# kr_epi/stochastic/tau_leap.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from kr_epi.models.reactions import ReactionSystem
# from kr_epi.models.reactions import ReactionSystem # May need import

Array = np.ndarray

@dataclass
class TauLeap:
    """Tau-Leaping approximate stochastic simulation for a ReactionSystem."""
    system: 'ReactionSystem' # Use quotes for forward reference if needed
    seed: Optional[int] = None
    adapt: bool = True       # Use adaptive tau based on state?
    eps: float = 0.03        # Epsilon for adaptive tau step (relative change tolerance)
    max_dt: float = 1.0      # Maximum tau allowed
    min_dt: float = 1e-6     # Minimum tau allowed (to prevent tiny steps)
    safety: float = 0.9      # Safety factor for adaptive tau
    critical_threshold: int = 10 # Reactions involving species < threshold use SSA step
    max_ssa_steps: int = 100 # Max SSA steps per tau-leap interval if needed

    def run(self,
            x0: Dict[str, float],
            t_max: float,
            *,
            dt: float = 0.1,         # Initial/Fixed tau if adapt=False
            record_times: Optional[np.ndarray] = None
            ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Runs the Tau-Leaping simulation.

        Parameters are similar to Direct.run, with 'dt' used as the
        initial or fixed time step.
        """
        rng = np.random.default_rng(self.seed)
        labels = self.system.labels
        # Use precomputed index map and stoichiometry matrix
        idx = self.system._idx
        S = self.system.get_stoichiometry_matrix() # <-- CHANGE HERE

        x = np.array([float(x0.get(k, 0.0)) for k in labels], dtype=float)
        x = np.maximum(0.0, np.floor(x + 1e-12)) # Ensure non-negative integer counts

        T_list: List[float] = [0.0]
        X_list: List[np.ndarray] = [x.copy()]
        t = 0.0

        while t < t_max:
            # Calculate current hazards
            a = self.system.calculate_hazards(t, x) # <-- CHANGE HERE
            a = np.maximum(a, 0.0) # Ensure non-negative hazards
            a_sum = a.sum()

            if a_sum <= 1e-10:
                # No reactions possible, jump to end
                if T_list[-1] < t_max:
                    T_list.append(t_max)
                    X_list.append(x.copy())
                break

            # Determine timestep tau
            if self.adapt:
                tau = self._suggest_tau(x, a, S)
            else:
                tau = dt
            tau = max(self.min_dt, min(tau, self.max_dt, t_max - t))

            # --- Critical Reaction Check (Gillespie et al., 2001 adaptation) ---
            # Identify reactions that could deplete a reactant below zero
            critical_reactions = self._get_critical_reactions(x, a, S, tau)

            if critical_reactions.any():
                 # Use one or more SSA steps for critical reactions
                 # Here, we'll take a simplified approach: take *one* SSA step if any reaction is critical
                 # A more advanced method would take SSA steps until no reaction is critical for a small tau_ssa.

                 # --- Perform one SSA Step ---
                 a_ssa = a # Use current hazards
                 a0_ssa = a_ssa.sum()
                 if a0_ssa <= 1e-10: # Double check in case states changed subtly
                     if T_list[-1] < t_max:
                        T_list.append(t_max)
                        X_list.append(x.copy())
                     break

                 tau_ssa = rng.exponential(1.0 / a0_ssa)

                 # If SSA step is larger than tau-leap step, just do tau-leap
                 if tau_ssa >= tau:
                      num_events = rng.poisson(a * tau)
                      x_change = S @ num_events
                      x_new = x + x_change
                      t = t + tau
                 else:
                     # Take the SSA step
                     probabilities_ssa = a_ssa / a0_ssa
                     probabilities_ssa /= probabilities_ssa.sum() # Normalize
                     j_ssa = rng.choice(len(a_ssa), p=probabilities_ssa)

                     x_new = x + S[:, j_ssa]
                     t = t + tau_ssa # Advance time by SSA step

                     # If SSA step exceeds t_max, handle it
                     if t >= t_max:
                          t = t_max
                          # State x remains as it was *before* this failed step
                          x_new = x.copy()
                          if T_list[-1] < t_max:
                              T_list.append(t)
                              X_list.append(x_new)
                          break # Exit loop


            else:
                 # --- Perform Tau-Leap Step ---
                 # Poisson draws for number of times each reaction occurs in [t, t+tau)
                 num_events = rng.poisson(a * tau)
                 x_change = S @ num_events # Calculate net change in state vector
                 x_new = x + x_change
                 t = t + tau # Advance time

            # Check for and handle negative populations (clamp to zero)
            if np.any(x_new < -1e-9):
                 # This indicates tau might be too large, but for simplicity here, we clamp.
                 # A more sophisticated method might reduce tau and retry.
                 print(f"Warning: Negative state encountered at t={t:.2f}. Clamping to zero.")
                 x_new = np.maximum(0.0, x_new)

            x = x_new
            T_list.append(t)
            X_list.append(x.copy())


        t_arr = np.array(T_list, dtype=float)
        X_arr = np.vstack(X_list).T # shape (n_states, n_times)

        # Interpolate if needed
        if record_times is not None:
            rt = np.asarray(record_times, dtype=float)
            indices = np.searchsorted(t_arr, rt, side="right") - 1
            indices = np.clip(indices, 0, len(T_list) - 1)
            X_interp = X_arr[:, indices]
            return rt, X_interp

        return t_arr, X_arr

    def _suggest_tau(self, x: np.ndarray, a: np.ndarray, S: np.ndarray) -> float:
        """Suggests tau based on Gillespie's 2001 condition."""
        small = 1e-12 # Prevent division by zero

        # Estimate mean and variance of change for each species i
        mu = S @ a      # E[dX_i/dt]
        sigma_sq = S**2 @ a # Approx Var[dX_i/dt] (only works if reactions are independent, approximation)

        # Condition: |mu_i * tau| << x_i  AND  sigma_sq_i * tau^2 << x_i
        # Simplified using epsilon:
        # tau <= eps * x_i / |mu_i|
        # tau <= eps^2 * x_i / sigma_sq_i
        # We need tau >