# kr_epi/stochastic/direct.py (COMPLETELY REWRITE)
"""
Gillespie's Direct Method (SSA) for stochastic simulation.

Implementation of the Direct Method algorithm (Gillespie 1977) for
exact stochastic simulation of reaction systems.
"""
from typing import Dict, Optional, List, Tuple
import numpy as np
from kr_epi.models.reactions import ReactionSystem  # IMPORT, don't duplicate!

Array = np.ndarray


class Direct:
    """
    Gillespie's Direct Method for exact stochastic simulation.
    
    Implements the Stochastic Simulation Algorithm (SSA) from:
    Gillespie, D. T. (1977). Exact stochastic simulation of coupled
    chemical reactions. The Journal of Physical Chemistry, 81(25), 2340-2361.
    
    Parameters
    ----------
    system : ReactionSystem
        The reaction system to simulate
    seed : int, optional
        Random seed for reproducibility
        
    Examples
    --------
    >>> from kr_epi.models.reactions import sir_demography_counts_reactions
    >>> system = sir_demography_counts_reactions(beta=0.5, gamma=0.1, v=0.02, mu=0.02)
    >>> engine = Direct(system, seed=42)
    >>> t, y = engine.run(x0={'X': 990, 'Y': 10, 'Z': 0}, t_max=100)
    
    References
    ----------
    Keeling & Rohani (2008), Section 6.2
    """
    
    def __init__(self, system: ReactionSystem, seed: Optional[int] = None):
        self.system = system
        self.rng = np.random.default_rng(seed)
    
    def run(
        self,
        x0: Dict[str, float],
        t_max: float,
        *,
        record_times: Optional[Array] = None
    ) -> Tuple[Array, Array]:
        """
        Run Gillespie's Direct Method simulation.
        
        Parameters
        ----------
        x0 : dict
            Initial state as {state_name: count}
        t_max : float
            Maximum simulation time
        record_times : array_like, optional
            Specific times to record state. If None, records all events.
            
        Returns
        -------
        times : ndarray
            Array of time points
        states : ndarray
            State matrix (n_states × n_times)
            
        Notes
        -----
        Algorithm:
        1. Calculate hazards (propensities) for all reactions
        2. Draw time to next event from exponential distribution
        3. Select which reaction fires using weighted random choice
        4. Update state according to reaction stoichiometry
        5. Repeat until t_max reached
        """
        # Convert initial state dict to array
        labels = list(self.system.labels)
        x = np.array([float(x0.get(k, 0.0)) for k in labels], dtype=float)
        x = np.maximum(0.0, np.round(x))  # Ensure non-negative integers
        
        # Storage for trajectory
        T_list: List[float] = [0.0]
        X_list: List[Array] = [x.copy()]
        
        t = 0.0
        
        # Get stoichiometry matrix S (n_states × n_reactions)
        S = self.system.get_stoichiometry_matrix()
        
        while t < t_max:
            # Step 1: Calculate all reaction hazards
            hazards = self.system.calculate_hazards(t, x)
            a0 = hazards.sum()
            
            # If no reactions can fire, we're done
            if a0 <= 1e-12:
                if T_list[-1] < t_max:
                    T_list.append(t_max)
                    X_list.append(x.copy())
                break
            
            # Step 2: Draw time to next reaction
            tau = self.rng.exponential(scale=1.0 / a0)
            t_next = t + tau
            
            # If we've exceeded t_max, stop
            if t_next >= t_max:
                if T_list[-1] < t_max:
                    T_list.append(t_max)
                    X_list.append(x.copy())
                break
            
            # Step 3: Select which reaction fires
            # Use cumulative sum method for numerical stability
            cumsum = np.cumsum(hazards)
            u = self.rng.uniform(0, a0)
            j = np.searchsorted(cumsum, u)  # Index of reaction that fires
            
            # Step 4: Update state
            x = x + S[:, j]
            x = np.maximum(0.0, x)  # Clamp negative values to zero
            
            # Step 5: Record event
            t = t_next
            T_list.append(t)
            X_list.append(x.copy())
        
        # Convert lists to arrays
        t_arr = np.array(T_list, dtype=float)
        X_arr = np.vstack(X_list).T  # shape: (n_states, n_times)
        
        # Interpolate to record_times if requested
        if record_times is not None:
            rt = np.asarray(record_times, dtype=float)
            # Use right-sided search: value at time t is state just before t
            indices = np.searchsorted(t_arr, rt, side='right') - 1
            indices = np.clip(indices, 0, len(T_list) - 1)
            X_interp = X_arr[:, indices]
            return rt, X_interp
        
        return t_arr, X_arr