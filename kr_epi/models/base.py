#/!/usr/bin/env python3


from typing import Callable, Dict, Optional, Sequence, Tuple, List
import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray

class ODEBase:
    @property
    def labels(self) -> Sequence[str]:
        raise NotImplementedError

    def rhs(self, t: float, y: Array, beta_fn: Optional[Callable[[float], float]] = None) -> Array: # type: ignore
        raise NotImplementedError

    def integrate(
        self,
        y0: Dict[str, float] | np.ndarray | List[float], # <-- Updated type hint
        t_span,
        t_eval=None,
        beta_fn: Optional[Callable[[float], float]] = None,
        stiff_fallback: bool = True,
        guard_negatives: bool = True,
        **solve_kw
    ):
        """
        Integrate the ODE system.

        Args:
            y0 (Dict[str, float] | np.ndarray | List[float]):
                Initial conditions. Can be a dictionary mapping state labels
                to values (e.g., {'S': 0.99, 'I': 0.01}) or a
                numpy array/list with values in the correct order
                as defined by model.labels.
        ... (rest of docstring) ...
        """
        labels = list(self.labels)
        
        # --- NEW LOGIC TO HANDLE y0 ---
        if isinstance(y0, dict):
            # Convert from dict to array based on labels
            y0v = np.array([y0.get(k, 0.0) for k in labels], dtype=float)
        elif isinstance(y0, (np.ndarray, list, tuple)):
            # Assume it's already a correctly ordered array/list
            y0v = np.asarray(y0, dtype=float)
            if len(y0v) != len(labels):
                raise ValueError(
                    f"y0 array has length {len(y0v)}, but model has "
                    f"{len(labels)} states: {labels}"
                )
        else:
            raise TypeError(
                f"y0 must be a dict, numpy.ndarray, list, or tuple, "
                f"got {type(y0)}"
            )
        # --- END NEW LOGIC ---

        def f(t, y):
            dy = self.rhs(t, y, beta_fn)
            return dy

        method = solve_kw.pop("method", "RK45")
        try:
            sol = solve_ivp(f, t_span, y0v, method=method, t_eval=t_eval, **solve_kw)
        except Exception:
            if stiff_fallback and method != "BDF":
                sol = solve_ivp(f, t_span, y0v, method="BDF", t_eval=t_eval, **solve_kw)
            else:
                raise

        return sol.t, sol.y
