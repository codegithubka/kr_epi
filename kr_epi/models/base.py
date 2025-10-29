#/!/usr/bin/env python3


from typing import Callable, Dict, Optional, Sequence, Tuple
import numpy as np
from scipy.integrate import solve_ivp

Array = np.ndarray

class ODEBase:
    def state_labels(self) -> Sequence[str]:
        raise NotImplementedError

    def rhs(self, t: float, y: Array, beta_fn: Optional[Callable[[float], float]] = None) -> Array: # type: ignore
        raise NotImplementedError

    def integrate(
        self,
        y0: Dict[str, float],
        t_span: Tuple[float, float],
        t_eval: Optional[Array] = None, # type: ignore
        beta_fn: Optional[Callable[[float], float]] = None,
        stiff_fallback: bool = True,
        guard_negatives: bool = True,
        **solve_kw
    ):
        labels = list(self.state_labels())
        y0v = np.array([y0.get(k, 0.0) for k in labels], dtype=float)

        def f(t, y):
            dy = self.rhs(t, y, beta_fn)
            if guard_negatives:
                y[:] = np.maximum(y, -1e-12)
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
