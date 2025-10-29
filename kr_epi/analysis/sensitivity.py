# kr_epi/analysis/sensitivity.py
from __future__ import annotations
from typing import Callable, Dict, Any, List, Optional, Tuple, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from kr_epi.models.base import ODEBase

Array = np.ndarray

def calculate_local_sensitivity(
    model_class: Callable[..., ODEBase],
    baseline_params: Dict[str, Any],
    param_names: List[str],
    metric_func: Callable[[Array, Array], float],
    y0: Array,
    t_span: Tuple[float, float],
    *,
    t_eval: Optional[Array] = None,
    beta_fn: Optional[Callable[[float], float]] = None,
    perturbation_frac: float = 0.01
) -> Dict[str, float]:
    """
    Calculates the local, one-at-a-time (OAT) sensitivity of an output
    metric to specified parameters.

    It computes the "elasticity" or normalized sensitivity index:
    S_p = (dM / M) / (dp / p) = (dM / dp) * (p / M)

    Parameters
    ----------
    model_class : Callable[..., ODEBase]
        The constructor for the model (e.g., SIRDemographyCounts).
    baseline_params : Dict[str, Any]
        Dictionary of baseline parameter values for the model.
    param_names : List[str]
        A list of parameter names (must be keys in baseline_params)
        to analyze.
    metric_func : Callable[[Array, Array], float]
        A function that takes (t, y) and returns a single scalar
        metric (e.g., lambda t, y: np.max(y[1]) for peak infection).
    y0 : Array
        Initial state vector for the simulation.
    t_span : Tuple[float, float]
        Simulation time span (t_min, t_max).
    t_eval : Optional[Array], optional
        Times to evaluate the solution at. Defaults to None.
    beta_fn : Optional[Callable[[float], float]], optional
        Seasonal forcing function, if any. Defaults to None.
    perturbation_frac : float, optional
        The relative amount to perturb each parameter (e.g., 0.01 = 1%).
        Defaults to 0.01.

    Returns
    -------
    Dict[str, float]
        A dictionary mapping parameter names to their normalized
        sensitivity index (elasticity).

    Raises
    ------
    ValueError
        If a parameter name is not in baseline_params, or if the
        baseline metric or parameter value is zero (preventing division).
    """
    sensitivities = {}

    # --- 1. Run baseline simulation and get baseline metric ---
    try:
        model_base = model_class(**baseline_params)
    except Exception as e:
        raise ValueError(f"Failed to instantiate model with baseline params: {e}")

    t_base, y_base = model_base.integrate(y0, t_span, t_eval=t_eval, beta_fn=beta_fn)
    metric_base = metric_func(t_base, y_base)

    if np.isclose(metric_base, 0.0):
        print(f"Warning: Baseline metric is zero. Cannot calculate normalized sensitivities.")
        return {p_name: np.nan for p_name in param_names}

    # --- 2. Loop through parameters, perturb, and re-run ---
    for p_name in param_names:
        if p_name not in baseline_params:
            raise ValueError(f"Parameter '{p_name}' not found in baseline_params.")

        p_base = baseline_params[p_name]
        if not isinstance(p_base, (int, float)):
             print(f"Warning: Skipping non-numeric param '{p_name}'.")
             continue
        
        if np.isclose(p_base, 0.0):
            print(f"Warning: Skipping param '{p_name}' with baseline value of zero.")
            sensitivities[p_name] = 0.0 # Or np.nan
            continue
            
        # Calculate perturbation (h = dp)
        h = p_base * perturbation_frac
        p_perturbed = p_base + h

        # Create perturbed parameters
        params_pert = baseline_params.copy()
        params_pert[p_name] = p_perturbed

        # --- 3. Run perturbed simulation ---
        try:
            model_pert = model_class(**params_pert)
        except Exception as e:
            print(f"Warning: Failed to instantiate model for param '{p_name}': {e}. Skipping.")
            sensitivities[p_name] = np.nan
            continue
            
        t_pert, y_pert = model_pert.integrate(y0, t_span, t_eval=t_eval, beta_fn=beta_fn)
        metric_pert = metric_func(t_pert, y_pert)

        # --- 4. Calculate sensitivity ---
        dM = metric_pert - metric_base
        dp = h

        # Raw sensitivity: dM / dp
        sensitivity_raw = dM / dp
        
        # Normalized sensitivity (Elasticity): (dM / dp) * (p / M)
        sensitivity_norm = sensitivity_raw * (p_base / metric_base)
        
        sensitivities[p_name] = sensitivity_norm

    return sensitivities