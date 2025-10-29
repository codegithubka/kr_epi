# kr_epi/analysis/stability.py
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from kr_epi.models.base import ODEBase

Array = np.ndarray

def calculate_jacobian(model: ODEBase, y_eq: np.ndarray, t: float = 0.0, step: float = 1e-6) -> Array:
    """
    Calculates the Jacobian matrix of the model's RHS function at a
    specific equilibrium point using numerical central differences.

    J[i, j] = d(rhs_i) / dy_j

    Parameters
    ----------
    model : ODEBase
        An instance of an ODE model (e.g., SIR, SEIRDemographyCounts).
    y_eq : np.ndarray
        The state vector at the equilibrium point.
    t : float, optional
        The time at which to evaluate the Jacobian (matters if rhs is
        time-dependent, though equilibria are usually time-invariant
        unless analyzing a snapshot). Defaults to 0.0.
    step : float, optional
        The finite difference step size. Defaults to 1e-6.

    Returns
    -------
    np.ndarray
        The (n_states x n_states) Jacobian matrix.
    """
    n_states = len(y_eq)
    jacobian = np.zeros((n_states, n_states), dtype=float)
    h_step = np.zeros(n_states, dtype=float)

    # Use central differences for better accuracy: f'(x) approx (f(x+h) - f(x-h)) / (2h)
    for j in range(n_states):
        # Create perturbation vector h_j
        h_step_vec = np.zeros(n_states)
        h_step_vec[j] = step

        y_plus = y_eq + h_step_vec
        y_minus = y_eq - h_step_vec

        # Calculate rhs at (y + h) and (y - h)
        # Note: beta_fn is not passed, assuming equilibrium analysis
        # uses the model's internal baseline parameters.
        rhs_plus = model.rhs(t, y_plus)
        rhs_minus = model.rhs(t, y_minus)

        # Calculate the j-th column of the Jacobian
        col_j = (rhs_plus - rhs_minus) / (2.0 * step)
        jacobian[:, j] = col_j

    return jacobian


def analyze_stability(model: ODEBase, point: np.ndarray, t: float = 0.0) -> tuple[Array, float, bool]:
    """
    Analyzes the local stability of an equilibrium point by calculating
    the eigenvalues of the Jacobian matrix.

    Parameters
    ----------
    model : ODEBase
        An instance of an ODE model.
    point : np.ndarray
        The state vector at the equilibrium point (DFE or endemic).
    t : float, optional
        The time at which to evaluate. Defaults to 0.0.

    Returns
    -------
    tuple[np.ndarray, float, bool]
        - eigvals: The array of (potentially complex) eigenvalues.
        - max_real_part: The dominant eigenvalue's real part.
        - is_stable: True if max_real_part < 0 (with tolerance), False otherwise.
    """
    # Calculate the Jacobian at the equilibrium point
    jac = calculate_jacobian(model, point, t=t)

    # Calculate the eigenvalues
    eigvals = np.linalg.eigvals(jac)

    # Find the maximum real part (this determines stability)
    max_real_part = np.max(np.real(eigvals))

    # Check for stability (use a small tolerance)
    # If max_real_part is negative, all real parts are negative (stable node/spiral)
    # If max_real_part is positive, at least one is positive (unstable node/spiral, saddle)
    is_stable = max_real_part < -1e-12 # Tolerance for numerical zero

    return eigvals, max_real_part, is_stable