# kr_epi/plotting.py
from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Sequence, Tuple, List
import numpy as np
from kr_epi.sweeps.runners import EnsembleResult # Add this import
from typing import Callable # Ensure Callable is imported
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

if TYPE_CHECKING:
    from kr_epi.models.base import ODEBase

Array = np.ndarray


def plot_ode_ts(
    t: Array,
    y: Array,
    labels: Sequence[str],
    *,
    ax: Optional[Axes] = None,
    title: str = "Epidemic Time Series",
    xlabel: str = "Time",
    ylabel: str = "Fraction",
) -> Axes:
    """
    Plots the time series of an ODE model's compartments.

    Parameters
    ----------
    t : Array
        Time points (1D array).
    y : Array
        State variables (shape: n_states x n_timepoints).
    labels : Sequence[str]
        List of labels for each state variable.
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    for i, label in enumerate(labels):
        ax.plot(t, y[i, :], label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return ax


def plot_phase_portrait(
    model: ODEBase,
    x_label: str,
    y_label: str,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    *,
    trajectories: Optional[List[Tuple[Array, Array]]] = None,
    equilibria: Optional[List[Array]] = None,
    show_vector_field: bool = True,
    show_nullclines: bool = True,
    ax: Optional[Axes] = None,
    title: str = "Phase Portrait",
) -> Axes:
    """
    Plots a 2D phase portrait for an ODE model, including vector field,
    nullclines, and optional trajectories/equilibria.

    Note: This is intended for 2D models (e.g., SI, SIR, SIS).
          For 3D+ models, it plots the first two state variables.

    Parameters
    ----------
    model : ODEBase
        An instance of an ODE model (e.g., SIR).
    x_label : str
        Label for the state variable on the x-axis (e.g., 'S').
    y_label : str
        Label for the state variable on the y-axis (e.g., 'I').
    x_range : Tuple[float, float]
        (min, max) range for the x-axis.
    y_range : Tuple[float, float]
        (min, max) range for the y-axis.
    trajectories : Optional[List[Tuple[Array, Array]]], optional
        A list of (t, y) tuples from model.integrate() to plot.
    equilibria : Optional[List[Array]], optional
        A list of [x, y] points (e.g., DFE, endemic) to plot as markers.
    show_vector_field : bool, optional
        Whether to draw the vector field (default: True).
    show_nullclines : bool, optional
        Whether to draw the nullclines (default: True).
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    title : str, optional
        Plot title.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 6))

    if len(model.labels) < 2:
        raise ValueError("Phase portrait requires at least a 2D model.")

    x_idx = model.labels.index(x_label)
    y_idx = model.labels.index(y_label)

    # --- Create grid for vector field and nullclines ---
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_range[0], x_range[1], 20),
        np.linspace(y_range[0], y_range[1], 20),
    )
    
    # Initialize derivative grids
    dxdt_grid = np.zeros_like(x_grid)
    dydt_grid = np.zeros_like(x_grid)
    
    # Create a dummy state vector (use DFE as default for other states)
    # This is imperfect but works for simple models where S/I are primary.
    # A more complex model might need a `point` argument to evaluate around.
    y_base = np.zeros(len(model.labels))
    if equilibria: # Use first equilibrium as base if available
        y_base = equilibria[0].copy()

    # --- Calculate derivatives on the grid ---
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            y_point = y_base.copy()
            y_point[x_idx] = x_grid[i, j]
            y_point[y_idx] = y_grid[i, j]
            
            # Assume t=0 for RHS evaluation
            derivs = model.rhs(0, y_point) 
            dxdt_grid[i, j] = derivs[x_idx]
            dydt_grid[i, j] = derivs[y_idx]

    # --- Plot Vector Field ---
    if show_vector_field:
        # Normalize vectors for better visualization
        magnitudes = np.sqrt(dxdt_grid**2 + dydt_grid**2)
        magnitudes[magnitudes == 0] = 1.0 # Avoid division by zero
        ax.quiver(
            x_grid, y_grid, 
            dxdt_grid / magnitudes, dydt_grid / magnitudes, 
            angles="xy", scale_units="xy", scale=25, 
            alpha=0.4, color="gray"
        )

    # --- Plot Nullclines ---
    if show_nullclines:
        ax.contour(
            x_grid, y_grid, dxdt_grid, levels=[0], 
            colors="red", linestyles="--", linewidths=1.5
        )
        ax.contour(
            x_grid, y_grid, dydt_grid, levels=[0], 
            colors="blue", linestyles="--", linewidths=1.5
        )
        # Add dummy plots for legend
        ax.plot([], [], 'r--', label=f'{y_label}-nullcline (d{y_label}/dt=0)')
        ax.plot([], [], 'b--', label=f'{x_label}-nullcline (d{x_label}/dt=0)')


    # --- Plot Trajectories ---
    if trajectories:
        for k, (t, y) in enumerate(trajectories):
            ax.plot(y[x_idx, :], y[y_idx, :], c="black", lw=1.5, alpha=0.8, 
                    label=f"Trajectory {k+1}" if k==0 else None)
            ax.plot(y[x_idx, 0], y[y_idx, 0], 'o', c="black", ms=5) # Start point

    # --- Plot Equilibria ---
    if equilibria:
        for k, eq in enumerate(equilibria):
            ax.plot(eq[x_idx], eq[y_idx], 'X', c="purple", ms=10, 
                    label=f"Equilibrium {k+1}" if k==0 else None)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_title(title)
    if show_nullclines or trajectories or equilibria:
        ax.legend(fontsize='small')
    ax.grid(True, alpha=0.2)
    
    return ax


def plot_ensemble_summary(
    ens_result: EnsembleResult,
    *,
    states_to_plot: Optional[List[str]] = None,
    show_iqr: bool = True,
    iqr_quantiles: Tuple[float, float] = (0.25, 0.75),
    show_runs: int = 0,
    ax: Optional[Axes] = None,
    title: str = "Stochastic Ensemble Summary",
    xlabel: str = "Time",
    ylabel: str = "Count",
) -> Axes:
    """
    Plots a summary of a stochastic ensemble simulation.

    Shows the median trajectory, an optional inter-quantile range (IQR),
    and a specified number of individual "spaghetti" runs.

    Parameters
    ----------
    ens_result : EnsembleResult
        The result object from runners.run_ensemble.
    states_to_plot : Optional[List[str]], optional
        List of state labels to plot. If None, plots all.
    show_iqr : bool, optional
        Whether to show the shaded inter-quantile range (default: True).
    iqr_quantiles : Tuple[float, float], optional
        The (lower, upper) quantiles for the shaded range (default: 0.25, 0.75).
    show_runs : int, optional
        Number of individual trajectories to overlay (default: 0).
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    labels = list(ens_result.labels)
    t = ens_result.times
    series = ens_result.series  # Shape: (n_runs, n_states, n_times)

    if states_to_plot is None:
        indices_to_plot = list(range(len(labels)))
    else:
        indices_to_plot = [labels.index(s) for s in states_to_plot]

    colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

    for i, state_idx in enumerate(indices_to_plot):
        state_label = labels[state_idx]
        state_series = series[:, state_idx, :]  # (n_runs, n_times)

        # Plot Median
        median = np.median(state_series, axis=0)
        line, = ax.plot(t, median, label=f'{state_label} (Median)', color=colors[i], lw=2)

        # Plot IQR
        if show_iqr:
            q_low = np.quantile(state_series, iqr_quantiles[0], axis=0)
            q_high = np.quantile(state_series, iqr_quantiles[1], axis=0)
            ax.fill_between(
                t, q_low, q_high, alpha=0.2, color=line.get_color(),
                label=f'{state_label} ({iqr_quantiles[0]*100:.0f}-{iqr_quantiles[1]*100:.0f}%)'
            )

        # Plot individual runs
        if show_runs > 0:
            num_to_show = min(show_runs, series.shape[0])
            for run_idx in range(num_to_show):
                ax.plot(
                    t, state_series[run_idx, :], color=line.get_color(),
                    alpha=0.15, linewidth=0.5
                )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0) # Counts cannot be negative
    return ax


def plot_deterministic_vs_stochastic(
    t_ode: Array,
    y_ode: Array,
    t_stoch: Array,
    y_stoch: Array,
    labels: Sequence[str],
    *,
    states_to_plot: Optional[List[str]] = None,
    ax: Optional[Axes] = None,
    title: str = "Deterministic vs. Stochastic Simulation",
    xlabel: str = "Time",
    ylabel: str = "Count / Fraction",
) -> Axes:
    """
    Overlays a single stochastic run on top of a deterministic ODE run.

    Parameters
    ----------
    t_ode : Array
        Time points for the ODE simulation.
    y_ode : Array
        State variables for the ODE (shape: n_states x n_timepoints).
    t_stoch : Array
        Time points for the stochastic simulation.
    y_stoch : Array
        State variables for the stochastic run (shape: n_states x n_timepoints).
    labels : Sequence[str]
        List of labels for each state variable.
    states_to_plot : Optional[List[str]], optional
        List of state labels to plot. If None, plots all.
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
        
    if states_to_plot is None:
        indices_to_plot = list(range(len(labels)))
    else:
        indices_to_plot = [labels.index(s) for s in states_to_plot]

    colors = plt.cm.viridis(np.linspace(0, 1, len(indices_to_plot)))

    for i, state_idx in enumerate(indices_to_plot):
        state_label = labels[state_idx]
        color = colors[i]
        
        # Plot ODE (Deterministic)
        ax.plot(
            t_ode, y_ode[state_idx, :], color=color, 
            label=f'{state_label} (ODE)', lw=2
        )
        
        # Plot Stochastic
        ax.plot(
            t_stoch, y_stoch[state_idx, :], color=color, 
            label=f'{state_label} (Stochastic)', lw=1.5,
            linestyle='--', alpha=0.8
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize='small')
    ax.grid(True, alpha=0.3)
    return ax


def plot_forcing(
    beta_fn: Callable[[float], float],
    t_span: Tuple[float, float],
    *,
    ax: Optional[Axes] = None,
    n_points: int = 500,
    title: str = "Forcing Function",
    xlabel: str = "Time",
    ylabel: str = "Beta(t)",
) -> Axes:
    """
    Plots the value of a forcing function over a time span.

    Parameters
    ----------
    beta_fn : Callable[[float], float]
        The forcing function (e.g., BetaSinusoid instance).
    t_span : Tuple[float, float]
        (t_min, t_max) to plot over.
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    n_points : int, optional
        Number of points to evaluate for the plot (default: 500).
    title : str, optional
        Plot title.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 3))

    t = np.linspace(t_span[0], t_span[1], n_points)
    y = [beta_fn(t_i) for t_i in t]

    ax.plot(t, y, lw=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0) # Beta cannot be negative
    
    return ax


def plot_psd(
    f: Array,
    pxx: Array,
    *,
    peaks_idx: Optional[Array] = None,
    plot_as_period: bool = True,
    log_x: bool = True,
    log_y: bool = True,
    ax: Optional[Axes] = None,
    title: str = "Power Spectral Density (PSD)",
) -> Axes:
    """
    Plots the Power Spectral Density (PSD) from a frequency analysis.

    Can plot either Frequency vs. Power or Period vs. Power.

    Parameters
    ----------
    f : Array
        Array of sample frequencies (from psd_welch).
    pxx : Array
        Array of power spectral density (from psd_welch).
    peaks_idx : Optional[Array], optional
        Indices of dominant peaks (from dominant_peaks) to mark on the plot.
    plot_as_period : bool, optional
        If True (default), plots Period (1/f) on the x-axis.
        If False, plots Frequency (f) on the x-axis.
    log_x : bool, optional
        Whether to use a log scale for the x-axis (default: True).
    log_y : bool, optional
        Whether to use a log scale for the y-axis (default: True).
    ax : Optional[Axes], optional
        Matplotlib axes to plot on. If None, a new one is created.
    title : str, optional
        Plot title.

    Returns
    -------
    Axes
        The axes object containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))

    xlabel = "Frequency (1 / time unit)"
    x_data = f

    if plot_as_period:
        # Convert frequency to period
        # Handle f=0 (DC component) by setting period to infinity
        with np.errstate(divide='ignore'):
            period = 1.0 / f
        
        # We'll plot period vs power
        x_data = period
        xlabel = "Period (time units)"
        
        # Exclude the DC component (inf period) from the plot if present
        if f[0] == 0:
            x_data = x_data[1:]
            pxx = pxx[1:]
            if peaks_idx is not None:
                # Adjust peak indices to account for removing the first element
                peaks_idx = peaks_idx[peaks_idx > 0] - 1
    
    # Plot the PSD line
    ax.plot(x_data, pxx, label="PSD")

    # Plot the peaks, if provided
    if peaks_idx is not None:
        if peaks_idx.size > 0:
            ax.plot(
                x_data[peaks_idx], pxx[peaks_idx], "x", ms=8, mew=2,
                label=f"Dominant Peaks"
            )
            # Annotate the peak periods/frequencies
            for idx in peaks_idx:
                ax.text(
                    x_data[idx], pxx[idx], f" {x_data[idx]:.1f}", 
                    verticalalignment='bottom', horizontalalignment='left',
                    fontsize='small'
                )

    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    
    if plot_as_period:
        # Invert x-axis so long periods are on the left
        ax.invert_xaxis()

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Power / Frequency")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3) # "both" is good for log scales
    if peaks_idx is not None:
        ax.legend()
        
    return ax