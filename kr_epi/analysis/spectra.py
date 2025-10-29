# kr_epi/analysis/spectra.py
# ... (keep existing imports and functions) ...
import numpy as np # Make sure numpy is imported
from scipy.signal import welch, find_peaks
from kr_epi.sweeps.runners import EnsembleResult # Add this import

# ... (psd_welch and dominant_peaks functions remain here) ...

def ensemble_psd(ens_result: EnsembleResult, state_name: str, fs: float, nperseg: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculates the average Power Spectral Density (PSD) across all runs
    in an ensemble for a specific state variable.

    Parameters
    ----------
    ens_result : EnsembleResult
        The result object from run_ensemble. Assumes times are regularly spaced.
    state_name : str
        The name of the state variable to analyze (e.g., 'Y' or 'I').
    fs : float
        The sampling frequency of the time series (1 / time_step).
    nperseg : int | None, optional
        Length of each segment for Welch's method. If None, uses default
        (often 256). See scipy.signal.welch documentation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - f_avg: Array of sample frequencies common to all runs.
        - pxx_avg: Array of the averaged power spectral density across runs.

    Raises
    ------
    ValueError
        If state_name is not found in the ensemble labels.
    """
    try:
        state_idx = ens_result.labels.index(state_name)
    except ValueError:
        raise ValueError(f"State '{state_name}' not found in labels {ens_result.labels}")

    all_pxx = []
    f_common = None

    # Calculate PSD for each run
    for i in range(ens_result.series.shape[0]): # Iterate through runs
        y_run = ens_result.series[i, state_idx, :] # Get time series for the state in this run
        f, pxx = psd_welch(y_run, fs=fs, nperseg=nperseg)

        if f_common is None:
            f_common = f
        elif not np.allclose(f_common, f):
            # This shouldn't happen if fs and nperseg are constant, but good to check
            raise ValueError("Frequency arrays differ between runs. Ensure consistent parameters.")

        all_pxx.append(pxx)

    if not all_pxx: # Handle empty ensemble case
         return np.array([]), np.array([])

    # Average the PSDs across all runs
    pxx_avg = np.mean(np.array(all_pxx), axis=0)

    # mypy complains about f_common potentially being None if loop doesn't run
    if f_common is None:
        return np.array([]), np.array([])

    return f_common, pxx_avg