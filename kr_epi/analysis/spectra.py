"""
Spectral analysis tools for epidemic time series.

Provides functions for analyzing periodicities and frequencies in
epidemic dynamics, particularly useful for seasonal forcing analysis.
"""

import numpy as np
from scipy.signal import welch, find_peaks
from typing import Tuple, Optional

# Type alias
Array = np.ndarray


def psd_welch(
    signal: Array,
    fs: float,
    nperseg: Optional[int] = None,
    **welch_kwargs
) -> Tuple[Array, Array]:
    """
    Calculate Power Spectral Density using Welch's method.
    
    Welch's method divides the signal into overlapping segments,
    computes a periodogram for each segment, and averages them.
    This reduces noise compared to a single periodogram.
    
    Parameters
    ----------
    signal : array_like
        Input time series (1D array)
    fs : float
        Sampling frequency (inverse of time step)
        Example: if time step = 0.1 days, fs = 10 per day
    nperseg : int, optional
        Length of each segment. If None, uses default from scipy.
        Larger nperseg gives better frequency resolution but more variance.
    **welch_kwargs
        Additional keyword arguments passed to scipy.signal.welch
        
    Returns
    -------
    f : ndarray
        Array of sample frequencies
    Pxx : ndarray
        Power spectral density of signal
        
    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 100, 1000)
    >>> signal = np.sin(2*np.pi*0.1*t) + 0.1*np.random.randn(1000)
    >>> f, Pxx = psd_welch(signal, fs=10)
    >>> dominant_freq = f[np.argmax(Pxx)]
    >>> print(f"Dominant frequency: {dominant_freq:.3f} cycles/day")
    
    Notes
    -----
    For epidemic time series, common periodicities include:
    - Annual cycles: ~365 days (frequency ~0.00274 per day)
    - School term: ~weeks to months
    - Multi-year cycles: several years (frequency < 0.001 per day)
    
    References
    ----------
    Welch, P. (1967). The use of fast Fourier transform for the estimation
    of power spectra. IEEE Transactions on Audio and Electroacoustics, 15(2), 70-73.
    
    See Also
    --------
    dominant_peaks : Find dominant frequencies in PSD
    ensemble_psd : Average PSD across multiple realizations
    """
    # Ensure signal is 1D
    signal = np.asarray(signal).ravel()
    
    # Calculate PSD using Welch's method
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, **welch_kwargs)
    
    return f, Pxx


def dominant_peaks(
    f: Array,
    Pxx: Array,
    height: Optional[float] = None,
    prominence: Optional[float] = None,
    distance: Optional[int] = None,
    num_peaks: int = 5
) -> Tuple[Array, Array, Array]:
    """
    Find dominant peaks in power spectral density.
    
    Identifies the most prominent frequencies in a PSD,
    useful for detecting periodic patterns in epidemic data.
    
    Parameters
    ----------
    f : array_like
        Frequency array (from psd_welch)
    Pxx : array_like
        Power spectral density array (from psd_welch)
    height : float, optional
        Minimum height of peaks (in power units)
        If None, uses median(Pxx)
    prominence : float, optional
        Minimum prominence of peaks
        If None, uses 0.1 * max(Pxx)
    distance : int, optional
        Minimum distance between peaks (in array indices)
    num_peaks : int, default=5
        Maximum number of peaks to return
        
    Returns
    -------
    peak_freqs : ndarray
        Frequencies of detected peaks (Hz or per time unit)
    peak_powers : ndarray
        Power values at peaks
    peak_periods : ndarray
        Periods corresponding to peak frequencies (1/f)
        
    Examples
    --------
    >>> f, Pxx = psd_welch(epidemic_time_series, fs=1.0)  # fs=1 per day
    >>> freqs, powers, periods = dominant_peaks(f, Pxx)
    >>> for freq, period in zip(freqs, periods):
    ...     print(f"Frequency: {freq:.4f}/day, Period: {period:.1f} days")
    
    Notes
    -----
    For epidemic time series with daily sampling (fs=1 per day):
    - Annual cycle: frequency ≈ 0.00274/day, period ≈ 365 days
    - 6-month cycle: frequency ≈ 0.00548/day, period ≈ 182 days
    - Biennial: frequency ≈ 0.00137/day, period ≈ 730 days
    
    See Also
    --------
    psd_welch : Calculate PSD
    scipy.signal.find_peaks : Peak detection algorithm used internally
    """
    f = np.asarray(f)
    Pxx = np.asarray(Pxx)
    
    # Set default height threshold
    if height is None:
        height = np.median(Pxx)
    
    # Set default prominence threshold
    if prominence is None:
        prominence = 0.1 * np.max(Pxx)
    
    # Find peaks using scipy
    peak_indices, properties = find_peaks(
        Pxx,
        height=height,
        prominence=prominence,
        distance=distance
    )
    
    # Extract peak information
    peak_freqs = f[peak_indices]
    peak_powers = Pxx[peak_indices]
    
    # Calculate periods (avoid division by zero)
    with np.errstate(divide='ignore', invalid='ignore'):
        peak_periods = np.where(peak_freqs > 0, 1.0 / peak_freqs, np.inf)
    
    # Sort by power (descending) and take top num_peaks
    sort_idx = np.argsort(peak_powers)[::-1]
    sort_idx = sort_idx[:num_peaks]
    
    peak_freqs = peak_freqs[sort_idx]
    peak_powers = peak_powers[sort_idx]
    peak_periods = peak_periods[sort_idx]
    
    return peak_freqs, peak_powers, peak_periods


def ensemble_psd(
    times: Array,
    series: Array,
    state_idx: int,
    fs: float,
    nperseg: Optional[int] = None
) -> Tuple[Array, Array]:
    """
    Calculate average Power Spectral Density across ensemble realizations.
    
    For stochastic simulations, averaging PSDs across multiple realizations
    reduces noise and reveals underlying periodicities more clearly.
    
    Parameters
    ----------
    times : array_like
        Time points (must be regularly spaced)
    series : array_like
        State trajectories with shape (n_realizations, n_states, n_times)
        or (n_states, n_times) for single realization
    state_idx : int
        Index of state variable to analyze
        Example: 1 for 'I' in SIR model
    fs : float
        Sampling frequency (1 / time_step)
        Example: if Δt = 0.1 days, fs = 10 per day
    nperseg : int, optional
        Segment length for Welch's method
        
    Returns
    -------
    f_avg : ndarray
        Array of frequencies
    Pxx_avg : ndarray
        Average power spectral density across realizations
        
    Examples
    --------
    >>> from kr_epi.sweeps.runners import run_ensemble
    >>> ens = run_ensemble(engine, x0={'S': 990, 'I': 10}, t_max=1000, n_runs=50)
    >>> f, Pxx = ensemble_psd(ens.times, ens.series, state_idx=1, fs=1.0)
    >>> plot_psd(f, Pxx)
    
    Notes
    -----
    This function is particularly useful for analyzing stochastic epidemic
    models where individual realizations may be noisy. By averaging PSDs,
    we can identify robust periodic patterns that appear consistently
    across simulations.
    
    For seasonal forcing analysis, compare the dominant peak in the ensemble
    PSD to the forcing frequency to verify that the model responds to
    seasonal variation as expected.
    
    See Also
    --------
    psd_welch : Calculate PSD for single realization
    dominant_peaks : Extract dominant frequencies
    """
    series = np.asarray(series)
    times = np.asarray(times)
    
    # Verify times are regularly spaced
    if len(times) > 1:
        dt = np.diff(times)
        if not np.allclose(dt, dt[0], rtol=0.01):
            raise ValueError(
                "Times must be regularly spaced for spectral analysis. "
                f"Got dt ranging from {dt.min():.6f} to {dt.max():.6f}"
            )
    
    # Handle both ensemble (3D) and single realization (2D)
    if series.ndim == 2:
        # Single realization: (n_states, n_times)
        series = series[np.newaxis, :, :]  # Add realization dimension
    elif series.ndim != 3:
        raise ValueError(
            f"series must be 2D or 3D array, got shape {series.shape}"
        )
    
    n_realizations, n_states, n_times = series.shape
    
    # Validate state_idx
    if not 0 <= state_idx < n_states:
        raise ValueError(
            f"state_idx={state_idx} out of bounds for {n_states} states"
        )
    
    # Calculate PSD for each realization
    all_pxx = []
    f_common = None
    
    for i in range(n_realizations):
        # Extract time series for this state in this realization
        signal = series[i, state_idx, :]
        
        # Calculate PSD
        f, pxx = psd_welch(signal, fs=fs, nperseg=nperseg)
        
        # Verify frequency arrays are consistent
        if f_common is None:
            f_common = f
        elif not np.allclose(f_common, f):
            raise ValueError(
                "Frequency arrays differ between realizations. "
                "Ensure consistent fs and nperseg."
            )
        
        all_pxx.append(pxx)
    
    # Average PSDs across realizations
    pxx_avg = np.mean(all_pxx, axis=0)
    
    return f_common, pxx_avg


# Backward compatibility aliases
def calculate_psd(signal: Array, fs: float, **kwargs) -> Tuple[Array, Array]:
    """Alias for psd_welch. Deprecated, use psd_welch instead."""
    import warnings
    warnings.warn(
        "calculate_psd is deprecated, use psd_welch instead",
        DeprecationWarning,
        stacklevel=2
    )
    return psd_welch(signal, fs, **kwargs)


# Export all public functions
__all__ = [
    'psd_welch',
    'dominant_peaks',
    'ensemble_psd',
]