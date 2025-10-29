# kr_epi/analysis/__init__.py

from .equilibria import find_endemic_equilibrium
from .spectra import psd_welch, dominant_peaks, ensemble_psd
from .stability import calculate_jacobian, analyze_stability
from .sensitivity import calculate_local_sensitivity

__all__ = [
    # equilibria
    "find_endemic_equilibrium",
    # spectra
    "psd_welch",
    "dominant_peaks",
    "ensemble_psd",
    # stability
    "calculate_jacobian",
    "analyze_stability",
    # sensitivity
    "calculate_local_sensitivity"
]