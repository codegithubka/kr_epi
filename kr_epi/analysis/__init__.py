# kr_epi/analysis/__init__.py

# Import functions that actually exist in your files
from .equilibria import (
    sis_equilibria, 
    sir_final_size, 
    sirs_equilibria, 
    critical_vaccination_coverage,
    vaccination_impact,
    sir_attack_rate,
    R0_from_parameters
)
from .spectra import psd_welch, dominant_peaks, ensemble_psd
from .stability import calculate_jacobian, analyze_stability
from .sensitivity import calculate_local_sensitivity

__all__ = [
    # from equilibria.py
    "sis_equilibria",
    "sir_final_size",
    "sirs_equilibria",
    "critical_vaccination_coverage",
    "vaccination_impact",
    "sir_attack_rate",
    "R0_from_parameters",
    
    # from spectra.py
    "psd_welch",
    "dominant_peaks",
    "ensemble_psd",
    
    # from stability.py
    "calculate_jacobian",
    "analyze_stability",
    
    # from sensitivity.py
    "calculate_local_sensitivity"
]