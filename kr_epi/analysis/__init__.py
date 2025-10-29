"""
Analysis tools for epidemic models.
"""

from .equilibria import (
    sis_equilibria,
    sir_final_size,
    sirs_equilibria,
    critical_vaccination_coverage,
    vaccination_impact,
    sir_attack_rate,
    R0_from_parameters,
)

from .spectra import (
    psd_welch,
    dominant_peaks,
)

__all__ = [
    # Equilibria
    'sis_equilibria',
    'sir_final_size',
    'sirs_equilibria',
    'critical_vaccination_coverage',
    'vaccination_impact',
    'sir_attack_rate',
    'R0_from_parameters',
    # Spectra
    'psd_welch',
    'dominant_peaks',
]