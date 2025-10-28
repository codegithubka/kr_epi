from dataclasses import dataclass
import math

@dataclass(frozen=True)
class BetaSinusoid:
    """
    Sinusoidal seasonal forcing for the transmission rate β(t).

    β(t) = β0 * [1 + a * cos(2πt / period + phase)]

    Parameters
    ----------
    beta0 : float
        Baseline transmission rate.
    amp : float
        Amplitude of seasonal variation (0 ≤ amp < 1). Example: 0.2 for ±20% variation.
    period : float
        Period of oscillation (default: 365 days).
    phase : float
        Phase shift (radians). phase = 0 means maximum β at t = 0.
    """
    beta0: float
    amp: float = 0.0
    period: float = 365.0
    phase: float = 0.0

    def __call__(self, t: float) -> float:
        if self.amp == 0.0:
            return self.beta0
        return self.beta0 * (1.0 + self.amp * math.cos(2 * math.pi * t / self.period + self.phase))
