# kr_epi/forcing/term_time_smooth.py
from dataclasses import dataclass
import math
import numpy as np # Import numpy for tanh

@dataclass(frozen=True)
class TermTimeSmooth:
    """
    Smooth term-time (school) forcing using sigmoid functions.

    Approximates a square wave where beta(t) varies smoothly between
    beta0*(1-amp) and beta0*(1+amp).

    Parameters
    ----------
    beta0 : float
        Baseline transmission rate (average value).
    amp : float
        Relative amplitude of variation (0 <= amp < 1).
        beta will oscillate between beta0*(1-amp) and beta0*(1+amp).
    period : float
        Repeat period, usually 365 (days).
    term_frac : float, optional
        Approximate fraction of the year in "high transmission" (term) time.
        Defaults to 0.6 (approx 219 days). This defines the center
        of the high/low plateaus.
    edge : float, optional
        Controls the smoothness/sharpness of transitions (in days).
        Smaller values (~1-5) give sharper edges, larger values give
        gentler transitions. Defaults to 5.0.
    phase : float, optional
        Phase shift (in days) added before modulo operation.
        Adjusts the timing of the peaks/troughs. Defaults to 0.0.
    """
    beta0: float
    amp: float = 0.0
    period: float = 365.0
    term_frac: float = 0.6
    edge: float = 5.0
    phase: float = 0.0

    def __post_init__(self):
        if not 0 <= self.amp < 1:
            raise ValueError(f"Amplitude (amp) must be in [0, 1), got {self.amp}")
        if self.edge <= 0:
            raise ValueError(f"Edge smoothness must be positive, got {self.edge}")
        if not 0 < self.term_frac < 1:
             raise ValueError(f"Term fraction must be between 0 and 1, got {self.term_frac}")

    def __call__(self, t: float) -> float:
        """Calculates the smooth time-varying beta(t)."""
        if self.amp == 0.0 or self.beta0 == 0.0:
            return self.beta0

        # Effective time within the cycle [0, period)
        t_eff = (t + self.phase) % self.period

        # --- Use hyperbolic tangent (tanh) for smooth step ---
        # Tanh transitions smoothly from -1 to +1.
        # We'll create two transitions per period.
        # Adjusting the phase/center might be needed depending on desired peak time.

        # Calculate width of the "high" plateau (term time)
        term_width = self.term_frac * self.period
        # Calculate center points for transitions (simplistic approach assumes symmetry)
        # Transition up center (start of term)
        t_up = (self.period - term_width) / 2.0
        # Transition down center (end of term)
        t_down = t_up + term_width

        # Calculate the smooth wave using tanh.
        # The 'edge' parameter scales the steepness. Smaller edge -> steeper transition.
        # We need a factor (like pi or 2) to get a reasonable transition width.
        # Using 2/edge makes the transition roughly span a few multiples of 'edge'.
        steepness = 2.0 / self.edge

        # Sigmoid up from low to high
        s_up = np.tanh(steepness * (t_eff - t_up))
        # Sigmoid down from high to low (inverted and shifted)
        s_down = -np.tanh(steepness * (t_eff - t_down))

        # Combine: This creates a wave that is approx +1 during term, -1 otherwise
        # Add 1 to each tanh (range 0 to 2), sum (range 0 to 4), shift (range -2 to 2), scale (range -1 to 1)
        s_combined = ((s_up + 1.0) + (s_down + 1.0)) / 2.0 - 1.0

        # Apply amplitude and baseline
        # beta(t) = beta0 * (1 + amp * s_combined)
        beta_t = self.beta0 * (1.0 + self.amp * s_combined)

        # Ensure beta_t remains non-negative (important if amp is close to 1)
        return max(0.0, beta_t)