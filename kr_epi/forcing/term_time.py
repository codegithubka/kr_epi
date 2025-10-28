from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Tuple

@dataclass(frozen=True)
class TermTime:
    """
    Term-time (school) forcing:
      beta(t) = beta0 * (1 + amp) during 'term' windows,
                beta(t) = beta0 * (1 - amp) otherwise.

    Parameters
    ----------
    beta0 : float
        Baseline transmission.
    amp : float
        Amplitude in [0, 1). Typical 0.1–0.3.
    period : float
        Repeat period, usually 365 (days).
    term_windows : Iterable[Tuple[float, float]]
        (start_day, end_day) intervals within [0, period). If end < start, the
        interval wraps around period. Example:
        [(0, 45), (70, 120), (150, 200), (230, 285), (310, 340)]
    phase : float
        Phase shift (days) added before modulo.
    """
    beta0: float
    amp: float = 0.0
    period: float = 365.0
    term_windows: Iterable[Tuple[float, float]] = ()
    phase: float = 0.0

    def __call__(self, t: float) -> float:
        if self.amp == 0.0 or self.beta0 == 0.0 or not self.term_windows:
            return self.beta0
        # map t to [0, period)
        x = (t + self.phase) % self.period
        in_term = False
        for (start, end) in self.term_windows:
            if start <= end:
                if start <= x <= end:
                    in_term = True
                    break
            else:
                # wrapped interval
                if x >= start or x <= end:
                    in_term = True
                    break
        return self.beta0 * (1.0 + self.amp if in_term else 1.0 - self.amp)
