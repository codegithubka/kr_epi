# kr_epi/forcing/term_time_smooth.py
from dataclasses import dataclass
import math
@dataclass(frozen=True)
class TermTimeSmooth:
    beta0: float; amp: float=0.0; period: float=365.0
    term_frac: float=0.6  # fraction of the year in "term"
    edge: float=5.0       # smoothness (days): smaller = sharper edges
    phase: float=0.0      # days
    def __call__(self, t: float) -> float:
        if self.amp==0.0: return self.beta0
        x = (t + self.phase) % self.period
        center = self.term_frac*self.period/2.0
        # 2 plateaus per period; combine two sigmoids to make a smooth square wave
        def sq(x, c):
            return 1/(1+math.exp(-(x-c)/self.edge)) - 1/(1+math.exp(-(x-(c+self.period/2))/self.edge))
        s = sq(x, center) - sq(x, center + self.period/2)
        s = (s - (s.min() if hasattr(s,'min') else -1))  # keep amplitude roughly ±1
        return self.beta0 * (1.0 + self.amp * (1 if s>=0 else -1))
