from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from kr_epi.stochastic.direct import Direct

Array = np.ndarray

@dataclass
class EnsembleResult:
    times: Array                 # type: ignore # common grid
    series: Array                # shape (n_runs, n_states, n_times)
    labels: Tuple[str, ...]
    seeds: List[int]

def run_ensemble(direct: Direct, x0: Dict[str, float], t_max: float,
                 *, n_runs: int = 50, seed0: int = 7,
                 record_times: Optional[Array] = None) -> EnsembleResult:
    seeds = [seed0 + i for i in range(n_runs)]
    series = []
    times_ref = None

    for s in seeds:
        sim = Direct(direct.system, seed=s)
        t, X = sim.run(x0, t_max, record_times=record_times)
        if record_times is not None:
            times = record_times
        else:
            # Use the longest time grid encountered (fallback – best to pass record_times)
            times = t
        times_ref = times if times_ref is None else times_ref
        series.append(X)

    arr = np.stack(series, axis=0)  # (n_runs, n_states, n_times)
    return EnsembleResult(times=times_ref, series=arr, labels=direct.system.labels, seeds=seeds)

def extinction_probability(ens: EnsembleResult, state_name: str = "Y", threshold: float = 0.5) -> float:
    """Fraction of runs that go extinct by the final time (state <= threshold)."""
    idx = ens.labels.index(state_name)
    final_vals = ens.series[:, idx, -1]
    return float((final_vals <= threshold).mean())

def ensemble_summary(ens: EnsembleResult):
    """Return time, median and IQR for each state."""
    q50 = np.median(ens.series, axis=0)              # (n_states, n_times)
    q25 = np.quantile(ens.series, 0.25, axis=0)
    q75 = np.quantile(ens.series, 0.75, axis=0)
    return ens.times, q50, q25, q75


