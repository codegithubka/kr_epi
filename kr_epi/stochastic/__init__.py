# kr_epi/stochastic/__init__.py
from .direct import Direct
from .tau_leap import TauLeap

def make_engine(name: str, system, **kwargs):
    """
    Factory: name in {"direct", "tau"}.
    Example: make_engine("tau", sys, seed=1, adapt=True, eps=0.03)
    """
    name = (name or "").lower()
    if name in ("direct", "ssa", "gillespie"):
        return Direct(system, **kwargs)
    if name in ("tau", "tau-leap", "tauleap"):
        return TauLeap(system, **kwargs)
    raise ValueError(f"Unknown engine: {name}")
