import numpy as np
from kr_epi.models.reactions import sir_demography_counts
from kr_epi.stochastic.direct import Direct
from kr_epi.sweeps.runners import run_ensemble, extinction_probability

def test_direct_runs_and_conserves_nonnegativity():
    sys = sir_demography_counts(beta=0.22, gamma=0.1, v=1/70, mu=1/70, mixing="frequency")
    sim = Direct(sys, seed=1)
    N0 = 5000
    t, X = sim.run({"X": N0-10, "Y": 10, "Z": 0}, t_max=365*2)
    assert (X >= 0).all()
    assert t[-1] <= 365*2 + 1e-9

def test_extinction_probability_subcritical_density():
    # density mixing with R0 < 1 at N0
    N0 = 2000
    gamma, mu = 0.1, 1/70
    beta = 0.00002  # makes R0 = beta*N0/(gamma+mu) < 1
    sys = sir_demography_counts(beta=beta, gamma=gamma, v=mu, mu=mu, mixing="density")
    sim = Direct(sys, seed=2)
    ens = run_ensemble(sim, {"X": N0-5, "Y": 5, "Z": 0}, t_max=365*5, n_runs=40, seed0=10,
                       record_times=np.linspace(0, 365*5, 300))
    pext = extinction_probability(ens, "Y", threshold=0.5)
    assert pext > 0.5
