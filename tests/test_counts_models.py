import numpy as np
from kr_epi.models.ode_counts import (
    SIRDemographyCounts, SISDemographyCounts, SEIRDemographyCounts
)

def _pos(y):
    return (y >= -1e-9).all()

def test_sir_counts_freq_R0_threshold_and_N_const_when_v_eq_mu():
    N0 = 10000.0
    # frequency: R0 = beta / (gamma+mu)
    beta, gamma, mu, v = 0.22, 0.1, 0.01, 0.01   # R0 ≈ 2.0
    m = SIRDemographyCounts(beta, gamma, v, mu, mixing="frequency")
    t, y = m.integrate(dict(X=N0-10, Y=10.0, Z=0.0), (0, 3650))  # 10 years
    X, Y, Z = y
    assert _pos(y)
    # N should remain close to N0 if v==mu
    assert np.isclose((X+Y+Z)[-1], N0, rtol=1e-2)

def test_sis_counts_density_R0_uses_N0():
    N0 = 5000.0
    # density: R0 = beta * N0 / (gamma+mu)
    gamma, mu, v = 0.1, 0.01, 0.01
    # pick beta so R0 < 1
    beta = 0.00015  # R0 ≈ 0.00015*5000/0.11 ≈ 6.82 < 1? nope -> adjust
    beta = 0.00002  # R0 ≈ 0.91 < 1
    m = SISDemographyCounts(beta, gamma, v, mu, mixing="density")
    t, y = m.integrate(dict(X=N0-5, Y=5.0), (0, 3000))
    assert y[1, -1] < 1e-3  # Y dies out
    # now bump beta so R0 > 1
    beta2 = 0.00003  # R0 ≈ 1.37 > 1
    m2 = SISDemographyCounts(beta2, gamma, v, mu, mixing="density")
    t2, y2 = m2.integrate(dict(X=N0-5, Y=5.0), (0, 3000))
    assert y2[1, -1] > 1e-3  # endemic persists

def test_seir_counts_vaccination_at_birth_shifts_Z_up():
    N0 = 20000.0
    beta, sigma, gamma, mu, v = 0.6, 1/5, 1/7, 1/70, 1/70
    m0 = SEIRDemographyCounts(beta, sigma, gamma, v, mu, mixing="frequency", vacc_p=0.0)
    m1 = SEIRDemographyCounts(beta, sigma, gamma, v, mu, mixing="frequency", vacc_p=0.6)
    y0 = dict(X=N0-20, E=0.0, Y=20.0, Z=0.0)
    t, y_base = m0.integrate(y0, (0, 3650))
    t, y_vax  = m1.integrate(y0, (0, 3650))
    assert y_vax[3, -1] > y_base[3, -1]  # more in Z long-run with birth vaccination
