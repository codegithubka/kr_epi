import numpy as np
from kr_epi.models.ode import SI, SIS, SIR, SIRS, SEIR

def _cons_total(y, tol=1e-6):
    return np.allclose(np.sum(y, axis=0), 1.0, atol=tol)

def test_si_monotone_and_conservation():
    m = SI(beta=0.5)
    t, y = m.integrate(dict(S=0.99, I=0.01), (0, 40))
    S, I = y
    assert _cons_total(y)
    assert S[-1] < S[0]
    assert I[-1] > I[0]

def test_sis_equilibrium_threshold():
    # R0 = beta/gamma
    m1 = SIS(beta=0.09, gamma=0.1)     # R0 < 1 -> disease-free
    m2 = SIS(beta=0.3,  gamma=0.1)     # R0 > 1 -> endemic
    y0 = dict(S=0.99, I=0.01)

    t, y = m1.integrate(y0, (0, 800))
    assert _cons_total(y)
    assert y[1, -1] < 5e-6           # I* ~ 0

    t, y = m2.integrate(y0, (0, 800))
    I_star = 1.0 - 1.0/(m2.params.beta/m2.params.gamma)  # I* = 1 - 1/R0
    assert np.isclose(y[1, -1], I_star, rtol=0.1)

def test_sir_closed_population_properties():
    m = SIR(beta=0.3, gamma=0.1)
    t, y = m.integrate(dict(S=0.99, I=0.01, R=0.0), (0, 200))
    S, I, R = y
    assert _cons_total(y)
    assert np.all(np.diff(S) <= 1e-10)  # S nonincreasing (allow tiny num. jitter)
    assert np.all(np.diff(R) >= -1e-10) # R nondecreasing
    assert I[-1] < 1e-4                 # epidemic burns out eventually

def test_sirs_waning_equilibrium_shares():
    beta, gamma, omega = 0.3, 0.1, 0.05
    m = SIRS(beta=beta, gamma=gamma, omega=omega)
    t, y = m.integrate(dict(S=0.9, I=0.1, R=0.0), (0, 1500))
    S, I, R = y
    assert _cons_total(y)
    R0 = beta/gamma
    S_star = 1.0/R0
    I_star = (omega/(omega+gamma)) * (1.0 - 1.0/R0)
    R_star = (gamma/(omega+gamma)) * (1.0 - 1.0/R0)
    assert np.isclose(S[-1], S_star, rtol=0.15)
    assert np.isclose(I[-1], I_star, rtol=0.2)
    assert np.isclose(R[-1], R_star, rtol=0.2)

def test_seir_conservation_and_latent_dynamics():
    m = SEIR(beta=0.3, sigma=0.2, gamma=0.1)
    t, y = m.integrate(dict(S=0.99, E=0.0, I=0.01, R=0.0), (0, 200))
    S, E, I, R = y
    assert _cons_total(y)
    assert E.max() > 0                  # latent class activates
    assert I[-1] < 1e-4                 # burns out in closed population
