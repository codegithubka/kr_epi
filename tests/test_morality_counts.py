import numpy as np
from kr_epi.models.ode_counts_mortality import (
    SIRDemographyCountsMortality, delta_from_cfr,
)

def _N(y): return y.sum(axis=0)

def test_frequency_R0_threshold_with_delta():
    N0 = 10000.0
    gamma, mu, v = 0.1, 1/70, 1/70     # keep v == mu so N would be  approx. constant without delta
    delta = 0.05                       # infection-induced death rate
    # Choose beta so R0 < 1: R0 = beta / (gamma+mu+delta)
    denom = gamma + mu + delta
    beta_sub = 0.9 * denom
    beta_sup = 1.3 * denom

    # Below threshold: infection dies out
    m1 = SIRDemographyCountsMortality(beta_sub, gamma, v, mu, delta, mixing="frequency")
    t, y = m1.integrate(dict(X=N0-10, Y=10.0, Z=0.0), (0, 365*15))
    assert y[1, -1] < 1e-3

    # Above threshold: endemic persists (Y not near zero)
    m2 = SIRDemographyCountsMortality(beta_sup, gamma, v, mu, delta, mixing="frequency")
    t2, y2 = m2.integrate(dict(X=N0-10, Y=10.0, Z=0.0), (0, 365*15))
    assert y2[1, -1] > 1.0  # some ongoing prevalence

def test_delta_reduces_population_when_v_eq_mu():
    N0 = 20000.0
    beta, gamma, mu, v = 0.24, 0.1, 1/70, 1/70
    delta = 0.1
    m = SIRDemographyCountsMortality(beta, gamma, v, mu, delta, mixing="frequency")
    t, y = m.integrate(dict(X=N0-10, Y=10.0, Z=0.0), (0, 365*10))
    # since dN/dt = (v - mu)N - delta Y and v=mu, N should decline relative to N0
    assert _N(y)[-1] < N0

def test_density_R0_uses_N0_and_delta_in_denominator():
    N0 = 8000.0
    gamma, mu, v, delta = 0.1, 0.01, 0.01, 0.05
    # R0 = beta*N0 / (gamma+mu+delta)  => choose sub/super critical betas
    denom = gamma + mu + delta
    beta_sub = 0.8 * denom / N0
    beta_sup = 1.2 * denom / N0

    m_sub = SIRDemographyCountsMortality(beta_sub, gamma, v, mu, delta, mixing="density")
    t, y = m_sub.integrate(dict(X=N0-5, Y=5.0, Z=0.0), (0, 365*8))
    assert y[1, -1] < 1e-2

    m_sup = SIRDemographyCountsMortality(beta_sup, gamma, v, mu, delta, mixing="density")
    t2, y2 = m_sup.integrate(dict(X=N0-5, Y=5.0, Z=0.0), (0, 365*8))
    assert y2[1, -1] > 1e-2

def test_delta_from_cfr_roundtrip():
    gamma, mu = 0.1, 0.01
    rho = 0.2
    delta = delta_from_cfr(rho, gamma, mu)
    # Check CFR implied by hazards (approx)
    implied = delta / (gamma + mu + delta)
    assert np.isclose(implied, rho, rtol=1e-12, atol=1e-12)
