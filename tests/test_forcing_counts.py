import numpy as np
from kr_epi.models.ode_counts import SIRDemographyCounts
from kr_epi.forcing.sinusoid import BetaSinusoid
from kr_epi.forcing.term_time import TermTime

def test_sinusoid_amp_zero_equals_constant_beta_counts():
    N0 = 5000.0
    beta0, gamma, mu, v = 0.22, 0.1, 1/70, 1/70
    m_const = SIRDemographyCounts(beta=beta0, gamma=gamma, v=v, mu=mu, mixing="frequency")
    m_forced = SIRDemographyCounts(beta=0.0,   gamma=gamma, v=v, mu=mu, mixing="frequency")
    y0 = dict(X=N0-5, Y=5.0, Z=0.0)
    t_span = (0, 365*3)

    t1, y1 = m_const.integrate(y0, t_span)
    t2, y2 = m_forced.integrate(y0, t_span, beta_fn=BetaSinusoid(beta0=beta0, amp=0.0))
    # allow small numerical drift
    assert np.allclose(y1, y2, rtol=1e-5, atol=1e-7)

def test_term_time_amp_zero_equals_constant_beta_counts():
    N0 = 8000.0
    beta0, gamma, mu, v = 0.3, 0.1, 1/70, 1/70
    m_const = SIRDemographyCounts(beta=beta0, gamma=gamma, v=v, mu=mu, mixing="frequency")
    m_forced = SIRDemographyCounts(beta=0.0,   gamma=gamma, v=v, mu=mu, mixing="frequency")
    y0 = dict(X=N0-10, Y=10.0, Z=0.0)
    tt = TermTime(beta0=beta0, amp=0.0, period=365,
                  term_windows=[(0, 45), (70, 120)], phase=0.0)
    t1, y1 = m_const.integrate(y0, (0, 365*2))
    t2, y2 = m_forced.integrate(y0, (0, 365*2), beta_fn=tt)
    assert np.allclose(y1, y2, rtol=1e-5, atol=1e-7)

def test_forcing_changes_trajectory_when_amp_positive():
    N0 = 12000.0
    beta0, gamma, mu, v = 0.22, 0.1, 1/70, 1/70
    m = SIRDemographyCounts(beta=0.0, gamma=gamma, v=v, mu=mu, mixing="frequency")
    y0 = dict(X=N0-10, Y=10.0, Z=0.0)
    t_span = (0, 365*5)

    base = BetaSinusoid(beta0=beta0, amp=0.0)
    forced = BetaSinusoid(beta0=beta0, amp=0.3)

    t0, y0c = m.integrate(y0, t_span, beta_fn=base)
    t1, y1c = m.integrate(y0, t_span, beta_fn=forced)
    # trajectories should differ if amp>0
    assert not np.allclose(y0c, y1c, rtol=1e-6, atol=1e-8)

def test_term_time_vs_sinusoid_not_identical_but_same_scale():
    # Just a sanity: forcing types differ → different outputs,
    # but average beta over a year is similar when amp equal.
    N0 = 10000.0
    beta0, gamma, mu, v = 0.22, 0.1, 1/70, 1/70
    y0 = dict(X=N0-10, Y=10.0, Z=0.0)
    m = SIRDemographyCounts(beta=0.0, gamma=gamma, v=v, mu=mu, mixing="frequency")

    sin = BetaSinusoid(beta0=beta0, amp=0.2, period=365)
    tt = TermTime(beta0=beta0, amp=0.2, period=365,
                  term_windows=[(0, 45), (70, 120), (150, 200), (230, 285), (310, 340)], phase=0.0)
    t_s, y_s = m.integrate(y0, (0, 365*5), beta_fn=sin)
    t_t, y_t = m.integrate(y0, (0, 365*5), beta_fn=tt)

    # Different shapes
    assert not np.allclose(y_s, y_t, rtol=1e-6, atol=1e-8)

    # Average beta over one period is the same for symmetric +/- amp
    # (not asserted on trajectories; just document rationale)
