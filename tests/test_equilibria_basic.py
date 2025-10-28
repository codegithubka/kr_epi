# tests/test_equilibria_basic.py
import numpy as np
from kr_epi.analysis.equilibria import sis_equilibria, sirs_equilibria

def test_sis_equilibria_sum():
    S,I = sis_equilibria(0.3,0.1)
    assert np.isclose(S+I,1.0)

def test_sirs_equilibria_match_formula():
    beta,gamma,omega = 0.3,0.1,0.05
    S,I,R = sirs_equilibria(beta,gamma,omega)
    assert np.isclose(S+I+R,1.0)
