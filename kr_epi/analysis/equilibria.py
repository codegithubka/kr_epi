import numpy as np

def sis_equilibria(beta, gamma):
    R0 = beta/gamma
    if R0 <= 1: return 1.0, 0.0
    S, I = 1.0/R0, 1.0 - 1.0/R0
    return S, I

def sir_equilibria_closed(beta, gamma):
    # no births: only disease-free equilibrium
    return 1.0, 0.0, 0.0

def sirs_equilibria(beta, gamma, omega):
    R0 = beta/gamma
    if R0 <= 1: return 1.0, 0.0, 0.0
    S = 1.0/R0
    I = (omega/(omega+gamma))*(1.0 - 1.0/R0)
    R = (gamma/(omega+gamma))*(1.0 - 1.0/R0)
    return S, I, R
