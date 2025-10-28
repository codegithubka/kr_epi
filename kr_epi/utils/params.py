def beta_for_R0_density(R0: float, gamma: float, mu: float, N0: float, extra: float=0.0):
    """extra = delta for mortality, or 0."""
    return R0 * (gamma + mu + extra) / N0