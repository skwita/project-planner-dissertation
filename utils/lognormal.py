"""
Shared lognormal parameter helpers.

Single source of truth for converting between the user-facing
parameterisation (arithmetic mean μ, stddev σ) and the two scipy
conventions used throughout the project.

scipy.stats.lognorm convention
-------------------------------
If  X ~ LogNormal  then  ln(X) ~ Normal(mu_log, sigma_log²),  and scipy
parameterises this as  lognorm(s=sigma_log, scale=exp(mu_log)).

Conversion from arithmetic (μ, σ)
-----------------------------------
    a         = 1 + (σ/μ)²
    sigma_log = sqrt(ln(a))          # shape  — same as scipy's ``s``
    mu_log    = ln(μ) − sigma_log²/2 # location in log-space
    scale     = exp(mu_log)          # scipy's ``scale`` parameter

Both representations are kept because:
  - ``lognorm_scipy_params`` → direct use in lognorm.ppf / lognorm.rvs
  - ``mu_log`` / ``sigma_log`` → needed for Bayesian arithmetic in
    bayes_rescheduler (computing z = ln(actual) − mu_log)
"""

from __future__ import annotations

import numpy as np
from scipy.stats import lognorm as _lognorm


def lognorm_scipy_params(mean: float, stddev: float) -> tuple[float, float]:
    """
    Return ``(s, scale)`` for direct use in ``scipy.stats.lognorm``.

    Args:
        mean:   Arithmetic mean of the lognormal distribution (> 0).
        stddev: Arithmetic standard deviation (> 0).

    Returns:
        ``(s, scale)`` where ``s = sigma_log`` and ``scale = exp(mu_log)``.

    Example::

        s, scale = lognorm_scipy_params(6.0, 1.5)
        p90 = lognorm.ppf(0.9, s=s, scale=scale)
        sample = lognorm.rvs(s=s, scale=scale, random_state=rng)
    """
    a         = 1.0 + (stddev / mean) ** 2
    sigma_log = float(np.sqrt(np.log(a)))
    mu_log    = float(np.log(mean) - 0.5 * sigma_log ** 2)
    return sigma_log, float(np.exp(mu_log))


def lognorm_log_params(mean: float, stddev: float) -> tuple[float, float]:
    """
    Return ``(mu_log, sigma_log)`` — the Normal parameters of ln(X).

    Useful for Bayesian arithmetic where you work in log-space directly
    (e.g. computing residuals  z = ln(actual) − mu_log).

    Args:
        mean:   Arithmetic mean (> 0).
        stddev: Arithmetic standard deviation (> 0).

    Returns:
        ``(mu_log, sigma_log)`` where ``sigma_log`` is the shape and
        ``exp(mu_log)`` is scipy's ``scale``.
    """
    a         = 1.0 + (stddev / mean) ** 2
    sigma_log = float(np.sqrt(np.log(a)))
    mu_log    = float(np.log(mean) - 0.5 * sigma_log ** 2)
    return mu_log, sigma_log


def lognorm_ppf(percentile: float, mean: float, stddev: float) -> float:
    """Return the ``percentile``-quantile of LogNormal(mean, stddev)."""
    s, scale = lognorm_scipy_params(mean, stddev)
    return float(_lognorm.ppf(percentile, s=s, scale=scale))


def lognorm_rvs(mean: float, stddev: float,
                rng: np.random.Generator) -> float:
    """Draw one sample from LogNormal(mean, stddev) using ``rng``."""
    s, scale = lognorm_scipy_params(mean, stddev)
    return float(_lognorm.rvs(s=s, scale=scale, random_state=rng))
