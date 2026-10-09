"""The seven solver parameters: their search ranges, how the parameter files
store them, and the limits applied when an episode uses them.

    beta_min  start of the anneal
    r         span beta_max / beta_min
    s         steps (L-BFGS-B calls) from beta_min to beta_max
    sigma     perturbation scale, in radians
    nu        perturbation degrees of freedom
    rho       rollback: after a perturbation, beta returns to
              beta_max^(1-rho) * beta_min^rho
    eps       L-BFGS-B stopping tolerance (SciPy's ftol)

The parameter files store a parameter set as a vector u on the scale of the
parameter search: for the six parameters searched as logarithms,
    value = midpoint * exp(u), clipped to the search range,
and for the rollback, searched on a logit scale,
    rho = sigmoid(u).
The midpoint is the geometric midpoint of the search range (0.5 for rho), so
u = 0 is where the global fit starts.
"""
from __future__ import annotations

import math

PARAM_NAMES = ("beta_min", "r", "s", "sigma", "nu", "rho", "eps")

# Search ranges of the six parameters searched as logarithms.
RANGE = {"beta_min": (0.05, 200.0), "r": (1e2, 1e10), "s": (2, 10000),
         "sigma": (1e-6, 1.5), "nu": (0.5, 100.0), "eps": (1e-13, 1e-6)}
MIDPOINT = {k: math.sqrt(lo * hi) for k, (lo, hi) in RANGE.items()}

# Limits applied when an episode uses a parameter: every beta is limited to
# BETA_LIMITS, and sigma divided by N^(-1/(D-1)) to SIGMA_LIMITS. If beta_max
# is above the upper beta limit, beta never reaches beta_max and the episode
# never perturbs.
BETA_LIMITS = (0.05, 1e9)
SIGMA_LIMITS = (2e-6, 3.0)

_ONE_BELOW = math.nextafter(1.0, 0.0)   # largest double < 1
_TINY = math.nextafter(0.0, 1.0)        # smallest double > 0


def _rho_from_u(u_rho: float) -> float:
    rho = 1.0 / (1.0 + math.exp(-u_rho))
    # The paper's runs passed rho through its logit once more before use.
    # That round trip changes rho by one unit in the last place on 16
    # instances, and is repeated here to follow those runs bit for bit.
    rho = min(max(rho, _TINY), _ONE_BELOW)
    logit = math.log(rho / (1.0 - rho))
    try:
        return 1.0 / (1.0 + math.exp(-logit))
    except OverflowError:
        e = math.exp(logit)
        return e / (1.0 + e)


def params_from_u(u) -> dict:
    """The seven parameters from their stored vector u, plus the derived
    beta_max = r * beta_min and the factor r^(1/s) of one step of the anneal."""
    u = [float(v) for v in u]
    x = dict(zip(PARAM_NAMES, u))

    def value(k):
        lo, hi = RANGE[k]
        return min(max(MIDPOINT[k] * math.exp(x[k]), lo), hi)

    beta_min, r, s = value("beta_min"), value("r"), int(round(value("s")))
    beta_max = beta_min * r
    return {"beta_min": beta_min, "r": r, "s": s, "sigma": value("sigma"),
            "nu": value("nu"), "rho": _rho_from_u(x["rho"]), "eps": value("eps"),
            "beta_max": beta_max,
            "step_factor": (beta_max / beta_min) ** (1.0 / s)}


def u_from_value(name: str, value: float) -> float:
    """One parameter value -> its stored u (the inverse of params_from_u,
    before clipping)."""
    if name not in PARAM_NAMES:
        raise ValueError(f"unknown parameter {name!r}; the seven are {PARAM_NAMES}")
    value = float(value)
    if name == "rho":
        if not 0.0 < value < 1.0:
            raise ValueError("rho must be strictly between 0 and 1")
        return math.log(value / (1.0 - value))
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return math.log(value / MIDPOINT[name])


def as_used(value: float, lo: float, hi: float) -> float:
    """`value` limited to [lo, hi], as an episode uses it. The paper's runs
    stored the value on a logarithmic scale over [lo, hi] before using it.
    That round trip changes the last bits only, and is repeated here so that
    an episode follows those runs bit for bit."""
    v = min(max(value, lo), hi)
    a = 2.0 * math.log10(v / lo) / math.log10(hi / lo) - 1.0
    a = min(max(a, -1.0), 1.0)
    return 10.0 ** (math.log10(lo) + (a + 1.0) / 2.0 * (math.log10(hi) - math.log10(lo)))
