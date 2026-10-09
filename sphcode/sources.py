"""Where a parameter set comes from: the global parameters, the per-instance
parameters, the predictor, a JSON file, or explicit values."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .parameters import PARAM_NAMES, u_from_value

PARAMS_DIR = Path(__file__).resolve().parent / "params"
K_NEAREST = 16


def _load(name: str) -> dict:
    return json.loads((PARAMS_DIR / f"{name}.json").read_text())


def global_u() -> list[float]:
    return list(_load("global")["u"])


def per_instance_entry(n: int, d: int) -> dict:
    """The per-instance fit of (n, d): {"u": [...7], "seed": int, "best_ang": deg}."""
    for c in _load("per_instance")["instances"]:
        if c["n"] == n and c["d"] == d:
            return c
    raise KeyError(f"no per-instance parameters for N={n} D={d} "
                   f"(the fits cover 3<=D<=32, N<=1024, N>2D)")


def predictor_u(n: int, d: int) -> list[float]:
    """The predictor: the component-wise median of the per-instance u of the
    16 nearest other instances in standardized (log N, log D)."""
    fitted = _load("per_instance")["instances"]
    log_n = np.log(np.array([c["n"] for c in fitted], float))
    log_d = np.log(np.array([c["d"] for c in fitted], float))
    U = np.array([c["u"] for c in fitted], float)
    P = np.column_stack([(log_n - log_n.mean()) / log_n.std(),
                         (log_d - log_d.mean()) / log_d.std()])
    q = np.array([[(np.log(n) - log_n.mean()) / log_n.std(),
                   (np.log(d) - log_d.mean()) / log_d.std()]])
    _, idx = cKDTree(P).query(q, k=K_NEAREST + 1)
    idx = [i for i in idx[0] if (fitted[i]["n"], fitted[i]["d"]) != (n, d)][:K_NEAREST]
    return [float(v) for v in np.median(U[idx], axis=0)]


def resolve(source: str, n: int, d: int, overrides: dict | None = None):
    """(u, description). source: 'global' | 'per-instance' | 'predictor' |
    path to a JSON file with {"u": [...]} or {"params": {...}}. `overrides`
    replaces single parameters by value, {name: value}."""
    if source == "global":
        u, desc = global_u(), "global parameters"
    elif source == "per-instance":
        u, desc = list(per_instance_entry(n, d)["u"]), f"per-instance parameters of N={n} D={d}"
    elif source == "predictor":
        u, desc = predictor_u(n, d), f"predictor (median of the {K_NEAREST} nearest instances)"
    else:
        f = json.loads(Path(source).read_text())
        if "u" in f:
            u = [float(v) for v in f["u"]]
        elif "params" in f:
            u = [u_from_value(k, f["params"][k]) for k in PARAM_NAMES]
        else:
            raise ValueError(f"{source}: expected a JSON object with 'u' (7 values) or 'params'")
        desc = f"file {source}"
    if len(u) != 7:
        raise ValueError(f"{desc}: expected 7 values, got {len(u)}")
    if overrides:
        for k, v in overrides.items():
            u[PARAM_NAMES.index(k)] = u_from_value(k, v)
        desc += " with " + ", ".join(f"{k}={v}" for k, v in overrides.items())
    return u, desc
