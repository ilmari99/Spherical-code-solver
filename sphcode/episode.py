"""One episode (Algorithm 1 of the paper): from the starting configuration,
L-BFGS-B steps on the LogSumExp surrogate at an increasing beta, and a
perturbation and a rollback at beta_max, until the time limit. Returns the
best configuration."""
from __future__ import annotations

import os
import time

import numpy as np
from scipy.optimize import minimize

from .geometry import ang_deg, max_cosine, perturb, sobol_fps
from .kernel import LogSumExp
from .parameters import BETA_LIMITS, RANGE, SIGMA_LIMITS, as_used, params_from_u

IMPROVE_TOL = 1e-12         # a new best must beat the old one by this in cosine
MAXCOR = 20                 # L-BFGS-B memory
MAXITER = 20_000            # iteration limit of every L-BFGS-B call
MAXFUN = 10 * MAXITER + 50
START_BETA = 100.0          # beta of the L-BFGS-B call of the starting configuration
START_EPS = 2.220446049250313e-09       # its stopping tolerance, SciPy's default


def fitting_seed(n: int, d: int) -> int:
    """The perturbation seed the fits used on instance (n, d)."""
    return int(np.random.default_rng((71, n, d)).integers(0, 2 ** 31))


def pin_to_core(cpu: int | None = None) -> int | None:
    """Pin this process to one CPU core, as the paper's episodes were. `cpu`
    None = the first core the process may run on. Returns the core, or None
    where the system has no such call (it exists on Linux)."""
    if not hasattr(os, "sched_setaffinity"):
        return None
    if cpu is None:
        cpu = min(os.sched_getaffinity(0))
    os.sched_setaffinity(0, {cpu})
    return cpu


def normalize(pts) -> np.ndarray:
    pts = np.ascontiguousarray(pts, dtype=float)
    return pts / np.linalg.norm(pts, axis=1, keepdims=True)


def run_episode(n: int, d: int, u, time_limit: float, seed: int,
                start: np.ndarray | None = None, max_steps: int | None = None,
                log=None) -> dict:
    """Run the solver on instance (n, d) with the parameter set `u` (the 7
    stored values).

    time_limit  wall-clock seconds. The starting configuration is not
                counted towards it.
    seed        the perturbation seed. Episodes with the same (n, d, u, seed,
                start) on the same machine follow the same trajectory until
                the time limit cuts an L-BFGS-B call.
    start       None = the paper's starting configuration: a Sobol sample,
                farthest-point sampling, and one L-BFGS-B call at beta = 100.
                An (n, d) array = these points instead, normalized; they get
                the same L-BFGS-B call, and the result is never worse than
                them.
    max_steps   stop after this many steps instead of at the time limit (for
                tests).
    log         called as log(i, event, cos, best_cos) after every step and
                perturbation.
    Returns dict(points, best_cos, best_ang, start_ang, input_ang, start_sec,
                 steps, perturbations, truncated, elapsed).
    """
    n, d = int(n), int(d)
    p = params_from_u(u)
    loss = LogSumExp(n, d)
    rng = np.random.default_rng(np.random.SeedSequence([n, d, int(seed)]))

    # the parameters with the two limits applied (parameters.py)
    beta_min, beta_max, rho = p["beta_min"], p["beta_max"], p["rho"]
    eps = as_used(p["eps"], *RANGE["eps"])
    nu = as_used(p["nu"], *RANGE["nu"])
    scale = n ** (-1.0 / (d - 1))
    sigma = as_used(p["sigma"] / scale, *SIGMA_LIMITS) * scale

    def lbfgs(pts, beta, eps, deadline=None):
        cut = []
        cb = None
        if deadline is not None:
            def cb(_xk):
                if time.perf_counter() >= deadline:
                    cut.append(1)
                    raise StopIteration
        res = minimize(loss.value_and_grad, pts.ravel(), args=(float(beta),),
                       method="L-BFGS-B", jac=True, callback=cb,
                       options={"gtol": 0.0, "ftol": eps, "maxcor": MAXCOR,
                                "maxiter": MAXITER, "maxfun": MAXFUN})
        out = res.x.reshape(n, d)
        return out / np.linalg.norm(out, axis=1, keepdims=True), bool(cut)

    # the starting configuration; its L-BFGS-B call is not counted towards the time limit
    pts = sobol_fps(n, d).copy() if start is None else normalize(start)
    if pts.shape != (n, d):
        raise ValueError(f"start has shape {pts.shape}, expected {(n, d)}")
    input_pts, input_cos = pts.copy(), max_cosine(pts)
    t = time.perf_counter()
    pts, _ = lbfgs(pts, START_BETA, START_EPS)
    start_sec = time.perf_counter() - t
    start_cos = max_cosine(pts)
    best_pts, best_cos = pts.copy(), start_cos
    if start is not None and input_cos < best_cos - IMPROVE_TOL:
        # A supplied configuration can already be tighter than beta = 100
        # resolves, and the L-BFGS-B call can then loosen it.
        best_pts, best_cos = input_pts, input_cos

    t0 = time.perf_counter()
    i = steps = perturbations = truncated = 0

    def finished(event) -> bool:
        """Keep the best configuration; True when the episode is over."""
        nonlocal best_pts, best_cos, i
        cos = max_cosine(pts)
        if cos < best_cos - IMPROVE_TOL:
            best_pts, best_cos = pts.copy(), cos
        i += 1
        if log is not None:
            log(i, event, cos, best_cos)
        return (time.perf_counter() - t0 >= time_limit
                or (max_steps is not None and steps >= max_steps))

    beta = as_used(beta_min, *BETA_LIMITS)
    while True:
        pts, cut = lbfgs(pts, beta, eps, t0 + time_limit)
        steps += 1
        truncated += int(cut)
        if finished({"perturb": False, "beta": beta, "eps": eps}):
            break
        if beta < beta_max * (1.0 - 1e-12):
            beta = as_used(min(beta * p["step_factor"], beta_max), *BETA_LIMITS)
        else:
            # perturb at beta_max, then roll back
            pts = perturb(pts, sigma, nu, rng)
            perturbations += 1
            if finished({"perturb": True, "sigma": sigma, "nu": nu}):
                break
            beta = as_used(beta * (beta_min / beta_max) ** rho, *BETA_LIMITS)
    return {"points": best_pts, "best_cos": best_cos,
            "best_ang": ang_deg(best_cos), "start_ang": ang_deg(start_cos),
            "input_ang": ang_deg(input_cos), "start_sec": start_sec,
            "steps": steps, "perturbations": perturbations,
            "truncated": truncated, "elapsed": time.perf_counter() - t0}
