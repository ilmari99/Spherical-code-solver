"""The largest cosine, the starting configuration and the perturbation."""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.special import ndtri
from scipy.stats import qmc

SOBOL_M = 13            # the Sobol sample has 2^13 = 8192 points


def max_cosine(pts: np.ndarray) -> float:
    """Largest pairwise cosine (the quantity being minimized)."""
    dots = pts @ pts.T
    np.fill_diagonal(dots, -2.0)
    return float(np.max(dots))


def ang_deg(cos_val: float) -> float:
    return float(np.degrees(np.arccos(np.clip(cos_val, -1.0, 1.0))))


def isotropic_student_t(shape: tuple[int, ...], nu: float,
                        rng: np.random.Generator) -> np.ndarray:
    """Isotropic multivariate Student-t: a standard normal vector scaled by
    one chi^2 draw per vector, so directions are uniform on the sphere and
    only the magnitude is heavy-tailed."""
    z = rng.standard_normal(shape)
    w = rng.chisquare(nu, size=shape[:-1] + (1,))
    return z * np.sqrt(nu / np.maximum(w, 1e-300))


def perturb(pts: np.ndarray, sigma: float, nu: float,
            rng: np.random.Generator) -> np.ndarray:
    """Perturb every point of an (N, D) configuration on the unit sphere:
    an isotropic Student-t vector, projected onto the point's tangent plane
    and applied along the geodesic. `sigma` is the scale of the arc length,
    in radians."""
    if sigma <= 0:
        return pts.copy()
    v = isotropic_student_t(pts.shape, nu, rng)
    v = v - np.sum(v * pts, axis=-1, keepdims=True) * pts
    norms = np.linalg.norm(v, axis=-1, keepdims=True)
    theta = sigma * (norms / math.sqrt(max(pts.shape[-1] - 1, 1)))
    v_hat = np.where(norms > 1e-12, v / np.maximum(norms, 1e-12), 0.0)
    moved = np.cos(theta) * pts + np.sin(theta) * v_hat
    moved /= np.linalg.norm(moved, axis=-1, keepdims=True)
    return moved


def sobol_sample(d: int, m: int) -> np.ndarray:
    """2^m unscrambled Sobol points in [0,1]^D, mapped to the unit sphere
    through the Gaussian inverse CDF. The first two points, 0 and
    (1/2, ..., 1/2), map to -inf and to the origin and are dropped."""
    raw = qmc.Sobol(d=d, scramble=False).random_base2(m)
    g = ndtri(raw)
    g = g[np.isfinite(g).all(axis=1)]
    nrm = np.linalg.norm(g, axis=1)
    keep = nrm > 1e-9
    return g[keep] / nrm[keep, None]


def farthest_points(g: np.ndarray, n: int) -> np.ndarray:
    """Greedy farthest-point sampling of n points from `g`, starting from
    its first point. Uses no random numbers."""
    sel = np.empty(n, dtype=int)
    sel[0] = 0
    closeness = g @ g[0]
    for i in range(1, n):
        sel[i] = int(np.argmin(closeness))
        closeness = np.maximum(closeness, g @ g[sel[i]])
    pts = np.ascontiguousarray(g[sel])
    pts.setflags(write=False)
    return pts


@lru_cache(maxsize=256)
def sobol_fps(n: int, d: int) -> np.ndarray:
    """The Sobol sample on the sphere, reduced to N points by greedy
    farthest-point sampling. A function of (N, D) only. Read-only; callers
    copy. The starting configuration of an episode is this configuration
    after one L-BFGS-B call (episode.py)."""
    g = sobol_sample(d, SOBOL_M)
    if len(g) < n:
        raise ValueError(f"the Sobol sample has {len(g)} points, fewer than N = {n}")
    return farthest_points(g, n)
