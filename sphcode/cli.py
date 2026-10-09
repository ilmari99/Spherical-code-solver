"""Run the spherical-code solver on one instance, or print its parameters."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from . import __version__
from .episode import fitting_seed, pin_to_core, run_episode
from .geometry import ang_deg
from .parameters import params_from_u
from .sources import resolve


def _parse_set(items):
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--set expects name=value, got {it!r}")
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _load_start(path: str, n: int, d: int):
    p = Path(path)
    pts = np.load(p) if p.suffix == ".npy" else np.loadtxt(p, delimiter=",", ndmin=2)
    if pts.shape != (n, d):
        raise SystemExit(f"{path}: shape {pts.shape}, expected ({n}, {d})")
    return pts


def _print_params(p: dict, desc: str, u):
    print(f"parameters: {desc}")
    print("  u = [" + ", ".join(f"{v:.6f}" for v in u) + "]")
    print(f"  beta_min {p['beta_min']:.6g}  r {p['r']:.6g}  s {p['s']}  "
          f"sigma {p['sigma']:.6g}  nu {p['nu']:.6g}  rho {p['rho']:.6g}  eps {p['eps']:.3g}")


def _print_step(i, event, cos, best_cos):
    what = (f"perturb  sigma {event['sigma']:.3e} nu {event['nu']:.3g}" if event["perturb"]
            else f"step     beta {event['beta']:.4g} eps {event['eps']:.2e}")
    print(f"  {i:5d}  {what}  angle {ang_deg(cos):.6f}  best {ang_deg(best_cos):.6f}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="sphcode", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("solve", "params"):
        sp = sub.add_parser(name)
        sp.add_argument("N", type=int)
        sp.add_argument("D", type=int)
        sp.add_argument("--params", default="per-instance",
                       help="global | per-instance | predictor | FILE.json (default per-instance)")
        sp.add_argument("--set", action="append", metavar="NAME=VALUE",
                       help="override one of beta_min r s sigma nu rho eps")
        if name == "solve":
            sp.add_argument("--time-limit", type=float, default=60.0, metavar="SECONDS",
                           help="time limit of the episode in wall-clock seconds; the starting "
                                "configuration is not counted towards it (default 60)")
            sp.add_argument("--seed", default="fit",
                            help="perturbation seed: an integer, or 'fit' = the seed of the "
                                 "paper's fits on this instance (default)")
            sp.add_argument("--cpu", type=int, metavar="CORE",
                            help="pin the episode to this CPU core (default: the first core "
                                 "the process may run on; Linux only)")
            sp.add_argument("--start", metavar="FILE",
                           help="your own configuration to start from: .npy, or text with one "
                                "point per line and comma-separated coordinates (default: the "
                                "starting configuration of the paper)")
            sp.add_argument("--out", metavar="FILE",
                           help="write the best configuration: .npy, or text with one point per "
                                "line and comma-separated coordinates")
            sp.add_argument("--trace", action="store_true",
                           help="print every step and perturbation")
    a = ap.parse_args(argv)

    n, d = a.N, a.D
    if d < 2 or n < d + 1:
        raise SystemExit("need D >= 2 and N >= D + 1")
    try:
        u, desc = resolve(a.params, n, d, _parse_set(a.set))
    except KeyError as e:
        raise SystemExit(f"{e.args[0]}. Use --params global or --params predictor, "
                         f"which work for any instance.")
    except (ValueError, OSError) as e:
        raise SystemExit(str(e))
    _print_params(params_from_u(u), desc, u)
    if a.cmd == "params":
        return
    if a.seed != "fit" and not a.seed.lstrip("-").isdigit():
        raise SystemExit(f"--seed expects an integer or 'fit', got {a.seed!r}")
    seed = fitting_seed(n, d) if a.seed == "fit" else int(a.seed)
    start = _load_start(a.start, n, d) if a.start else None
    core = pin_to_core(a.cpu)
    print(f"instance N={n} D={d}, time limit {a.time_limit:g} s, perturbation seed {seed}, "
          f"starting configuration {'from file ' + a.start if a.start else 'of the paper'}, "
          f"{'not pinned' if core is None else f'pinned to core {core}'}", flush=True)

    res = run_episode(n, d, u, a.time_limit, seed, start=start, log=_print_step if a.trace else None)
    print(f"starting configuration {res['start_ang']:.6f} deg ({res['input_ang']:.6f} deg before its "
          f"L-BFGS-B call, which took {res['start_sec']:.1f} s and is not counted)")
    print(f"best angle  {res['best_ang']:.9f} deg  (largest cosine {res['best_cos']:.15f})")
    print(f"{res['steps']} steps, {res['perturbations']} perturbations, "
          f"{res['truncated']} steps cut by the time limit, "
          f"{res['elapsed']:.1f} s")
    if a.out:
        if Path(a.out).suffix == ".npy":
            np.save(a.out, res["points"])
        else:
            np.savetxt(a.out, res["points"], fmt="%.17g", delimiter=",")
        print(f"wrote {a.out}")


if __name__ == "__main__":
    sys.exit(main())
