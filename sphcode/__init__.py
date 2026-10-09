"""Spherical-code solver: repeated LogSumExp annealing with seven parameters,
run on one instance (N points on the unit sphere in R^D, maximize the
minimum angle).

Single-threaded BLAS is set here, before numpy is imported anywhere: the
episode has a wall-clock time limit, and a multi-threaded BLAS changes both the
speed and the last bits of every Gram matrix."""
import os as _os
import sys as _sys
import warnings as _w

_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
         "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
if "numpy" in _sys.modules and any(_os.environ.get(_v) != "1" for _v in _VARS):
    _w.warn("numpy was imported before sphcode and the BLAS thread variables are "
            "not set to 1: the BLAS may run multi-threaded, which makes the result "
            "depend on the thread count. Import sphcode first, or set these variables to 1.",
            RuntimeWarning, stacklevel=2)
for _v in _VARS:
    _os.environ.setdefault(_v, "1")

__version__ = "1.0.0"
