# This file contains all third party library imports for the m2 (easy / convex) problem
#
# Libraries used:
#   numpy: array manipulation and linear algebra (least squares, eigenvalues, SVD)
#   scipy.integrate.solve_ivp: numerical integration of Newton ODEs for validation and comparison
#   erfa: IAU SOFA library wrapper for accurate Earth ephemeris (epv00)
#   matplotlib: plotting figures for validation, holdout predictions, and convexity verification
#   json, os, sys, time, pathlib: file I/O, timing, and path handling

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

# scipy
from scipy.integrate import solve_ivp

# pyerfa
import erfa

# matplotlib (configured for headless execution)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# standard library
import json
import sys
import time
from pathlib import Path

if __name__ == "__main__":
    print("=" * 60)
    print("m2/a_imports.py sanity check")
    print("=" * 60)
    print("numpy version     :", np.__version__)
    print("pyerfa available  : PASS")
    print("matplotlib backend:", matplotlib.get_backend())
    print("=" * 60)
    print("ALL PASSED")
    print("=" * 60)

# Program Completed
