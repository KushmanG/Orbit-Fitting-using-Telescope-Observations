# This file defines standard physical and astronomical constants used in m2
from a_imports import *

# Gravitational parameter of the Sun in astronomical units: au^3 / day^2 (= k^2, Gaussian gravitational constant)
GM_SUN = 2.9591220828559115e-04

# Earth equatorial radius in kilometers (IAU standard)
R = 6378.137

# Speed of light in au / day
c = 173.144632674

# Definition of the Astronomical Unit (au) in kilometers (IAU 2012 resolution B2)
au = 149_597_870.7

# Conversion factor: number of arcseconds in 1 radian (180 * 3600 / pi)
ARCSEC = 206264.806

# Standard epoch for the Eros orbit fit: 2025-11-08 00:00:00 TT (Julian Date)
T0 = 2460987.5

# Time offset between Terrestrial Time (TT) and Coordinated Universal Time (UTC) in seconds
TT_MINUS_UTC_SECONDS = 69.184

if __name__ == "__main__":
    print("=" * 60)
    print("m2/b_constants.py sanity check")
    print("=" * 60)
    print(f"GM_SUN               : {GM_SUN:.16e} au^3/day^2")
    print(f"Speed of light (c)   : {c:.8f} au/day")
    print(f"1 AU                 : {au:.1f} km")
    print(f"Arcseconds per rad   : {ARCSEC:.3f}")
    print(f"Fit epoch (T0)       : {T0} JD (TT)")
    print("=" * 60)
    flags = [
        GM_SUN > 0,
        c > 170,
        au > 1e8,
        abs(ARCSEC - 206264.8) < 1.0,
        T0 == 2460987.5
    ]
    print("ALL PASSED" if all(flags) else "SOME FAILED")
    print("=" * 60)

# Program Completed

