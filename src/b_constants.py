# This file contains all important constants we use consistently through this program
from a_imports import *

GM_SUN = 2.9591220828559115e-04          # Sun's GM in au^3/day^2 (= k^2, Gaussian constant squared)
R = 6378.137                             # Earth's Radius in km since MPC obscodes return coords with R_e as unit, we need SI kilometers
c = 173.144632674                        # Speed of light in au/day
au = 149_597_870.7                       # IAU 2012 definition of the Astronomical Unit
# DONE