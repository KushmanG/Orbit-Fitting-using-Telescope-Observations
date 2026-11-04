# This file gives the relative position of the celestial body w.r.t the telescope
from a_imports import *
from c_Position import *
from b_constants import R, c, au
'''
Intuition:
In my mind the concept is simple, we have:
Vector r1 from Sun to Earth                                                                             [Obtained from: erfa.epv00 library]
Vector r2 from Earth's center to Telescope (Considering the spinning of earth)                          [Obtained from: MPC Obscode database]
Vector r3 = r1+r2 from Sun to Telescope (wnv I say Sun/Earth or anything I mean it's center obviously)  [Vector addition ez]
Vector r4 from Sun to the celestial body (eros in this case)                                            [Obtained from: c_Position.py]

(Why did we choose eros? Cz I felt like it tbh, sounded cool, was totally arbitrary and my codebase is generalized anyway)

So now the vector of our interest, main output, is r5, the vector from the telescope to eros
Basic geometry, r3 + r5 = r4 (Imagine the vectors) so r5 = r4 - r3, r5 gives us the Radial Angle (RA) and Declination w.r.t earth's axis plane 

We also calculate TT from UTC using TT = UTC + 69.184s, TT is used by erfa.epv00 (Earth's coord lib) and Eros's position, UTC is the usual standard for time stamps

Also, we need to consider (I forgot this earlier causing issues) the LIGHT DELAY is significant in stellar distances so, r5 actually becomes:
    r5(t) = r4(t-T) - r1(t) - r2(t);

since what we observe at time t is actually the position of Eros at time t-T 

Now say r5 = (x y z) then RA and Dec have standard formulae in astrophysics being:
    RA = atan2(y,x)
    Dec = atan2(z, sqrt(x2 + y2))

So for formulation of r5 I am going to use:
1. r1 (Sun to Earth)
2. r2 (Earth to telescope)
3. r4 (Sun to Eros)
4. T (Light Delay)
5. jd_utc (Julian Day in UTC)
6. d,h and L (MPC Obscode)
'''

# Obtain r1
def R1(jd_utc):
    secondsPerDay = 60 * 60 * 24
    jd_tt = jd_utc + 69.184/secondsPerDay
    jd1 = np.floor(jd_tt)
    jd2 = jd_tt - jd1
    pvh, _ = erfa.epv00(jd1, jd2)
    return pvh["p"]

# r4 is js simulate(r0,v0,t0,t), lemme js rename it for my own sake
def R4(r0,v0,t0,t):
    return simulate(r0,v0,t0,t)

# Obtain and normalize telescope coords (r2) using MPC + Consider Earth's rotation
'''
MPC gives us:
    1. d (Perpendicular distance of Telescope from the axis of spin)
    2. h (Distance of Telescope from plane containing center and perpendicular to axis of spin)
    3. Longitude (L)
Another thing, d and h are in R_e units in the MPC database so we need D = dR_e and H = hR_e
This is all considering earth to be static in a conventional orientation

To this we add the angle Earth has rotated from this conventional orientation using a standard formula:
    A = 2π · (0.7790572732640 + 1.00273781191135448 · (jd_utc - 2451545.0)) [Earth's angle calc deliberately uses jd_utc, civil time]

Thus the net angle is, B = A + L (in radians)
Therefore r2 = (DcosB, DsinB, H)

As seen above only variable we need to input here is jd_utc
'''
def R2(jd_utc, d, h, L):
    r = R/au
    B = 2 * np.pi * (0.7790572732640 + 1.00273781191135448 * (jd_utc - 2451545.0)) + np.radians(L)
    x = d*r*np.cos(B)
    y = d*r*np.sin(B)
    z = h*r
    return np.stack([x,y,z], axis = 1)


def R5(jd_utc, r0, v0, t0, d, h, L):
    secondsPerDay = 60 * 60 * 24
    jd_tt = jd_utc + 69.184/secondsPerDay 

    r3 = R1(jd_utc) + R2(jd_utc,d,h,L)

    '''
    Calculating Light Delay (T) using the method mentioned in: 
    Explanatory Supplement to the Astronomical Almanac, 3rd edition (2013), edited by Sean E. Urban and P. Kenneth Seidelmann, University Science Books
    ISBN 9781891389856.
    
    Publisher's Page of the Referenced Book: uscibooks.aip.org
    '''
    T = 0
    for _ in range(0,3):
        r4 = R4(r0,v0,t0,jd_tt-T)
        r5 = r4-r3
        T = np.linalg.norm(r5, axis = 1)/c

    return r5

def result(jd_utc, r0, v0, t0, d, h, L):
    r5 = R5(jd_utc, r0, v0, t0, d, h, L)
    x = r5[:, 0]
    y = r5[:, 1]
    z = r5[:, 2]

    RA = np.remainder(np.arctan2(y,x), 2 * np.pi)
    Dec = np.arctan2(z, np.sqrt(x*x + y*y))

    return RA, Dec



