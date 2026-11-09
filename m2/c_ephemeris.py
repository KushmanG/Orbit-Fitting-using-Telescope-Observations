# This file computes Earth and telescope positions and line-of-sight reference frames
#
# Helpers R1 and R2 are adapted from m1/d_relativePosition.py as per repo guidelines
# (Rule 4: self-contained m2 package without cross-folder imports).
from a_imports import *
from b_constants import R, c, au, TT_MINUS_UTC_SECONDS

'''
Heliocentric position of Earth's center (Sun -> Earth vector):
1. Converts Julian Date from UTC to Terrestrial Time (TT = UTC + 69.184s)
2. Splits JD into whole and fractional parts to retain sub-millisecond numerical precision
3. Calls erfa.epv00(jd1, jd2) to obtain high-precision heliocentric position in au (ICRF frame)
[Adapted from m1/d_relativePosition.py:R1]
'''
def R1(jd_utc):
    seconds_per_day = 86400.0
    jd_tt = jd_utc + TT_MINUS_UTC_SECONDS / seconds_per_day
    jd1 = np.floor(jd_tt)
    jd2 = jd_tt - jd1
    pvh, _ = erfa.epv00(jd1, jd2)
    return pvh["p"]


'''
Topocentric position of the telescope relative to Earth's center (Earth -> Telescope vector):
1. Uses MPC observatory parameters:
   - d: distance from Earth rotation axis (in Earth radii)
   - h: distance above Earth equatorial plane (in Earth radii)
   - L: East longitude of the observatory (in degrees)
2. Multiplies by R/au to convert Earth radii into astronomical units (au)
3. Calculates Greenwich Mean Sidereal rotation angle theta(t) using standard IAU formula
4. Total observatory angle is B = theta + radians(L)
5. Position vector in equatorial au is: (d*r*cos(B), d*r*sin(B), h*r)
[Adapted from m1/d_relativePosition.py:R2]
'''
def R2(jd_utc, d, h, L):
    r_earth_au = R / au
    theta = 2.0 * np.pi * (0.7790572732640 + 1.00273781191135448 * (jd_utc - 2451545.0))
    B = theta + np.radians(L)
    x = d * r_earth_au * np.cos(B)
    y = d * r_earth_au * np.sin(B)
    z = h * r_earth_au * np.ones_like(B)
    return np.stack([x, y, z], axis = 1)


'''
Total heliocentric position of the telescope at observation time:
R_total(t) = R1(t) [Sun -> Earth] + R2(t) [Earth -> Telescope]
'''
def telescope_position(jd_utc, d, h, L):
    return R1(jd_utc) + R2(jd_utc, d, h, L)


'''
Line-of-sight unit vector and orthonormal sky plane basis vectors:
For any observed sky position with Right Ascension (ra) and Declination (dec):
1. u: unit vector along the line of sight pointing from telescope toward the asteroid
      u = [cos(dec)*cos(ra), cos(dec)*sin(ra), sin(dec)]
2. e_ra: unit vector tangent to celestial sphere pointing in direction of increasing RA
      e_ra = [-sin(ra), cos(ra), 0]
3. e_dec: unit vector tangent to celestial sphere pointing in direction of increasing Dec
      e_dec = [-sin(dec)*cos(ra), -sin(dec)*sin(ra), cos(dec)]

Mathematical property:
{u, e_ra, e_dec} forms an orthonormal basis of R^3 for each observation:
   u . e_ra = 0,   u . e_dec = 0,   e_ra . e_dec = 0
   ||u|| = 1,      ||e_ra|| = 1,    ||e_dec|| = 1

Any vector rho = r - R pointing from the telescope to the asteroid can be decomposed as:
   rho = (rho . u) * u + (rho . e_ra) * e_ra + (rho . e_dec) * e_dec

The distance perpendicular to the observation ray is:
   rho_perp = (rho . e_ra) * e_ra + (rho . e_dec) * e_dec
Demanding rho_perp = 0 is equivalent to two scalar linear constraints:
   e_ra . (r - R) = 0
   e_dec . (r - R) = 0
'''
def line_of_sight_basis(ra, dec):
    cos_d = np.cos(dec)
    sin_d = np.sin(dec)
    cos_a = np.cos(ra)
    sin_a = np.sin(ra)

    # Line of sight unit vector
    u = np.stack([cos_d * cos_a, cos_d * sin_a, sin_d], axis = 1)

    # Unit vector along increasing Right Ascension
    e_ra = np.stack([-sin_a, cos_a, np.zeros_like(ra)], axis = 1)

    # Unit vector along increasing Declination
    e_dec = np.stack([-sin_d * cos_a, -sin_d * sin_a, cos_d], axis = 1)

    return u, e_ra, e_dec


if __name__ == "__main__":
    print("=" * 60)
    print("m2/c_ephemeris.py sanity check")
    print("=" * 60)
    
    # Test on arbitrary angles
    test_ra = np.array([0.5, 1.2, 3.4, 5.8])
    test_dec = np.array([-0.3, 0.1, 0.4, -0.8])
    u, e_ra, e_dec = line_of_sight_basis(test_ra, test_dec)

    norm_u = np.linalg.norm(u, axis = 1)
    norm_era = np.linalg.norm(e_ra, axis = 1)
    norm_edec = np.linalg.norm(e_dec, axis = 1)

    dot_u_era = np.sum(u * e_ra, axis = 1)
    dot_u_edec = np.sum(u * e_dec, axis = 1)
    dot_era_edec = np.sum(e_ra * e_dec, axis = 1)

    flag_norms = np.allclose(norm_u, 1.0) and np.allclose(norm_era, 1.0) and np.allclose(norm_edec, 1.0)
    flag_ortho = np.allclose(dot_u_era, 0.0) and np.allclose(dot_u_edec, 0.0) and np.allclose(dot_era_edec, 0.0)

    # Test telescope position on test epoch
    test_jd = 2460987.5
    r_tel = telescope_position(np.array([test_jd]), np.array([0.7]), np.array([0.7]), np.array([12.0]))
    flag_tel = (r_tel.shape == (1, 3)) and (0.95 < np.linalg.norm(r_tel[0]) < 1.05)

    print(f"Basis vectors normalized to 1.0 : {'PASS' if flag_norms else 'FAIL'}")
    print(f"Basis vectors mutually orthogonal: {'PASS' if flag_ortho else 'FAIL'}")
    print(f"Earth distance from Sun ~ 1 au  : {'PASS' if flag_tel else 'FAIL'}")
    print("=" * 60)
    print("ALL PASSED" if (flag_norms and flag_ortho and flag_tel) else "SOME FAILED")
    print("=" * 60)

# Program Completed

