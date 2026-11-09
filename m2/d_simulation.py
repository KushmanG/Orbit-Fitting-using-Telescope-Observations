# This file provides numerical ODE simulation and exact astrometric prediction for m2
#
# Rule 3 compliant: m1 and m2 never import each other. This simulation engine
# evaluates orbits using exact Newtonian 2-body physics + light-time iteration.
from a_imports import *
from b_constants import GM_SUN, c, ARCSEC, TT_MINUS_UTC_SECONDS
from c_ephemeris import telescope_position

'''
Newtonian two-body gravitational acceleration:
Phase space state vector y = [x, y, z, vx, vy, vz]
dr/dt = v
dv/dt = -GM_SUN * r / ||r||^3
'''
def gravity(t, y):
    r = y[:3]
    v = y[3:]
    r_norm = np.linalg.norm(r)
    drdt = v
    dvdt = -GM_SUN * r / (r_norm**3)
    return np.concatenate([drdt, dvdt])


'''
Numerical integration of the orbit from initial state (r0, v0) at epoch t0:
1. Splits target times t into t < t0 (backward integration) and t >= t0 (forward integration)
2. Uses solve_ivp with tight tolerances (rtol = 1e-12, atol = 1e-14)
3. Evaluates dense solution at requested times t
4. Recombines and returns positions as shape (len(t), 3)
'''
def simulate(r0, v0, t0, t):
    state0 = np.concatenate([r0, v0])

    t_before = [ti for ti in t if ti < t0]
    t_after = [ti for ti in t if ti >= t0]

    states = []
    if len(t_before) > 0:
        t_min = min(t_before)
        sol_before = solve_ivp(gravity, (t0, t_min), state0, dense_output = True, rtol = 1e-12, atol = 1e-14)
        states.append(sol_before.sol(t_before))

    if len(t_after) > 0:
        t_max = max(t_after)
        sol_after = solve_ivp(gravity, (t0, t_max), state0, dense_output = True, rtol = 1e-12, atol = 1e-14)
        states.append(sol_after.sol(t_after))

    if len(states) == 2:
        state_all = np.concatenate(states, axis = 1)
    else:
        state_all = states[0]

    # Return positions transposed: shape (N, 3)
    return state_all[:3].T


'''
Astrometric prediction of Right Ascension and Declination:
1. Computes heliocentric telescope position R(t) = R1(t) + R2(t)
2. Accounts for planetary aberration / light-travel time delay:
   r5(t) = r4(t - tau) - R(t), with tau = ||r5|| / c (3 fixed-point iterations)
3. Computes spherical coordinates on ICRF equatorial frame:
   RA  = atan2(y, x) mod 2pi
   Dec = atan2(z, sqrt(x^2 + y^2))
'''
def predict_exact(r0, v0, t0, jd_utc, d, h, L, light_delay = True):
    jd_tt = jd_utc + TT_MINUS_UTC_SECONDS / 86400.0
    R_tel = telescope_position(jd_utc, d, h, L)

    if light_delay:
        tau = np.zeros(len(jd_utc))
        for _ in range(3):
            t_retarded = jd_tt - tau
            r_eros = simulate(r0, v0, t0, t_retarded)
            r5 = r_eros - R_tel
            tau = np.linalg.norm(r5, axis = 1) / c
    else:
        r_eros = simulate(r0, v0, t0, jd_tt)
        r5 = r_eros - R_tel

    x = r5[:, 0]
    y = r5[:, 1]
    z = r5[:, 2]

    ra_pred = np.remainder(np.arctan2(y, x), 2.0 * np.pi)
    dec_pred = np.arctan2(z, np.sqrt(x**2 + y**2))
    return ra_pred, dec_pred


'''
Angular miss between measured angles and predicted angles, in arcseconds:
1. Wraps Right Ascension difference into [-pi, pi] to properly handle 0/2pi circular wrap
2. Projects dRA onto tangent plane by multiplying by cos(Dec) (great-circle geometry)
3. Multiplies by ARCSEC (206,264.806) to output arcseconds
'''
def misses(ra_meas, dec_meas, ra_pred, dec_pred):
    dra = (np.remainder(ra_meas - ra_pred + np.pi, 2.0 * np.pi) - np.pi) * np.cos(dec_meas)
    ddec = dec_meas - dec_pred
    return np.stack([dra, ddec], axis = 1) * ARCSEC


'''
Root-mean-square (RMS) of angular residuals in arcseconds:
RMS = sqrt(mean(dra_arcsec^2 + ddec_arcsec^2))
'''
def rms(m):
    return float(np.sqrt(np.mean(m**2)))


if __name__ == "__main__":
    print("=" * 60)
    print("m2/d_simulation.py sanity check")
    print("=" * 60)

    # Sanity check with a circular test orbit
    test_r0 = np.array([1.2, 0.0, 0.0])
    test_v0 = np.array([0.0, np.sqrt(GM_SUN / 1.2), 0.0])
    test_t0 = 2460987.5
    test_times = np.array([test_t0 - 10.0, test_t0, test_t0 + 10.0])

    pos = simulate(test_r0, test_v0, test_t0, test_times)
    flag_shape = (pos.shape == (3, 3))
    flag_t0 = np.allclose(pos[1], test_r0, atol = 1e-10)
    flag_radius = np.allclose(np.linalg.norm(pos, axis = 1), 1.2, atol = 1e-4)

    print(f"Simulation output shape (3, 3)     : {'PASS' if flag_shape else 'FAIL'}")
    print(f"Simulation reproduces r0 at t0     : {'PASS' if flag_t0 else 'FAIL'}")
    print(f"Circular orbit radius conserved    : {'PASS' if flag_radius else 'FAIL'}")
    print("=" * 60)
    print("ALL PASSED" if (flag_shape and flag_t0 and flag_radius) else "SOME FAILED")
    print("=" * 60)

# Program Completed

