# This is the "Hard" solution
from a_imports import *
from b_constants import ARCSEC, GM_SUN

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process_data import get_arrays, FIT_START, FIT_END 
from d_relativePosition import result, R1, R2, R4
# Firstly we load all the data
start = FIT_START
end = FIT_END
data = get_arrays(start,end)
'''
Data is of the format: {"jd_utc": [..], "ra": [..], "dec": [..], "d": [..], "h": [..], "L": [..], "codes": [..]}
We load them into indivual ndarrays
'''
jd_utc = data["jd_utc"]
ra = data["ra"]
dec = data["dec"]
d = data["d"]
h = data["h"]
L = data["L"]
obscodes = data["codes"]
t0 = 2460987.5

# y is again the phase vector [x,y,z,vx,vy,vz]
def miss(y):
    RA_pred, Dec_pred = result(jd_utc, y[:3], y[3:], t0, d, h, L)
    RA_measured= ra
    Dec_measured = dec

    Δ_RA = np.remainder(RA_measured - RA_pred + np.pi, 2 * np.pi) - np.pi # Standard angle wrapping code [-pi,pi]
    Δ_Dec = Dec_measured - Dec_pred
    COS = np.cos(dec)
    return np.concatenate([Δ_RA * COS, Δ_Dec])*ARCSEC # Since Δ_RA, Δ_Dec are in rad

# First prediction
'''
Standard algorithm to make a decent first prediction of a planet's position.
Reference:
    Curtis, Orbital Mechanics for Engineering Students, Algorithm 5.5 
    (“Gauss's method of preliminary orbit determination”), Section 5.10

Input: ANY 3 observation indices
Output:

'''
def gauss(i1, i2, i3):
    idx = [i1, i2, i3]
    jd = jd_utc[idx]
    t = jd + 69.184 / 86400
    u = np.stack([np.cos(dec[idx]) * np.cos(ra[idx]),
                  np.cos(dec[idx]) * np.sin(ra[idx]),
                  np.sin(dec[idx])], axis = 1)
    R = R1(jd) + R2(jd, d[idx], h[idx], L[idx])
    mu = GM_SUN

    tau1, tau3 = t[0] - t[1], t[2] - t[1]
    tau = tau3 - tau1
    p = [np.cross(u[1], u[2]), np.cross(u[0], u[2]), np.cross(u[0], u[1])]
    D0 = u[0] @ p[0]
    D = np.array([[R[i] @ p[j] for j in range(3)] for i in range(3)])

    A = (-D[0, 1] * tau3 / tau + D[1, 1] + D[2, 1] * tau1 / tau) / D0
    B = (D[0, 1] * (tau3**2 - tau**2) * tau3 / tau + D[2, 1] * (tau**2 - tau1**2) * tau1 / tau) / (6 * D0)
    E = R[1] @ u[1]
    a = -(A**2 + 2 * A * E + R[1] @ R[1])
    b = -2 * mu * B * (A + E)
    c = -mu**2 * B**2

    candidates = []
    for r2 in np.roots([1, 0, a, 0, 0, b, 0, 0, c]):
        if abs(r2.imag) > 1e-10 or r2.real <= 0:
            continue
        r2 = r2.real
        rho1 = ((6 * (D[2, 0] * tau1 / tau3 + D[1, 0] * tau / tau3) * r2**3 + mu * D[2, 0] * (tau**2 - tau1**2) * tau1 / tau3)
                / (6 * r2**3 + mu * (tau**2 - tau3**2)) - D[0, 0]) / D0
        rho2 = A + mu * B / r2**3
        rho3 = ((6 * (D[0, 2] * tau3 / tau1 - D[1, 2] * tau / tau1) * r2**3 + mu * D[0, 2] * (tau**2 - tau3**2) * tau3 / tau1)
                / (6 * r2**3 + mu * (tau**2 - tau1**2)) - D[2, 2]) / D0
        if min(rho1, rho2, rho3) <= 0:               # Eros must be IN FRONT of the telescope
            continue
        r = R + np.array([rho1, rho2, rho3])[:, None] * u
        f1, f3 = 1 - mu * tau1**2 / (2 * r2**3), 1 - mu * tau3**2 / (2 * r2**3)
        g1, g3 = tau1 - mu * tau1**3 / (6 * r2**3), tau3 - mu * tau3**3 / (6 * r2**3)
        v2 = (-f3 * r[0] + f1 * r[2]) / (f1 * g3 - f3 * g1)

        # move the state from t2 to the epoch t0 with c_'s simulator
        eps = 1e-3
        pos = R4(r[1], v2, t[1], np.array([t0 - eps, t0, t0 + eps]))
        candidates.append(np.concatenate([pos[1], (pos[2] - pos[0]) / (2 * eps)]))
    return candidates

if __name__ == "__main__":
    start_time = time.time()
    N = len(d)
    i1 = 0
    i2 = N//2
    i3 = N-1
    y0 = min(gauss(i1, i2, i3), key = lambda y: np.sum(miss(y)**2))
    sol = least_squares(miss, y0, x_scale = "jac")
    end_time = time.time()

    J = sol.jac
    print("Gradient should be ~0 :", np.abs(sol.grad).max())
    print("JᵀJ shd be Positive Definite:", np.linalg.eigvalsh(J.T @ J))
    print("Final RMS Error:", np.sqrt(np.mean(sol.fun**2)), "arcseconds")

    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULT = os.path.join(HERE, "results")
    JSON = os.path.join(RESULT, "hard_orbit.json")
    os.makedirs(RESULT, exist_ok = True)
    out_dict = dict(
        t0 = t0, 
        r0 = list(sol.x[:3]),
        v0 = list(sol.x[3:]),
        seconds = end_time - start_time
    )

    with open(JSON, "w") as f:
        json.dump(out_dict, f, indent = 1)
