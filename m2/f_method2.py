# This is the "Easy" (convex) problem solver for m2
#
# Formulates asteroid orbit determination as a convex linear least-squares problem
# using Lagrange f and g series expansion and line-of-sight orthogonal projections.
from a_imports import *
from b_constants import GM_SUN, ARCSEC, T0
from c_ephemeris import telescope_position, line_of_sight_basis
from d_simulation import predict_exact, misses, rms

# Import data processing from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from process_data import get_arrays, FIT_START, FIT_END

'''
Build the linear system A x = b for a given frozen heliocentric distance r_bar:
1. State variable: x = [r0, v0] in R^6 (r0 in au, v0 in au/day at epoch t0)
2. Trajectory approximation: r(t) approx f(t)*r0 + g(t)*v0
   where f(t) = 1 - 0.5*k*dt^2, g(t) = dt - (1/6)*k*dt^3, k = GM_SUN / r_bar^3
3. Orthogonal line-of-sight constraints:
   e_ra . (r(t) - R(t)) = 0  =>  f*e_ra . r0 + g*e_ra . v0 = e_ra . R
   e_dec . (r(t) - R(t)) = 0 =>  f*e_dec . r0 + g*e_dec . v0 = e_dec . R
4. Stacking for N observations yields A (2N x 6) and b (2N)
'''
def build_linear_system(jd_utc, ra, dec, d, h, L, t0, r_bar):
    jd_tt = jd_utc + 69.184 / 86400.0
    dt = jd_tt - t0

    # Gravitational parameter and series coefficients
    k = GM_SUN / (r_bar**3)
    f = 1.0 - 0.5 * k * (dt**2)
    g = dt - (1.0 / 6.0) * k * (dt**3)

    # Telescope positions and line of sight orthonormal basis
    R_tel = telescope_position(jd_utc, d, h, L)
    u, e_ra, e_dec = line_of_sight_basis(ra, dec)

    N = len(jd_utc)
    A = np.zeros((2 * N, 6))
    b = np.zeros(2 * N)

    for i in range(N):
        # Row 2*i: projection along Right Ascension tangent vector
        A[2 * i, :3] = f[i] * e_ra[i]
        A[2 * i, 3:] = g[i] * e_ra[i]
        b[2 * i] = np.dot(e_ra[i], R_tel[i])

        # Row 2*i + 1: projection along Declination tangent vector
        A[2 * i + 1, :3] = f[i] * e_dec[i]
        A[2 * i + 1, 3:] = g[i] * e_dec[i]
        b[2 * i + 1] = np.dot(e_dec[i], R_tel[i])

    return A, b


'''
Solve the convex least-squares problem min_x ||Ax - b||^2:
1. Matrix A has full column rank (rank 6)
2. Normal equations: (A^T A) x = A^T b
3. Uses SVD-based least squares (np.linalg.lstsq) for optimal numerical stability
'''
def solve_convex(A, b):
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond = None)
    return x, rank, s


'''
Find the self-consistent heliocentric distance r_bar:
1. The formulation requires choosing r_bar to approximate Sun-Eros distance.
2. Self-consistency demands: ||r0(r_bar)|| = r_bar.
3. Uses 1D bisection on g(r_bar) = ||r0(r_bar)|| - r_bar in [1.2, 1.8] au.
4. Guaranteed to bracket the unique physical asteroid solution (ignoring the ~1 au Earth branch).
'''
def find_self_consistent_rbar(jd_utc, ra, dec, d, h, L, t0, r_min = 1.2, r_max = 1.8, tol = 1e-6):
    low = r_min
    high = r_max

    for _ in range(40):
        mid = 0.5 * (low + high)
        A, b = build_linear_system(jd_utc, ra, dec, d, h, L, t0, mid)
        x, _, _ = solve_convex(A, b)
        diff = np.linalg.norm(x[:3]) - mid

        if abs(diff) < tol:
            return mid, x, A, b

        if diff > 0:
            low = mid
        else:
            high = mid

    r_best = 0.5 * (low + high)
    A, b = build_linear_system(jd_utc, ra, dec, d, h, L, t0, r_best)
    x, _, _ = solve_convex(A, b)
    return r_best, x, A, b


if __name__ == "__main__":
    start_time = time.time()

    # Step 1: Load fit observations (343 observations, 15 Oct 2025 to 15 Dec 2025)
    data = get_arrays(FIT_START, FIT_END)
    jd_utc = data["jd_utc"]
    ra = data["ra"]
    dec = data["dec"]
    d = data["d"]
    h = data["h"]
    L = data["L"]
    t0 = T0

    print("=" * 60)
    print("M2 CONVEX OPTIMIZATION SOLVER (EASY PROBLEM)")
    print("=" * 60)
    print(f"Loaded fit observations: {len(jd_utc)} from {len(set(data['codes']))} observatories")
    print(f"Reference Epoch (t0)   : {t0} JD (TT)")

    # Step 2: Solve with self-consistent r_bar
    r_bar_opt, x_opt, A_opt, b_opt = find_self_consistent_rbar(jd_utc, ra, dec, d, h, L, t0)
    end_time = time.time()
    elapsed_seconds = end_time - start_time

    r0_opt = x_opt[:3]
    v0_opt = x_opt[3:]
    r0_norm = np.linalg.norm(r0_opt)
    v0_norm = np.linalg.norm(v0_opt)

    # Step 3: Convexity & Optimality Verification
    # Objective: J(x) = ||Ax - b||^2
    # Gradient: grad J(x) = 2 A^T (Ax - b)
    # Hessian : H = 2 A^T A
    grad = 2.0 * A_opt.T @ (A_opt @ x_opt - b_opt)
    grad_norm = np.max(np.abs(grad))

    Hessian = 2.0 * (A_opt.T @ A_opt)
    eigvals = np.linalg.eigvalsh(Hessian)

    # Step 4: Physical Sanity Checks
    # Vis-viva specific orbital energy: E = 0.5*|v|^2 - GM/|r|
    specific_energy = 0.5 * (v0_norm**2) - GM_SUN / r0_norm
    is_bound = specific_energy < 0

    # Step 5: Astrometric verification with exact ODE physics
    ra_p, dec_p = predict_exact(r0_opt, v0_opt, t0, jd_utc, d, h, L)
    m = misses(ra, dec, ra_p, dec_p)
    exact_rms = rms(m)

    print("-" * 60)
    print("CONVEXITY & OPTIMALITY AUDIT")
    print("-" * 60)
    print(f"Self-consistent r_bar           : {r_bar_opt:.6f} au")
    print(f"Fitted ||r0||                   : {r0_norm:.6f} au (diff = {abs(r0_norm - r_bar_opt):.2e} au)")
    print(f"Gradient should be ~0 (||grad||): {grad_norm:.4e}  -> {'PASS' if grad_norm < 1e-8 else 'FAIL'}")
    print(f"Hessian eigenvalues (min, max)  : {eigvals[0]:.2e}, {eigvals[-1]:.2e}")
    print(f"Hessian strictly Positive Def.  : {'PASS' if eigvals[0] > 0 else 'FAIL'} (min eigval > 0)")
    print(f"Specific orbital energy         : {specific_energy:.4e} au^2/day^2")
    print(f"Orbit is gravitationally bound  : {'PASS' if is_bound else 'FAIL'}")
    print(f"Exact judged RMS fit miss       : {exact_rms:.2f} arcseconds")
    print(f"Solve runtime                   : {elapsed_seconds * 1000.0:.2f} ms")
    print("=" * 60)

    # Step 6: Save results to m2/results/easy_orbit.json
    HERE = os.path.dirname(os.path.abspath(__file__))
    RESULT_DIR = os.path.join(HERE, "results")
    JSON_PATH = os.path.join(RESULT_DIR, "easy_orbit.json")
    os.makedirs(RESULT_DIR, exist_ok = True)

    out_dict = dict(
        t0 = t0,
        r0 = list(r0_opt),
        v0 = list(v0_opt),
        seconds = elapsed_seconds
    )

    with open(JSON_PATH, "w") as f:
        json.dump(out_dict, f, indent = 1)

    print(f"Saved: {os.path.relpath(JSON_PATH, Path(HERE).parent)}")
    print("=" * 60)

# Program Completed

