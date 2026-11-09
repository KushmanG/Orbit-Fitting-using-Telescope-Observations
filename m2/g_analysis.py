# EASY PROBLEM analysis: evaluates holdout predictions, verifies convexity, and assesses the price of realism
from a_imports import *
from b_constants import GM_SUN, ARCSEC, T0
from c_ephemeris import telescope_position
from d_simulation import predict_exact, misses, rms
from f_method2 import build_linear_system
import h_plots

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import process_data

'''
Evaluate holdout predictions (data the fit never saw):
1. Takes the 1,022 holdout observations from 15 Dec 2025 to 1 Jul 2026
2. Uses exact physics to predict RA and Dec from easy orbit (r0, v0)
3. Computes angular miss in arcseconds for every observation
4. Aggregates median miss per 30-day bin
'''
def evaluate_predictions(x, t0, jd_utc, ra, dec, d, h, L, t_end):
    r0 = x[:3]
    v0 = x[3:]
    ra_p, dec_p = predict_exact(r0, v0, t0, jd_utc, d, h, L)
    m = misses(ra, dec, ra_p, dec_p)
    miss_total = np.hypot(m[:, 0], m[:, 1])

    days = jd_utc - t_end
    month = np.floor(days / 30.0)
    monthly = []

    for m_idx in np.unique(month):
        k = (month == m_idx)
        monthly.append(dict(
            days_mid = float(np.median(days[k])),
            n = int(np.sum(k)),
            median_miss = float(np.median(miss_total[k]))
        ))

    return days, miss_total, monthly


'''
Evaluate 1D slice of the convex objective function:
J(x) = ||Ax - b||^2
For any perturbation direction d in R^6:
J(x* + alpha * d) = ||A(x* + alpha*d) - b||^2 = ||(Ax* - b) + alpha*(Ad)||^2
This is a pure quadratic polynomial in alpha with positive leading coefficient 2 ||Ad||^2 > 0.
Every chord lies strictly above the curve.
'''
def evaluate_convexity_slice(x_opt, A, b, n_points = 101):
    alphas = np.linspace(-0.01, 0.01, n_points)
    # Choose perturbation direction along velocity unit vector (e.g. index 3)
    d = np.zeros(6)
    d[3] = 1.0

    J_values = []
    for a in alphas:
        x_perturbed = x_opt + a * d
        res = A @ x_perturbed - b
        J_values.append(float(np.sum(res**2)))

    return alphas, np.array(J_values)


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    JSON_PATH = os.path.join(HERE, "results", "easy_orbit.json")

    if not os.path.exists(JSON_PATH):
        print(f"Error: {JSON_PATH} not found. Run f_method2.py first.")
        sys.exit(1)

    with open(JSON_PATH) as f:
        orbit = json.load(f)

    r0 = np.array(orbit["r0"])
    v0 = np.array(orbit["v0"])
    x_opt = np.concatenate([r0, v0])
    t0 = orbit["t0"]

    # Load data
    fit_data = process_data.get_arrays(process_data.FIT_START, process_data.FIT_END)
    holdout_data = process_data.get_arrays(process_data.HOLDOUT_START, process_data.HOLDOUT_END)

    t_end = process_data.calendar_to_jd(*process_data.FIT_END)

    print("=" * 60)
    print("M2 (EASY PROBLEM) ANALYSIS REPORT")
    print("=" * 60)

    # -------------------------------------------------------------
    # Test 1: Predictions on holdout data
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 1: predictions on data the fit never saw (holdout window)")
    print("=" * 60)
    days_hold, miss_hold, monthly_stats = evaluate_predictions(
        x_opt, t0,
        holdout_data["jd_utc"], holdout_data["ra"], holdout_data["dec"],
        holdout_data["d"], holdout_data["h"], holdout_data["L"],
        t_end
    )

    for b_stat in monthly_stats:
        print(f"    day {b_stat['days_mid']:6.1f}: median miss {b_stat['median_miss']:8.2f}\"  ({b_stat['n']} obs)")

    # -------------------------------------------------------------
    # Test 2: Convexity proof and quadratic slice
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 2: convexity verification and objective slice")
    print("=" * 60)
    r_bar_norm = np.linalg.norm(r0)
    A_fit, b_fit = build_linear_system(
        fit_data["jd_utc"], fit_data["ra"], fit_data["dec"],
        fit_data["d"], fit_data["h"], fit_data["L"],
        t0, r_bar_norm
    )

    grad = 2.0 * A_fit.T @ (A_fit @ x_opt - b_fit)
    Hessian = 2.0 * (A_fit.T @ A_fit)
    eigvals = np.linalg.eigvalsh(Hessian)

    print(f"    Optimality gradient ||grad||  : {np.max(np.abs(grad)):.3e} (optimality condition = 0)")
    print(f"    Hessian minimum eigenvalue    : {eigvals[0]:.4e} (> 0 => strictly convex)")
    print(f"    Hessian condition number      : {eigvals[-1] / eigvals[0]:.2e}")

    alphas, J_slice = evaluate_convexity_slice(x_opt, A_fit, b_fit)
    print(f"    Convex objective at minimum   : {J_slice[len(alphas)//2]:.6e}")

    # -------------------------------------------------------------
    # Test 3: Residuals on fit window
    # -------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Test 3: fit window residuals (judged with exact physics)")
    print("=" * 60)
    ra_p, dec_p = predict_exact(r0, v0, t0, fit_data["jd_utc"], fit_data["d"], fit_data["h"], fit_data["L"])
    m_fit = misses(fit_data["ra"], fit_data["dec"], ra_p, dec_p)
    fit_rms = rms(m_fit)
    print(f"    Exact judged fit RMS: {fit_rms:.3f}\"")

    # Save analysis summary JSON
    analysis_dict = dict(
        fit_rms_arcsec = fit_rms,
        solve_seconds = orbit.get("seconds", 0.0),
        monthly_holdout = monthly_stats,
        gradient_max = float(np.max(np.abs(grad))),
        hessian_eig_min = float(eigvals[0]),
        hessian_eig_max = float(eigvals[-1])
    )

    ANALYSIS_JSON = os.path.join(HERE, "results", "analysis.json")
    with open(ANALYSIS_JSON, "w") as f:
        json.dump(analysis_dict, f, indent = 1)

    # Generate and save plots
    p1 = h_plots.predictions(days_hold, miss_hold, monthly_stats)
    p2 = h_plots.convexity_slice(alphas, J_slice)
    p3 = h_plots.fit_residuals(fit_data["jd_utc"], m_fit[:, 0], m_fit[:, 1])

    print(f"\nsaved: {os.path.relpath(p1, HERE)}")
    print(f"saved: {os.path.relpath(p2, HERE)}")
    print(f"saved: {os.path.relpath(p3, HERE)}")
    print(f"saved: {os.path.relpath(ANALYSIS_JSON, HERE)}")
    print("=" * 60)

# Program Completed

