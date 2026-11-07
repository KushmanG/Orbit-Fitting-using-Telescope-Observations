# HARD PROBLEM analysis: tests the refined orbit (predictions, global minimum, non-convexity, which physics matters)
from a_imports import *
from b_constants import GM_SUN, c, ARCSEC
from d_relativePosition import R1, R2, R4



'''
Why a separate predict() when d_ already has result():
    1. Same chain as result(): r5 = r4(t - T) - r1(t) - r2(t), then atan2 for RA, Dec
    2. But here I need two switches to turn physics OFF for the ablation test:
           light_time = False  -> no light delay (T = 0)
           topocentric = False -> telescope sits at Earth's centre (no r2)
    3. x is the usual 6 numbers: x[:3] = r0, x[3:] = v0
'''
def predict(x, jd_utc, t0, d, h, L, light_time = True, topocentric = True):
    r0, v0 = x[:3], x[3:]
    jd_tt = jd_utc + 69.184 / 86400
    r3 = R1(jd_utc)
    if topocentric:
        r3 = r3 + R2(jd_utc, d, h, L)

    T = np.zeros(len(jd_utc))
    for _ in range(3 if light_time else 1):
        r5 = R4(r0, v0, t0, jd_tt - T) - r3
        T = np.linalg.norm(r5, axis = 1) / c

    ra = np.remainder(np.arctan2(r5[:, 1], r5[:, 0]), 2 * np.pi)
    dec = np.arctan2(r5[:, 2], np.sqrt(r5[:, 0]**2 + r5[:, 1]**2))
    return ra, dec


'''
Miss between measured and predicted, in arcseconds, shape (N, 2):
    1. dRA is wrapped into [-pi, pi] so 359.9 deg vs 0.1 deg counts as 0.2 deg, not 359.8
    2. dRA is multiplied by cos(Dec) cz RA circles shrink towards the poles
'''
def misses(ra, dec, ra_p, dec_p):
    dra = (np.remainder(ra - ra_p + np.pi, 2 * np.pi) - np.pi) * np.cos(dec)
    ddec = dec - dec_p
    return np.stack([dra, ddec], axis = 1) * ARCSEC


def rms(m):
    return float(np.sqrt(np.mean(m**2)))


'''
Small fitter used by the tests below (same idea as g_refine_orbit, kept here so h_ can run on its own):
    1. residual = miss / sigma, both RA and Dec, flattened
    2. least_squares nudges the 6 numbers downhill until the miss stops shrinking
'''
def fit(x0, jd_utc, ra, dec, t0, d, h, L, sigma = None, light_time = True, topocentric = True, max_nfev = 200):
    sigma = np.ones(len(jd_utc)) if sigma is None else sigma

    def residuals(x):
        ra_p, dec_p = predict(x, jd_utc, t0, d, h, L, light_time, topocentric)
        return (misses(ra, dec, ra_p, dec_p) / sigma[:, None]).ravel()

    sol = least_squares(residuals, x0, x_scale = "jac", max_nfev = max_nfev)
    return sol.x


# ------------------------------------------------------------------------------------------------
# Test 1: predictions on LATER observations the fit never saw
# ------------------------------------------------------------------------------------------------
'''
The honest test of any model: predict data it was never shown.
    1. Give it the observations after the fit window (Dec 2025 -> Jul 2026)
    2. Miss of every observation, in arcsec
    3. Median miss per 30-day bin (median so a few bad measurements don't dominate)
'''
def predictions(x, t0, jd_utc, ra, dec, d, h, L, t_end):
    ra_p, dec_p = predict(x, jd_utc, t0, d, h, L)
    miss = np.hypot(*misses(ra, dec, ra_p, dec_p).T)
    days = jd_utc - t_end
    month = np.floor(days / 30)
    monthly = []
    for m in np.unique(month):
        k = month == m
        monthly.append(dict(days_mid = float(np.median(days[k])), n = int(k.sum()), median_miss = float(np.median(miss[k]))))
    return days, miss, monthly


# ------------------------------------------------------------------------------------------------
# Test 2: chi^2 slice, a picture of the non-convexity
# ------------------------------------------------------------------------------------------------
'''
Slide Eros along its OWN best orbit and watch the miss:
    1. Shift the starting point by s days along the orbit (new r0, v0 = where Eros is at t0 + s)
    2. Keep everything else the same and compute the total miss
    3. A convex function never rises above the straight chord between two of its points.
       This curve does (it is periodic in s), so the objective is not convex.
'''
def state_at(x, t0, t):
    r0, v0 = x[:3], x[3:]
    eps = 1e-3                                    # days, for a finite-difference velocity
    pos = R4(r0, v0, t0, np.array([t - eps, t, t + eps]))
    return np.concatenate([pos[1], (pos[2] - pos[0]) / (2 * eps)])


def chi2_slice(x_best, jd_utc, ra, dec, t0, d, h, L, sigma = None, shifts = None):
    sigma = np.ones(len(jd_utc)) if sigma is None else sigma
    a = 1.0 / (2.0 / np.linalg.norm(x_best[:3]) - x_best[3:] @ x_best[3:] / GM_SUN)       # vis-viva
    period = 2 * np.pi * np.sqrt(a**3 / GM_SUN)
    shifts = np.linspace(-period / 2, period / 2, 121) if shifts is None else shifts
    chi2 = []
    for s in shifts:
        x = state_at(x_best, t0, t0 + s)
        m = misses(ra, dec, *predict(x, jd_utc, t0, d, h, L)) / sigma[:, None]
        chi2.append(float(np.sum(m**2)))
    return shifts, np.array(chi2)


# ------------------------------------------------------------------------------------------------
# Test 3: ablation, which piece of physics matters
# ------------------------------------------------------------------------------------------------
'''
Switch off ONE piece of physics at a time, refit, and see how much worse it gets.
    1. Every variant is fitted on the same observations, starting from the best orbit
    2. Every variant is JUDGED the same way: full-model misses (so it is fair)
'''
def ablation(x_best, jd_utc, ra, dec, t0, d, h, L):
    variants = {
        "full model": dict(),
        "no light delay": dict(light_time = False),
        "telescope at Earth's centre": dict(topocentric = False),
    }
    rows = []
    for name, opt in variants.items():
        x = fit(x_best, jd_utc, ra, dec, t0, d, h, L, **opt)
        judged = rms(misses(ra, dec, *predict(x, jd_utc, t0, d, h, L)))
        rows.append(dict(variant = name, rms = judged))
        print(f"    {name:30s} {judged:8.3f}\"")
    return rows


'''
Runs all three tests on the orbit f_method1 saved, prints the numbers, saves them, and draws the figures (h_plots)
    Run f_method1.py first, this reads m1/results/hard_orbit.json
'''
if __name__ == "__main__":
    import h_plots
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import process_data

    HERE = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(HERE, "results", "hard_orbit.json")) as f:
        o = json.load(f)
    x, t0 = np.concatenate([o["r0"], o["v0"]]), o["t0"]

    fit_data = process_data.get_arrays(process_data.FIT_START, process_data.FIT_END)
    later = process_data.get_arrays(process_data.HOLDOUT_START, process_data.HOLDOUT_END)
    window = lambda a: (a["jd_utc"], a["ra"], a["dec"], a["d"], a["h"], a["L"])
    jd_utc, ra, dec, d, h, L = window(fit_data)
    t_end = process_data.calendar_to_jd(*process_data.FIT_END)

    print("=" * 60, "\nTest 1: predictions on data the fit never saw\n" + "=" * 60)
    days, miss, monthly = predictions(x, t0, *window(later), t_end)
    for b in monthly:
        print(f"    day {b['days_mid']:6.1f}: median miss {b['median_miss']:8.2f}\"  ({b['n']} obs)")

    print("=" * 60, "\nTest 2: chi^2 slice along the orbit\n" + "=" * 60)
    shifts, chi2 = chi2_slice(x, jd_utc, ra, dec, t0, d, h, L)
    mid = len(shifts) // 2
    print(f"    chi^2 at the best orbit {chi2[mid]:.3e}, worst along the orbit {chi2.max():.3e}")

    print("=" * 60, "\nTest 3: ablation\n" + "=" * 60)
    rows = ablation(x, jd_utc, ra, dec, t0, d, h, L)

    summary = dict(monthly = monthly, ablation = rows)
    with open(os.path.join(HERE, "results", "analysis.json"), "w") as f:
        json.dump(summary, f, indent = 1)

    for p in [h_plots.predictions(days, miss, monthly), h_plots.chi2_slice(shifts, chi2),
              h_plots.ablation([r["variant"] for r in rows], [r["rms"] for r in rows])]:
        print("saved:", os.path.relpath(p, HERE))
    print("saved: results/analysis.json")

# Program Completed
