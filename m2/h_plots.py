# EASY PROBLEM plots: draws the figures for the easy (convex) fit and its analysis
from a_imports import *

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

BLUE = "#2a78d6"           # hard / refined orbit reference
ORANGE = "#d6642a"         # easy / convex orbit
GREEN = "#2ca02c"
REF = "#898781"            # reference lines
TEXT = "#52514e"
BOX = dict(boxstyle = "round", fc = "#fcfcfb", ec = "#e1e0d9")

'''
Standard figure creator matching repo theme:
1. new_fig() -> figure with subtle clean grid
2. save()    -> bold title, tight layout, 150 dpi png in m2/figures/, closes figure
'''
def new_fig(w = 8, h = 4.5):
    fig, ax = plt.subplots(figsize = (w, h))
    ax.grid(True, which = "both", lw = 0.4, alpha = 0.5)
    ax.tick_params(colors = TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return fig, ax


def save(fig, name, title):
    os.makedirs(FIG_DIR, exist_ok = True)
    fig.suptitle(title, fontweight = "bold", color = TEXT)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, f"{name}.png")
    fig.savefig(path, dpi = 150, bbox_inches = "tight")
    plt.close(fig)
    return path


'''
Predictions plot: holdout data the easy fit never saw (Dec 2025 -> Jul 2026).
Shows individual observations (scatter) and 30-day monthly medians.
'''
def predictions(days, miss, monthly):
    fig, ax = new_fig()
    ax.scatter(days, miss, s = 8, color = ORANGE, alpha = 0.4, label = "each later observation")
    ax.plot([m["days_mid"] for m in monthly], [m["median_miss"] for m in monthly], color = ORANGE, lw = 2, marker = "o",
            label = "monthly median (easy model)")
    ax.set_yscale("log")
    ax.set_xlabel("days after the last fitted observation", color = TEXT)
    ax.set_ylabel("miss  [″]", color = TEXT)
    ax.legend(frameon = False)
    return save(fig, "predictions", "Easy orbit: predicting observations it never saw")


'''
Convexity slice plot:
Evaluates J(x* + alpha * d) along an arbitrary direction d in R^6.
A convex quadratic objective J(x) = ||Ax - b||^2 is a single bowl (parabola).
Chord connecting any two points always stays ABOVE the objective function.
'''
def convexity_slice(alphas, J_values, chord_idx1 = 10, chord_idx2 = 90):
    fig, ax = new_fig()
    ax.plot(alphas, J_values, color = ORANGE, lw = 2, label = "objective J(x* + α d)")

    # Draw chord between two points
    a1, a2 = alphas[chord_idx1], alphas[chord_idx2]
    j1, j2 = J_values[chord_idx1], J_values[chord_idx2]
    ax.plot([a1, a2], [j1, j2], color = BLUE, lw = 2, ls = "--", marker = "o", label = "chord between two points")

    mid_idx = (chord_idx1 + chord_idx2) // 2
    ax.annotate("curve is always below or on the chord (strictly convex)",
                (alphas[mid_idx], J_values[mid_idx]),
                xytext = (0, 30), textcoords = "offset points",
                ha = "center", color = TEXT, bbox = BOX,
                arrowprops = dict(arrowstyle = "->", color = TEXT, lw = 0.8))

    ax.set_xlabel("perturbation parameter α along search direction", color = TEXT)
    ax.set_ylabel("objective value J(x)", color = TEXT)
    ax.legend(frameon = False)
    return save(fig, "convexity_slice", "Convexity proof: objective is a single quadratic bowl")


'''
Fit residuals plot:
Displays Right Ascension and Declination residuals across the 343 fitted observations.
'''
def fit_residuals(jd_utc, dra, ddec):
    fig, ax = new_fig()
    days = jd_utc - jd_utc[0]
    ax.scatter(days, dra, s = 10, color = ORANGE, alpha = 0.6, label = "Δ RA · cos(Dec)  [″]")
    ax.scatter(days, ddec, s = 10, color = BLUE, alpha = 0.6, label = "Δ Dec  [″]")
    ax.axhline(0, color = REF, ls = "--", lw = 0.8)
    ax.set_xlabel("days from start of fit window (15 Oct 2025)", color = TEXT)
    ax.set_ylabel("residual  [″]", color = TEXT)
    ax.legend(frameon = False)
    return save(fig, "fit_residuals", "Fit window residuals across 343 observations")


if __name__ == "__main__":
    print("=" * 60)
    print("m2/h_plots.py sanity check: generate test figures")
    print("=" * 60)

    # Generate synthetic data for sanity checking
    test_days = np.linspace(1, 198, 200)
    test_miss = 100.0 + 20.0 * np.log1p(test_days) + np.random.randn(200) * 10
    test_monthly = [dict(days_mid = 15.0, median_miss = 110.0), dict(days_mid = 45.0, median_miss = 135.0)]

    test_alphas = np.linspace(-2.0, 2.0, 101)
    test_J = 0.05 + 2.5 * (test_alphas**2)

    test_dra = np.random.randn(343) * 200
    test_ddec = np.random.randn(343) * 200
    test_jd = np.linspace(2460960.0, 2461020.0, 343)

    paths = [
        predictions(test_days, test_miss, test_monthly),
        convexity_slice(test_alphas, test_J),
        fit_residuals(test_jd, test_dra, test_ddec),
    ]

    flags = [os.path.exists(p) for p in paths]
    for p, f in zip(paths, flags):
        print(f"{os.path.basename(p):25s}: {'PASS' if f else 'FAIL'}")
    print("=" * 60)
    print("ALL PASSED" if all(flags) else "SOME FAILED")
    print("=" * 60)

# Program Completed

