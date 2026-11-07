# HARD PROBLEM plots: draws the figures for the hard fit and its analysis
from a_imports import *

FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")      # m1/figures

BLUE = "#2a78d6"           # hard / refined orbit
ORANGE = "#d6642a"         # easy / initial orbit
REF = "#898781"            # reference lines
TEXT = "#52514e"
BOX = dict(boxstyle = "round", fc = "#fcfcfb", ec = "#e1e0d9")

'''
Every plot goes through these two helpers so they all look the same:
    1. new_fig() -> figure with my grid style
    2. save()    -> bold title, tight layout, 150 dpi png in figures/, close it (no memory leak lol)
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
Predictions: the hard orbit predicts observations it never saw.
    1. Every dot is one later observation
    2. The line is the median miss per 30-day bin (from h_analysis.predictions)
'''
def predictions(days, miss, monthly):
    fig, ax = new_fig()
    ax.scatter(days, miss, s = 8, color = BLUE, alpha = 0.4, label = "each later observation")
    ax.plot([m["days_mid"] for m in monthly], [m["median_miss"] for m in monthly], color = BLUE, lw = 2, marker = "o",
            label = "monthly median")
    ax.axhline(1, color = REF, ls = (0, (4, 4)), lw = 1)
    ax.text(days.max(), 1.15, "1 arcsec", ha = "right", color = TEXT, fontsize = 9)
    ax.set_yscale("log")
    ax.set_xlabel("days after the last fitted observation", color = TEXT)
    ax.set_ylabel("miss  [″]", color = TEXT)
    ax.legend(frameon = False)
    return save(fig, "predictions", "Hard orbit: predicting observations it never saw")


'''
Non-convexity picture: the objective as Eros slides along its own orbit, with one chord.
    1. Plotted on a LINEAR scale, cz convexity is about chi^2 itself, not log(chi^2)
    2. I search for the chord the curve rises above the most and draw it
    3. Curve above a chord anywhere = not convex (that is literally the definition)
'''
def chi2_slice(shifts, chi2):
    fig, ax = new_fig()
    best, pair = 0, (0, len(shifts) - 1)
    for i in range(0, len(shifts), 3):
        for j in range(i + 6, len(shifts), 3):
            m = (i + j) // 2
            gap = chi2[m] - (chi2[i] + chi2[j]) / 2
            if gap > best:
                best, pair = gap, (i, j)
    i, j = pair
    ax.plot(shifts, chi2, color = BLUE, lw = 2)
    ax.plot([shifts[i], shifts[j]], [chi2[i], chi2[j]], color = ORANGE, lw = 2, marker = "o")
    m = (i + j) // 2
    ax.annotate("curve above the chord", (shifts[m], chi2[m]), xytext = (0, 25), textcoords = "offset points",
                ha = "center", color = TEXT, bbox = BOX)
    ax.set_xlabel("shift along the orbit  [days]", color = TEXT)
    ax.set_ylabel("χ²", color = TEXT)
    return save(fig, "chi2_slice", "The objective is not convex")


'''
Ablation: one bar per variant, the full model highlighted.
'''
def ablation(names, rms):
    fig, ax = new_fig(8, 0.6 * len(names) + 1)
    y = np.arange(len(names))[::-1]
    ax.barh(y, rms, color = [BLUE] + [REF] * (len(names) - 1), height = 0.6)
    ax.set_yticks(y, names)
    for yy, v in zip(y, rms):
        ax.text(v, yy, f"  {v:.2f}″", va = "center", color = TEXT)
    ax.set_xlabel("RMS miss on the kept observations  [″]", color = TEXT)
    ax.set_xlim(0, max(rms) * 1.25)
    return save(fig, "ablation", "Which physics matters")


if __name__ == "__main__":
    print("=" * 60)
    print("i_plots sanity check: draw every figure from fake numbers")
    print("=" * 60)
    random = np.random.default_rng(seed = 42)
    paths = [
        predictions(np.arange(1, 200.0), 2 + random.random(199) * 10, [dict(days_mid = 15.0, median_miss = 2.0), dict(days_mid = 45.0, median_miss = 6.0)]),
        chi2_slice(np.linspace(-300, 300, 121), 10 ** (2 + 10 * np.abs(np.sin(np.linspace(-1.5, 1.5, 121))))),
        ablation(["full model", "no light delay", "telescope at Earth's centre"], [0.33, 11.8, 3.5]),
    ]
    flags = [os.path.exists(p) for p in paths]
    for p, f in zip(paths, flags):
        print(os.path.basename(p), "PASS" if f else "FAIL", "\n")
    print("ALL PASS" if all(flags) else "SOME FAILED")

# Program Completed
