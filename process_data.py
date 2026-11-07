# This file turns the raw MPC files (observations + observatory codes) into clean numpy arrays for the fit
import numpy as np
import os
import erfa
import json

# Time windows in UTC, [start, end), as (year, month, day)
FIT_START = (2025, 10, 15)
FIT_END = (2025, 12, 15)
HOLDOUT_START = (2025, 12, 15)
HOLDOUT_END = (2026, 7, 1)

# Where the raw files live (relative to this file so it works from any folder)
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")
OBS_FILE = os.path.join(RAW_DIR, "MPC_Observations_433Eros.json")
OBSCODE_FILE = os.path.join(RAW_DIR, "MPC_Observatory_Codes.json")

'''
Dates -> Julian Date (jd_utc):
1. A Julian Date is just "days counted since noon, 1 Jan 4713 BC" as one decimal number. No months, no leap year drama, ez to subtract
2. I use erfa.cal2jd(year, month, day) which gives the JD at 0h of that calendar day, split in 2 parts (2400000.5 + MJD) for precision
3. Then I just add the fraction of the day (MPC gives day as 11.596297 so fraction = 0.596297)
4. No astropy needed, erfa is already in a_imports and d_ uses it anyway
'''
def calendar_to_jd(year, month, day_decimal):
    day_whole = int(np.floor(day_decimal))
    day_fraction = day_decimal - day_whole
    jd_part1, jd_part2 = erfa.cal2jd(year, month, day_whole)
    return jd_part1 + jd_part2 + day_fraction

'''
Parsing one OBS80 line (80 characters, every field sits at a fixed column):
1. line[14]     -> observation type ("C" = CCD)
2. line[15:32]  -> UTC date "2025 05 11.596297" (year, month, decimal day)
3. line[32:44]  -> RA "23 12 51.514" (hours minutes seconds)
4. line[44:56]  -> Dec "-02 37 37.16" (sign, degrees minutes seconds)
5. line[77:80]  -> observatory code ("T08")

GOTCHA: For Dec "-00 12 34", float("-00") = 0.0 and the minus is GONE (cz -0 == 0 yk)
So I read the sign character at line[44] by itself and multiply at the very end

RA: hours -> degrees (x15, cz 24h = 360 deg) -> radians
Dec: degrees -> radians (with the sign)

If the line is short or anything fails to parse (old formats, weird rows) I return None and the caller skips it
'''
def parse_obs80(line):
    if len(line) < 80:
        return None
    try:
        # Step 1: type and station
        obs_type = line[14]
        code = line[77:80]

        # Step 2: date
        date_parts = line[15:32].split()
        year = int(date_parts[0])
        month = int(date_parts[1])
        day_decimal = float(date_parts[2])
        jd_utc = calendar_to_jd(year, month, day_decimal)

        # Step 3: RA (h m s)
        ra_parts = line[32:44].split()
        ra_hours = float(ra_parts[0]) + float(ra_parts[1])/60 + float(ra_parts[2])/3600
        ra = np.radians(ra_hours * 15)

        # Step 4: Dec, sign read separately (the GOTCHA)
        sign_char = line[44]
        if sign_char == "-":
            sign = -1.0
        elif sign_char == "+" or sign_char == " ":
            sign = 1.0
        else:
            return None
        dec_parts = line[45:56].split()
        dec_degrees = float(dec_parts[0]) + float(dec_parts[1])/60 + float(dec_parts[2])/3600
        dec = np.radians(sign * dec_degrees)
    except (ValueError, IndexError):
        return None

    return {"jd_utc": jd_utc, "ra": ra, "dec": dec, "type": obs_type, "code": code}

'''
Loading the observations file:
1. The MPC get-obs reply is either a dict with "OBS80" or a list whose first element is that dict, so I handle both
2. "OBS80" is one big string, one observation per line
3. Every line goes through parse_obs80, the ones that return None are dropped
'''
def load_observations(path = OBS_FILE):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = raw[0]

    observations = []
    for line in raw["OBS80"].splitlines():
        parsed = parse_obs80(line)
        if parsed is not None:
            observations.append(parsed)
    return observations

'''
Loading observatory codes:
1. File looks like {"C40": {"Longitude": 39.03, "cos": 0.708, "sin": 0.704, "Name": "..."}, ...}
2. Codes without "cos" are space telescopes / roving observers, no fixed spot on Earth so R2 can't handle them, skip
'''
def load_obscodes(path = OBSCODE_FILE):
    with open(path) as f:
        raw = json.load(f)

    obscodes = {}
    for code in raw:
        entry = raw[code]
        if "cos" in entry and "sin" in entry and "Longitude" in entry:
            obscodes[code] = entry
    return obscodes

'''
Cleaning (ONLY selects rows, never edits them):
1. jd_utc inside [start, end) -> fit vs hold-out stay separate, no data leaking between them
2. type == "C" -> CCD obs are the precise modern ones, photographic/visual/radar etc. have different errors or aren't even RA/Dec
3. code in obscodes (with "cos") -> otherwise we don't know where the telescope was, so R2 is impossible
'''
def clean(observations, obscodes, jd_start, jd_end):
    kept = []
    for obs in observations:
        if obs["jd_utc"] < jd_start or obs["jd_utc"] >= jd_end:
            continue
        if obs["type"] != "C":
            continue
        if obs["code"] not in obscodes:
            continue
        kept.append(obs)
    return kept

'''
Final arrays for a window, all length N and aligned (index i = same observation everywhere):
1. jd_utc      -> time of observation (Julian Date, UTC)
2. ra, dec     -> measured angles in radians
3. d, h, L     -> station's "cos", "sin" (Earth radii) and "Longitude" (degrees), exactly what R2(jd_utc, d, h, L) wants
4. codes       -> station codes, for reports
Sorted by time cz simulate() in c_ needs sorted t
'''
def get_arrays(start, end, observations = None, obscodes = None):
    # Step 1: load if not given (so I can load once and reuse for many windows)
    if observations is None:
        observations = load_observations()
    if obscodes is None:
        obscodes = load_obscodes()

    # Step 2: window in JD
    jd_start = calendar_to_jd(start[0], start[1], start[2])
    jd_end = calendar_to_jd(end[0], end[1], end[2])
    kept = clean(observations, obscodes, jd_start, jd_end)

    # Step 3: sort by time
    kept.sort(key = lambda obs: obs["jd_utc"])

    # Step 4: build the arrays
    jd_utc, ra, dec, d, h, L, codes = [], [], [], [], [], [], []
    for obs in kept:
        station = obscodes[obs["code"]]
        jd_utc.append(obs["jd_utc"])
        ra.append(obs["ra"])
        dec.append(obs["dec"])
        d.append(station["cos"])
        h.append(station["sin"])
        L.append(station["Longitude"])
        codes.append(obs["code"])

    return {
        "jd_utc": np.array(jd_utc),
        "ra": np.array(ra),
        "dec": np.array(dec),
        "d": np.array(d),
        "h": np.array(h),
        "L": np.array(L),
        "codes": np.array(codes),
    }


if __name__ == "__main__":
    observations = load_observations()
    obscodes = load_obscodes()
    fit = get_arrays(FIT_START, FIT_END, observations, obscodes)
    holdout = get_arrays(HOLDOUT_START, HOLDOUT_END, observations, obscodes)

    failed = []

    # flag1: fit window count and telescopes
    n_fit = len(fit["jd_utc"])
    n_tel = len(set(fit["codes"]))
    flag1 = (n_fit == 343 and n_tel == 11)
    if not flag1: failed.append("flag1 (fit window size)")

    # flag2: hold-out count
    n_hold = len(holdout["jd_utc"])
    flag2 = (n_hold == 1022)
    if not flag2: failed.append("flag2 (hold-out size)")

    # flag3: known T08 line parses right
    test_line = None
    with open(OBS_FILE) as f:
        raw = json.load(f)
    if isinstance(raw, list):
        raw = raw[0]
    for line in raw["OBS80"].splitlines():
        if "23 12 51.514-02 37 37.16" in line and line[77:80] == "T08":
            test_line = line
    test = parse_obs80(test_line) if test_line is not None else None
    if test is not None:
        ra_deg = np.degrees(test["ra"])
        dec_deg = np.degrees(test["dec"])
        flag3 = abs(ra_deg - 348.214) < 1e-3 and abs(dec_deg - (-2.627)) < 1e-3
    else:
        ra_deg, dec_deg = float("nan"), float("nan")
        flag3 = False
    if not flag3: failed.append("flag3 (T08 line parse)")

    # flag4: angle ranges in both windows
    all_ra = np.concatenate([fit["ra"], holdout["ra"]])
    all_dec = np.concatenate([fit["dec"], holdout["dec"]])
    flag4 = bool(np.all((all_ra >= 0) & (all_ra < 2 * np.pi)) and np.all((all_dec >= -np.pi/2) & (all_dec <= np.pi/2)))
    if not flag4: failed.append("flag4 (RA/Dec ranges)")

    # Report
    print("=====================================================")
    print("              e_data.py SANITY REPORT")
    print("=====================================================")
    print(f"Parsed OBS80 lines         : {len(observations)}")
    print(f"Ground observatory codes   : {len(obscodes)}")
    print("-----------------------------------------------------")
    print(f"flag1 fit: {n_fit} obs, {n_tel} telescopes (want 343, 11)   : {'PASS' if flag1 else 'FAIL'}")
    print(f"flag2 hold-out: {n_hold} obs (want 1022)               : {'PASS' if flag2 else 'FAIL'}")
    print(f"flag3 T08: RA {ra_deg:.4f} deg, Dec {dec_deg:.4f} deg       : {'PASS' if flag3 else 'FAIL'}")
    print(f"flag4 RA in [0, 2pi), Dec in [-pi/2, pi/2]       : {'PASS' if flag4 else 'FAIL'}")
    print("=====================================================")
    if len(failed) == 0:
        print("ALL PASSED")
    else:
        print(f"FAILED: {', '.join(failed)}")
    print("=====================================================")

# Program Completed
