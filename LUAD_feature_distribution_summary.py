#!/usr/bin/env python3
"""
Characterise the distribution of each WSI-level feature, per group.

For every feature this reports the variable type, support, candidate
distributions and notes (the reference table below), plus data-driven
checks: descriptive stats, zeros / boundary values, dispersion for counts,
and maximum-likelihood fits of each candidate distribution compared by
AIC / BIC.

Input files (one per group), e.g. output of LUAD_clean_tumor_core.py:
    {group_name}_{feature}_wsi_clean.csv

Outputs, one folder per group in --output-dir:
    {group}/{group}_{feature}_distribution_summary.csv   one row per feature
    {group}/{group}_{feature}_distribution_fits.csv      one row per feature x distribution

Dependencies (no scipy needed):
    pip install pandas numpy

Examples:
    # all groups
    python LUAD_feature_distribution_summary.py --group all

    # one or more specific groups
    python LUAD_feature_distribution_summary.py --group female_psg_negative
    python LUAD_feature_distribution_summary.py --group male_psg_negative male_psg_positive

    # also analyse numeric columns not in the reference table (types inferred)
    python LUAD_feature_distribution_summary.py --group all --columns all
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Reference table: feature -> (variable type, support, candidates, note)
# Candidate keys: normal, lognormal, gamma, beta, poisson, negbin
# --------------------------------------------------------------------------
COUNT = "Discrete count"
POS = "Positive continuous"
PROP = "Continuous proportion"
BOUNDED = "Bounded continuous"
CONT = "Continuous"

FEATURE_SPEC = {
    "n_clusters_with_tumor": (COUNT, "0,1,2,...", ["poisson", "negbin"],
                              "NB if variance is substantially greater than mean"),
    "wsi_tumor_n_islands": (COUNT, "0,1,2,...", ["poisson", "negbin"],
                            "May need area/exposure adjustment"),
    "wsi_cluster_area_mm2": (POS, ">0", ["lognormal", "gamma"], "Likely right-skewed"),
    "wsi_tumor_area_mm2": (POS, ">0", ["lognormal", "gamma"], "Likely right-skewed"),
    "wsi_tumor_perimeter_mm": (POS, ">0", ["lognormal", "gamma"], "Positive size measurement"),
    "wsi_tumor_fraction_of_cluster": (PROP, "0-1", ["beta"],
                                      "If exact 0/1 values occur, ordinary Beta needs modification"),
    "wsi_tumor_boundary_density_per_mm": (POS, ">=0", ["gamma", "lognormal"],
                                          "Ratio/density; inspect zeros"),
    "wsi_tumor_patch_density_per_mm2": (POS, ">=0", ["gamma", "lognormal"], "Density measurement"),
    "wsi_area_weighted_tumor_largest_patch_index": (PROP, "0-1", ["beta"],
                                                    "Fraction of tumor area in largest island"),
    "wsi_area_weighted_tumor_compactness_mean": (BOUNDED, "usually 0-1", ["beta", "normal"],
                                                 "Beta if clearly bounded/skewed; Normal may work if "
                                                 "values are away from boundaries and symmetric"),
    "wsi_area_weighted_tumor_solidity_mean": (BOUNDED, "0-1", ["beta"],
                                              "Natural proportion-like variable"),
    "wsi_area_weighted_tumor_elongation_mean": (POS, "typically >=1", ["lognormal", "gamma", "normal"],
                                                "Distribution depends strongly on observed shape"),
    "wsi_area_weighted_tumor_boundary_fractal_dimension": (CONT, "bounded physical range", ["normal"],
                                                           "Often relatively narrow; inspect histogram/Q-Q plot"),
    "wsi_area_weighted_tumor_island_nnd_median_um": (POS, ">=0", ["lognormal", "gamma"],
                                                     "Spatial distance; may be right-skewed"),
    "wsi_area_weighted_tumor_island_gap_median_um": (POS, ">=0", ["lognormal", "gamma"],
                                                     "Spatial distance; may contain zero"),
}

# Columns never treated as features when --columns all
NON_FEATURE_COLS = {
    "wsi_name", "sample_id", "subject_id", "Spacing", "SEX", "AGE", "OS_MONTHS",
    "OS_STATUS", "OS_Status", "stage", "AJCC_PATHOLOGIC_TUMOR_STAGE",
    "clinical__sample_id", "clinical__subject_id", "clinical_match", "group",
}

DIST_LABEL = {
    "normal": "Normal", "lognormal": "Log-normal", "gamma": "Gamma",
    "beta": "Beta", "poisson": "Poisson", "negbin": "Negative Binomial",
}


# --------------------------------------------------------------------------
# Special functions (numpy only)
# --------------------------------------------------------------------------
_lgamma = np.vectorize(math.lgamma, otypes=[float])


def digamma(x):
    x = np.asarray(x, dtype=float).copy()
    res = np.zeros_like(x)
    while np.any(x < 6):
        m = x < 6
        res[m] -= 1.0 / x[m]
        x[m] += 1.0
    inv2 = 1.0 / (x * x)
    res += (np.log(x) - 0.5 / x
            - inv2 * (1 / 12 - inv2 * (1 / 120 - inv2 * (1 / 252 - inv2 * (1 / 240 - inv2 / 132)))))
    return res


def trigamma(x):
    x = np.asarray(x, dtype=float).copy()
    res = np.zeros_like(x)
    while np.any(x < 6):
        m = x < 6
        res[m] += 1.0 / (x[m] * x[m])
        x[m] += 1.0
    inv = 1.0 / x
    inv2 = inv * inv
    res += inv + inv2 / 2 + inv * inv2 * (1 / 6 - inv2 * (1 / 30 - inv2 * (1 / 42 - inv2 / 30)))
    return res


# --------------------------------------------------------------------------
# Maximum-likelihood fits: each returns (params dict, log-likelihood, k)
# --------------------------------------------------------------------------
def fit_normal(x):
    mu, sd = x.mean(), x.std()
    ll = -len(x) * (math.log(sd) + 0.5 * math.log(2 * math.pi)) - ((x - mu) ** 2).sum() / (2 * sd ** 2)
    return {"mu": mu, "sigma": sd}, ll, 2


def fit_lognormal(x):
    lx = np.log(x)
    params, ll_log, k = fit_normal(lx)
    return {"meanlog": params["mu"], "sdlog": params["sigma"]}, ll_log - lx.sum(), k


def fit_gamma(x):
    n, m = len(x), x.mean()
    s = math.log(m) - np.log(x).mean()
    k = (3 - s + math.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(100):
        step = (math.log(k) - digamma([k])[0] - s) / (1 / k - trigamma([k])[0])
        k_new = k - step
        k = k_new if k_new > 0 else k / 2
        if abs(step) < 1e-10 * k:
            break
    theta = m / k
    ll = ((k - 1) * np.log(x).sum() - x.sum() / theta - n * k * math.log(theta) - n * math.lgamma(k))
    return {"shape": k, "scale": theta}, ll, 2


def fit_beta(x):
    n = len(x)
    m, v = x.mean(), x.var()
    common = m * (1 - m) / v - 1 if v > 0 else 1.0
    a, b = max(m * common, 1e-3), max((1 - m) * common, 1e-3)
    s1, s2 = np.log(x).mean(), np.log1p(-x).mean()
    for _ in range(200):
        dab = digamma([a + b])[0]
        g = np.array([digamma([a])[0] - dab - s1, digamma([b])[0] - dab - s2])
        tab = trigamma([a + b])[0]
        J = np.array([[trigamma([a])[0] - tab, -tab], [-tab, trigamma([b])[0] - tab]])
        step = np.linalg.solve(J, g)
        lam = 1.0
        while a - lam * step[0] <= 0 or b - lam * step[1] <= 0:
            lam /= 2
        a, b = a - lam * step[0], b - lam * step[1]
        if np.max(np.abs(step)) < 1e-10:
            break
    ll = (n * (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b))
          + (a - 1) * np.log(x).sum() + (b - 1) * np.log1p(-x).sum())
    return {"alpha": a, "beta": b}, ll, 2


def fit_poisson(x):
    lam = x.mean()
    ll = (x * math.log(lam) - lam).sum() - _lgamma(x + 1).sum() if lam > 0 else 0.0
    return {"lambda": lam}, ll, 1


def fit_negbin(x):
    n, m = len(x), x.mean()

    def prof_ll(log_r):
        r = math.exp(log_r)
        return (_lgamma(x + r).sum() - n * math.lgamma(r) - _lgamma(x + 1).sum()
                + n * r * math.log(r / (r + m)) + x.sum() * math.log(m / (r + m)))

    # golden-section search over log(r)
    lo, hi = -10.0, 20.0
    gr = (math.sqrt(5) - 1) / 2
    c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
    fc, fd = prof_ll(c), prof_ll(d)
    for _ in range(200):
        if fc > fd:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = prof_ll(c)
        else:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = prof_ll(d)
        if hi - lo < 1e-8:
            break
    log_r = (lo + hi) / 2
    r = math.exp(log_r)
    return {"size_r": r, "mu": m, "p": r / (r + m)}, prof_ll(log_r), 2


FITTERS = {
    "normal": fit_normal, "lognormal": fit_lognormal, "gamma": fit_gamma,
    "beta": fit_beta, "poisson": fit_poisson, "negbin": fit_negbin,
}


# --------------------------------------------------------------------------
# Per-feature analysis
# --------------------------------------------------------------------------
def infer_spec(x):
    """Guess type/candidates for a column not in FEATURE_SPEC."""
    is_int = np.all(np.isclose(x, np.round(x)))
    if is_int and x.min() >= 0:
        return COUNT, "0,1,2,...", ["poisson", "negbin"], "Inferred from data"
    if x.min() >= 0 and x.max() <= 1:
        return PROP, "0-1", ["beta", "normal"], "Inferred from data"
    if x.min() >= 0:
        return POS, ">=0", ["lognormal", "gamma", "normal"], "Inferred from data"
    return CONT, "real line", ["normal"], "Inferred from data"


def skewness(x):
    sd = x.std()
    return float(((x - x.mean()) ** 3).mean() / sd ** 3) if sd > 0 else float("nan")


def analyse_feature(name, series):
    raw = pd.to_numeric(series, errors="coerce")
    x = raw.dropna().to_numpy(dtype=float)

    if name in FEATURE_SPEC:
        vtype, support, cands, note = FEATURE_SPEC[name]
    else:
        vtype, support, cands, note = infer_spec(x) if len(x) else (CONT, "", [], "No data")

    row = {
        "feature": name,
        "variable_type": vtype,
        "support": support,
        "candidate_distributions": " / ".join(DIST_LABEL[c] for c in cands),
        "reference_note": note,
        "n": len(x),
        "n_missing": int(raw.isna().sum()),
    }
    fits = []
    if len(x) < 3:
        row["data_notes"] = "Too few observations to fit"
        return row, fits

    n_zero, n_one = int((x == 0).sum()), int((x == 1).sum())
    row.update({
        "n_zero": n_zero,
        "min": x.min(), "q25": np.percentile(x, 25), "median": np.median(x),
        "mean": x.mean(), "q75": np.percentile(x, 75), "max": x.max(),
        "sd": x.std(ddof=1), "variance": x.var(ddof=1), "skewness": skewness(x),
        "dispersion_index_var_over_mean": x.var(ddof=1) / x.mean() if vtype == COUNT and x.mean() > 0 else np.nan,
    })

    notes = []
    fit_x = x
    if vtype == COUNT:
        if not np.all(np.isclose(x, np.round(x))):
            notes.append("non-integer values present (rounded for count fits)")
        fit_x = np.round(x)
        di = row["dispersion_index_var_over_mean"]
        if di > 1.5:
            notes.append(f"overdispersed (var/mean={di:.1f}) -> Negative Binomial preferred over Poisson")
        elif di < 0.8:
            notes.append(f"underdispersed (var/mean={di:.2f})")
        else:
            notes.append(f"approximately equidispersed (var/mean={di:.2f})")
    elif "beta" in cands:
        if x.min() < 0 or x.max() > 1:
            notes.append(f"{int(((x < 0) | (x > 1)).sum())} values outside [0,1]; Beta fit skipped")
            cands = [c for c in cands if c != "beta"]
        elif n_zero or n_one:
            n = len(x)
            fit_x = (x * (n - 1) + 0.5) / n
            notes.append(f"exact 0 (n={n_zero}) / 1 (n={n_one}) values -> Smithson-Verkuilen squeeze "
                         "applied before fitting; consider zero/one-inflated Beta")
    elif any(c in cands for c in ("lognormal", "gamma")):
        if (x < 0).any():
            notes.append(f"{int((x < 0).sum())} negative values; Log-normal/Gamma fits skipped")
            cands = [c for c in cands if c not in ("lognormal", "gamma")]
        elif n_zero:
            fit_x = x[x > 0]
            notes.append(f"{n_zero} zeros excluded from all fits (Log-normal/Gamma need >0); "
                         "consider a hurdle / zero-inflated model")
    if vtype == POS and "elongation" in name and (x < 1).any():
        notes.append(f"{int((x < 1).sum())} values < 1")
    if row["skewness"] > 1:
        notes.append(f"right-skewed (skew={row['skewness']:.2f})")
    elif abs(row["skewness"]) < 0.5:
        notes.append(f"roughly symmetric (skew={row['skewness']:.2f})")

    # Always compare against a Normal baseline for continuous features
    if vtype != COUNT and "normal" not in cands:
        cands = cands + ["normal"]

    n_fit = len(fit_x)
    for c in cands:
        try:
            params, ll, k = FITTERS[c](fit_x)
        except Exception as exc:  # numerical failure on odd data
            fits.append({"feature": name, "distribution": DIST_LABEL[c], "error": str(exc)})
            continue
        fits.append({
            "feature": name, "distribution": DIST_LABEL[c], "n_fit": n_fit,
            "params": "; ".join(f"{p}={v:.6g}" for p, v in params.items()),
            "loglik": ll, "AIC": 2 * k - 2 * ll, "BIC": k * math.log(n_fit) - 2 * ll,
        })

    ok = [f for f in fits if "AIC" in f]
    if ok:
        best_aic = min(f["AIC"] for f in ok)
        weights = np.exp(-0.5 * np.array([f["AIC"] - best_aic for f in ok]))
        weights /= weights.sum()
        for f, w in zip(ok, weights):
            f["delta_AIC"] = f["AIC"] - best_aic
            f["akaike_weight"] = w
        ranked = sorted(ok, key=lambda f: f["AIC"])
        row["best_fit_by_AIC"] = ranked[0]["distribution"]
        row["best_fit_by_BIC"] = min(ok, key=lambda f: f["BIC"])["distribution"]
        row["best_fit_params"] = ranked[0]["params"]
        row["delta_AIC_to_2nd"] = ranked[1]["AIC"] - ranked[0]["AIC"] if len(ranked) > 1 else np.nan
        for f in ok:
            row[f"AIC_{f['distribution']}"] = f["AIC"]

    row["data_notes"] = "; ".join(notes)
    return row, fits


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def select_columns(df, mode):
    if mode == "spec":
        cols = [c for c in FEATURE_SPEC if c in df.columns]
        missing = [c for c in FEATURE_SPEC if c not in df.columns]
        if missing:
            print(f"  [warn] reference features missing from file: {missing}")
        return cols
    numeric = [c for c in df.columns
               if c not in NON_FEATURE_COLS and pd.to_numeric(df[c], errors="coerce").notna().any()]
    return [c for c in FEATURE_SPEC if c in numeric] + [c for c in numeric if c not in FEATURE_SPEC]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="luad_summary_outputs/cleaned/tumor_core",
                        help="Folder with the per-group CSVs")
    parser.add_argument("--feature", default="tumor_core",
                        help="Feature set name used in file names (default: tumor_core)")
    parser.add_argument("--suffix", default="_{feature}_wsi_clean.csv",
                        help="File-name suffix after the group name; {feature} is substituted. "
                             "Use _{feature}_wsi_summary.csv for the raw WSI summaries.")
    parser.add_argument("--group", nargs="+", default=["all"],
                        help="'all' or one or more group names, e.g. female_psg_negative")
    parser.add_argument("--columns", choices=["spec", "all"], default="spec",
                        help="'spec' = only features in the reference table; "
                             "'all' = every numeric non-ID/clinical column (types inferred)")
    parser.add_argument("--output-dir", default="luad_summary_outputs/distribution_summary",
                        help="Root output folder; one sub-folder per group")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    suffix = args.suffix.format(feature=args.feature)
    # 'all_groups' is the pooled file written by LUAD_clean_tumor_core.py, not a group
    available = {p.name[: -len(suffix)]: p for p in sorted(in_dir.glob(f"*{suffix}"))
                 if not p.name.startswith("all_groups")}
    if not available:
        raise SystemExit(f"No files matching *{suffix} in {in_dir}")

    if [g.lower() for g in args.group] == ["all"]:
        groups = list(available)
    else:
        unknown = [g for g in args.group if g not in available]
        if unknown:
            raise SystemExit(f"Unknown group(s) {unknown}. Available: {list(available)}")
        groups = args.group

    for group in groups:
        path = available[group]
        print(f"Group '{group}': {path.name}")
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        rows, all_fits = [], []
        for col in select_columns(df, args.columns):
            row, fits = analyse_feature(col, df[col])
            rows.append({"group": group, **row})
            all_fits.extend({"group": group, **f} for f in fits)

        out_dir = Path(args.output_dir) / group
        out_dir.mkdir(parents=True, exist_ok=True)
        summary_path = out_dir / f"{group}_{args.feature}_distribution_summary.csv"
        fits_path = out_dir / f"{group}_{args.feature}_distribution_fits.csv"
        pd.DataFrame(rows).to_csv(summary_path, index=False)
        pd.DataFrame(all_fits).to_csv(fits_path, index=False)
        print(f"  wrote {summary_path} ({len(rows)} features)")
        print(f"  wrote {fits_path} ({len(all_fits)} fits)")


if __name__ == "__main__":
    main()
