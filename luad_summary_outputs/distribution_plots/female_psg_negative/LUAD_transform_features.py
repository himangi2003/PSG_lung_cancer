#!/usr/bin/env python3
"""
Transform WSI-level features toward symmetry, then z-score them, per group.

The default transform for each feature (TRANSFORMS below) was chosen on the
pooled data of all groups, so every group gets the SAME transform and the
transformed values stay comparable across groups. Z-scoring uses the pooled
mean/SD of all groups by default (--zscore pooled), so z = 0 means "average
slide across all groups", not "average slide within this group".

Input (one file per group, output of LUAD_clean_tumor_core.py):
    {data-dir}/{group}_{feature}_wsi_clean.csv

Outputs in --output-dir:
    {group}/{group}_{feature}_transformed.csv   IDs + clinical + raw features (unless --drop-raw)
                                                + one column per feature: {feature}__{method}_z
                                                  ({feature}__{method} with --zscore none)
    {feature}_transform_report.csv               method, parameters, z-score reference, and
                                                 skewness before/after per group and pooled

Methods:
    none, log, log1p, sqrt, cbrt, logit, neg_reciprocal (-1/x, keeps direction),
    reflected_log (-log(c - x), keeps direction; c from REFLECT_CONSTANT), boxcox
    (power estimated on the pooled data).

Dependencies:
    pip install pandas numpy

Examples:
    python LUAD_transform_features.py --group all
    python LUAD_transform_features.py --group female_psg_negative
    python LUAD_transform_features.py --group all --set wsi_tumor_area_mm2=log --zscore group
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# Default transform per feature (skewness on pooled data: raw -> transformed)
TRANSFORMS = {
    "n_clusters_with_tumor": "log1p",                                   # 2.20 -> 0.12
    "wsi_tumor_n_islands": "cbrt",                                      # 1.65 -> 0.09
    "wsi_cluster_area_mm2": "cbrt",                                     # 1.19 -> -0.08
    "wsi_tumor_area_mm2": "cbrt",                                       # 1.86 -> 0.22
    "wsi_tumor_perimeter_mm": "cbrt",                                   # 1.47 -> 0.04
    "wsi_tumor_fraction_of_cluster": "logit",                           # 0.49 -> 0.40
    "wsi_tumor_boundary_density_per_mm": "cbrt",                        # 1.88 -> 0.28
    "wsi_tumor_patch_density_per_mm2": "sqrt",                          # 1.05 -> 0.11
    "wsi_area_weighted_tumor_largest_patch_index": "sqrt",              # 0.95 -> 0.33
    "wsi_area_weighted_tumor_compactness_mean": "sqrt",                 # 0.83 -> 0.13
    "wsi_area_weighted_tumor_solidity_mean": "none",                    # 0.34
    "wsi_area_weighted_tumor_elongation_mean": "log",                   # 1.37 -> 0.27
    "wsi_area_weighted_tumor_boundary_fractal_dimension": "reflected_log",  # -1.16 -> -0.34
    "wsi_area_weighted_tumor_island_nnd_median_um": "neg_reciprocal",   # 4.08 -> -0.04
    "wsi_area_weighted_tumor_island_gap_median_um": "cbrt",             # 1.70 -> 0.01
}

# Upper bound used by reflected_log: -log(c - x). Boundary fractal dimension is < 2.
REFLECT_CONSTANT = {
    "wsi_area_weighted_tumor_boundary_fractal_dimension": 2.0,
}

METHODS = ["none", "log", "log1p", "sqrt", "cbrt", "logit",
           "neg_reciprocal", "reflected_log", "boxcox"]


# --------------------------------------------------------------------------
# Transforms
# --------------------------------------------------------------------------
def boxcox(x, lam):
    return np.log(x) if abs(lam) < 1e-9 else (x ** lam - 1) / lam


def boxcox_lambda(x):
    """Profile-likelihood Box-Cox power on a grid in [-2, 2]."""
    lams = np.linspace(-2, 2, 801)
    n, slx = len(x), np.log(x).sum()
    ll = [-n / 2 * np.log(boxcox(x, l).var()) + (l - 1) * slx for l in lams]
    return float(lams[int(np.argmax(ll))])


def fit_params(feature, method, pooled):
    """Parameters estimated once on the pooled data (so all groups share them)."""
    x = pooled[np.isfinite(pooled)]
    if method == "boxcox":
        if (x <= 0).any():
            raise ValueError("boxcox needs all values > 0")
        return {"lambda": boxcox_lambda(x)}
    if method == "reflected_log":
        c = REFLECT_CONSTANT.get(feature)
        if c is None:
            c = float(x.max()) * 1.05  # fallback: just above the observed maximum
        if (x >= c).any():
            raise ValueError(f"reflected_log needs all values < {c}")
        return {"c": c}
    if method == "logit":
        if ((x < 0) | (x > 1)).any():
            raise ValueError("logit needs values in [0, 1]")
        # Smithson-Verkuilen squeeze only when exact 0/1 values exist
        return {"squeeze_n": len(x)} if ((x == 0) | (x == 1)).any() else {}
    return {}


def apply_transform(x, method, params):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if method == "none":
            y = x.copy()
        elif method == "log":
            y = np.where(x > 0, np.log(x), np.nan)
        elif method == "log1p":
            y = np.where(x > -1, np.log1p(x), np.nan)
        elif method == "sqrt":
            y = np.where(x >= 0, np.sqrt(x), np.nan)
        elif method == "cbrt":
            y = np.cbrt(x)
        elif method == "logit":
            p = x
            if "squeeze_n" in params:
                n = params["squeeze_n"]
                p = (x * (n - 1) + 0.5) / n
            y = np.where((p > 0) & (p < 1), np.log(p / (1 - p)), np.nan)
        elif method == "neg_reciprocal":
            y = np.where(x > 0, -1.0 / x, np.nan)
        elif method == "reflected_log":
            c = params["c"]
            y = np.where(x < c, -np.log(c - x), np.nan)
        elif method == "boxcox":
            y = np.where(x > 0, boxcox(np.where(x > 0, x, 1.0), params["lambda"]), np.nan)
        else:
            raise ValueError(f"Unknown method {method}")
    return y


def skewness(x):
    x = x[np.isfinite(x)]
    if len(x) < 3 or x.std() == 0:
        return np.nan
    return float(((x - x.mean()) ** 3).mean() / x.std() ** 3)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="luad_summary_outputs/cleaned/tumor_core",
                        help="Folder with the per-group cleaned CSVs")
    parser.add_argument("--feature", default="tumor_core", help="Feature set name in file names")
    parser.add_argument("--suffix", default="_{feature}_wsi_clean.csv",
                        help="File-name suffix after the group name; {feature} is substituted")
    parser.add_argument("--group", nargs="+", default=["all"],
                        help="'all' or one or more group names to write")
    parser.add_argument("--set", nargs="+", default=[], metavar="FEATURE=METHOD",
                        help=f"Override a transform, e.g. wsi_tumor_area_mm2=log. Methods: {', '.join(METHODS)}")
    parser.add_argument("--zscore", choices=["pooled", "group", "none"], default="pooled",
                        help="pooled = mean/SD over all groups (default, keeps groups comparable); "
                             "group = within each group; none = no z-scoring")
    parser.add_argument("--drop-raw", action="store_true",
                        help="Drop the untransformed feature columns from the output")
    parser.add_argument("--output-dir", default="luad_summary_outputs/transformed/tumor_core",
                        help="Root output folder; one sub-folder per group")
    args = parser.parse_args()

    transforms = dict(TRANSFORMS)
    for item in args.set:
        feat, _, method = item.partition("=")
        if method not in METHODS:
            raise SystemExit(f"--set {item}: unknown method '{method}'. Choose from {METHODS}")
        transforms[feat] = method

    in_dir = Path(args.data_dir)
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

    # Load ALL available groups: transform parameters and pooled z-scores always
    # use every group, even when only some groups are written out.
    data = {}
    for g, path in available.items():
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        data[g] = df

    features = [f for f in transforms if all(f in df.columns for df in data.values())]
    skipped = [f for f in transforms if f not in features]
    if skipped:
        print(f"[warn] features missing from at least one group, skipped: {skipped}")
    print(f"Pooled reference: {list(available)} ({sum(len(d) for d in data.values())} slides)")

    report, transformed = [], {g: {} for g in available}
    for feat in features:
        method = transforms[feat]
        raw = {g: pd.to_numeric(df[feat], errors="coerce").to_numpy(dtype=float) for g, df in data.items()}
        pooled_raw = np.concatenate(list(raw.values()))
        params = fit_params(feat, method, pooled_raw)

        t = {g: apply_transform(v, method, params) for g, v in raw.items()}
        pooled_t = np.concatenate(list(t.values()))
        n_invalid = int((np.isfinite(pooled_raw) & ~np.isfinite(pooled_t)).sum())
        if n_invalid:
            print(f"  [warn] {feat}: {n_invalid} values invalid for '{method}' -> set to NaN")

        rec = {"feature": feat, "method": method,
               "params": "; ".join(f"{k}={v:.6g}" for k, v in params.items()),
               "n_invalid_to_nan": n_invalid,
               "skew_raw_pooled": skewness(pooled_raw),
               "skew_transformed_pooled": skewness(pooled_t)}

        if args.zscore == "pooled":
            mu, sd = np.nanmean(pooled_t), np.nanstd(pooled_t, ddof=1)
            rec.update({"z_mean": mu, "z_sd": sd})
        for g, v in t.items():
            if args.zscore == "group":
                mu, sd = np.nanmean(v), np.nanstd(v, ddof=1)
                rec.update({f"z_mean_{g}": mu, f"z_sd_{g}": sd})
            out = (v - mu) / sd if args.zscore != "none" else v
            col = f"{feat}__{method}" + ("_z" if args.zscore != "none" else "")
            transformed[g][col] = out
            rec[f"skew_raw_{g}"] = skewness(raw[g])
            rec[f"skew_transformed_{g}"] = skewness(v)
        report.append(rec)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    for g in groups:
        df = data[g]
        if args.drop_raw:
            df = df.drop(columns=features)
        out = pd.concat([df.reset_index(drop=True), pd.DataFrame(transformed[g])], axis=1)
        out_dir = out_root / g
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{g}_{args.feature}_transformed.csv"
        out.to_csv(path, index=False)
        print(f"Group '{g}': wrote {path} ({len(out)} rows, {len(transformed[g])} transformed features)")

    report_path = out_root / f"{args.feature}_transform_report.csv"
    pd.DataFrame(report).to_csv(report_path, index=False)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
