#!/usr/bin/env python3
"""
Plot the best-fit distributions (by AIC and by BIC) for each feature, per group.

Reads the outputs of LUAD_feature_distribution_summary.py:
    {summary-dir}/{group}/{group}_{feature}_distribution_summary.csv
    {summary-dir}/{group}/{group}_{feature}_distribution_fits.csv   (parameters of every fit)
and the per-group data the fits were made on:
    {data-dir}/{group}_{feature}_wsi_clean.csv

For every feature it draws:
    left  - histogram of the data (density scale) with the fitted PDF/PMF of the
            best-by-AIC distribution (solid) and best-by-BIC distribution (dashed,
            only if it differs from AIC)
    right - empirical CDF vs the fitted CDF(s), to judge fit in the tails

Outputs, one folder per group in --output-dir:
    {group}/{group}_{feature}_best_fit_distributions.pdf   page 1 = overview grid,
                                                           then one page per feature
    {group}/png/{feature}.png                              with --png

Dependencies:
    pip install pandas numpy matplotlib

Examples:
    python LUAD_plot_best_fit_distributions.py --group all
    python LUAD_plot_best_fit_distributions.py --group female_psg_negative --png
"""

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


AIC_COLOR = "#2a78d6"   # blue
BIC_COLOR = "#eb6834"   # orange
HIST_COLOR = "#c9c8c0"  # neutral gray for the data
TEXT_MUTED = "#5f5e57"

COUNT_DISTS = {"Poisson", "Negative Binomial"}

_lgamma = np.vectorize(math.lgamma, otypes=[float])


# --------------------------------------------------------------------------
# Densities (numpy only; parameter names match LUAD_feature_distribution_summary.py)
# --------------------------------------------------------------------------
def density(dist, p, x):
    """PDF for continuous distributions, PMF for counts. x is a numpy array."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    if dist == "Normal":
        mu, s = p["mu"], p["sigma"]
        return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    if dist == "Log-normal":
        m = x > 0
        mu, s = p["meanlog"], p["sdlog"]
        out[m] = np.exp(-0.5 * ((np.log(x[m]) - mu) / s) ** 2) / (x[m] * s * math.sqrt(2 * math.pi))
        return out
    if dist == "Gamma":
        m = x > 0
        k, th = p["shape"], p["scale"]
        out[m] = np.exp((k - 1) * np.log(x[m]) - x[m] / th - k * math.log(th) - math.lgamma(k))
        return out
    if dist == "Beta":
        m = (x > 0) & (x < 1)
        a, b = p["alpha"], p["beta"]
        ln_b = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
        out[m] = np.exp((a - 1) * np.log(x[m]) + (b - 1) * np.log1p(-x[m]) - ln_b)
        return out
    if dist == "Poisson":
        lam = p["lambda"]
        m = x >= 0
        out[m] = np.exp(x[m] * math.log(lam) - lam - _lgamma(x[m] + 1))
        return out
    if dist == "Negative Binomial":
        r, prob = p["size_r"], p["p"]
        m = x >= 0
        out[m] = np.exp(_lgamma(x[m] + r) - math.lgamma(r) - _lgamma(x[m] + 1)
                        + r * math.log(prob) + x[m] * math.log1p(-prob))
        return out
    raise ValueError(f"Unknown distribution: {dist}")


def cdf(dist, p, grid):
    """CDF on a sorted grid: cumulative PMF for counts, trapezoid integral of the PDF otherwise."""
    if dist in COUNT_DISTS:
        k = np.arange(0, int(grid.max()) + 1)
        c = np.cumsum(density(dist, p, k))
        return c[np.clip(np.floor(grid).astype(int), 0, len(c) - 1)]
    n_pre = 2000
    if dist in ("Log-normal", "Gamma", "Beta"):
        # log-spaced from ~0: Gamma with shape < 1 / Beta with alpha < 1 spike at 0
        start = max(grid[0], 1e-12)
        pre = np.geomspace(start * 1e-10, start, n_pre, endpoint=False)
    else:
        pre = np.linspace(min(grid[0], p["mu"] - 10 * p["sigma"]), grid[0], n_pre, endpoint=False)
    # integrate on a dense internal grid, then interpolate at the requested points
    full = np.concatenate([pre, np.linspace(grid[0], grid[-1], 20000)])
    f = density(dist, p, full)
    c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(full))])
    return np.clip(np.interp(grid, full, c), 0, 1)


def parse_params(text):
    out = {}
    for part in str(text).split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = float(v)
    return out


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def best_fits_for_group(summary_path, fits_path):
    """Return {feature: {'aic': (dist, params, row), 'bic': (...)}}."""
    summ = pd.read_csv(summary_path)
    fits = pd.read_csv(fits_path) if fits_path.exists() else None
    if fits is None:
        print(f"  [warn] {fits_path.name} not found: BIC curve only drawn when it matches AIC")

    result = {}
    for _, r in summ.iterrows():
        feat = r["feature"]
        if pd.isna(r.get("best_fit_by_AIC")):
            continue
        entry = {}
        for crit in ("AIC", "BIC"):
            dist = r[f"best_fit_by_{crit}"]
            fit_row = None
            if fits is not None:
                sel = fits[(fits["feature"] == feat) & (fits["distribution"] == dist)]
                if len(sel):
                    fit_row = sel.iloc[0]
            if fit_row is not None:
                params = parse_params(fit_row["params"])
            elif dist == r["best_fit_by_AIC"]:
                params = parse_params(r["best_fit_params"])
            else:
                continue
            entry[crit.lower()] = (dist, params, fit_row)
        result[feat] = {"summary": r, **entry}
    return result


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def curve_label(crit_names, dist, fit_row):
    label = f"{dist} (best {' & '.join(crit_names)})"
    if fit_row is not None:
        label += f"\nAIC={fit_row['AIC']:.1f}, BIC={fit_row['BIC']:.1f}"
    return label


def curves_to_draw(info):
    """[(dist, params, fit_row, color, linestyle, label)], merging AIC/BIC if identical."""
    aic, bic = info.get("aic"), info.get("bic")
    if aic and bic and aic[0] == bic[0]:
        return [(*aic, AIC_COLOR, "-", curve_label(["AIC", "BIC"], aic[0], aic[2]))]
    out = []
    if aic:
        out.append((*aic, AIC_COLOR, "-", curve_label(["AIC"], aic[0], aic[2])))
    if bic:
        out.append((*bic, BIC_COLOR, "--", curve_label(["BIC"], bic[0], bic[2])))
    return out


def draw_density(ax, x, info, compact=False):
    curves = curves_to_draw(info)
    is_count = any(c[0] in COUNT_DISTS for c in curves)
    lo, hi = float(np.min(x)), float(np.max(x))

    if is_count and hi - lo <= 60:
        # small-range counts: empirical proportion per integer + PMF
        vals, cnts = np.unique(np.round(x).astype(int), return_counts=True)
        ax.bar(vals, cnts / len(x), width=0.8, color=HIST_COLOR, label="Observed", zorder=1)
        grid = np.arange(max(0, int(lo) - 2), int(hi) + 3)
        for dist, p, _, color, ls, label in curves:
            y = density(dist, p, grid)
            ax.plot(grid, y, color=color, ls=ls, lw=2, marker="o", ms=4 if compact else 5,
                    label=label, zorder=3)
        ax.set_ylabel("Probability")
    else:
        bins = min(40, max(10, int(math.sqrt(len(x)) * 2)))
        ax.hist(x, bins=bins, density=True, color=HIST_COLOR, edgecolor="white",
                linewidth=0.8, label="Observed", zorder=1)
        pad = 0.05 * (hi - lo)
        g_lo = max(lo - pad, 0.0) if lo >= 0 else lo - pad
        g_hi = hi + pad
        if any(c[0] == "Beta" for c in curves):
            g_lo, g_hi = max(g_lo, 1e-4), min(g_hi, 1 - 1e-4)
        grid = np.linspace(g_lo, g_hi, 500)
        for dist, p, _, color, ls, label in curves:
            # for counts on a wide range, PMF at integer x == density per unit
            y = density(dist, p, np.round(grid) if dist in COUNT_DISTS else grid)
            ax.plot(grid, y, color=color, ls=ls, lw=2, label=label, zorder=3)
        ax.set_ylabel("Density")

    ax.set_xlabel("Value")
    _style(ax)
    if not compact:
        ax.legend(frameon=False, fontsize=8, loc="upper right")


def draw_cdf(ax, x, info):
    xs = np.sort(x)
    ecdf = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ecdf, where="post", color="#3a3a36", lw=1.5, label="Empirical CDF", zorder=2)
    grid = np.linspace(xs[0], xs[-1], 500)
    if any(c[0] == "Beta" for c in curves_to_draw(info)):
        grid = np.clip(grid, 1e-6, 1 - 1e-6)
    for dist, p, _, color, ls, _label in curves_to_draw(info):
        crit = "AIC & BIC" if color == AIC_COLOR and ls == "-" and info.get("bic") and \
            info.get("aic") and info["aic"][0] == info["bic"][0] else ("AIC" if color == AIC_COLOR else "BIC")
        ax.plot(grid, cdf(dist, p, grid), color=color, ls=ls, lw=2,
                label=f"{dist} (best {crit})", zorder=3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Value")
    ax.set_ylabel("Cumulative probability")
    _style(ax)
    ax.legend(frameon=False, fontsize=8, loc="lower right")


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#b5b4ac")
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.grid(axis="y", color="#e6e5df", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


def feature_page(plt, group, feat, x, info):
    s = info["summary"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.69, 5.2))
    draw_density(ax1, x, info)
    draw_cdf(ax2, x, info)
    ax1.set_title("Observed distribution vs fitted PDF/PMF", fontsize=10, loc="left")
    ax2.set_title("Empirical vs fitted CDF", fontsize=10, loc="left")
    fig.suptitle(feat, fontsize=13, x=0.01, ha="left", y=0.99)

    gap = s.get("delta_AIC_to_2nd")
    gap_txt = f"   |   ΔAIC to 2nd best = {gap:.2f}" if pd.notna(gap) else ""
    if pd.notna(gap) and gap < 2:
        gap_txt += " (<2: candidates essentially tied)"
    sub = (f"{group}   |   n = {int(s['n'])}   |   {s['variable_type']}   |   "
           f"best AIC: {s['best_fit_by_AIC']}   |   best BIC: {s['best_fit_by_BIC']}{gap_txt}")
    fig.text(0.01, 0.925, sub, fontsize=8.5, color=TEXT_MUTED)

    params_lines = [f"{c[0]}: {c[2]['params'] if c[2] is not None else ''}" for c in curves_to_draw(info)]
    notes = s.get("data_notes")
    foot = "Fitted parameters - " + " | ".join(params_lines)
    if isinstance(notes, str) and notes:
        foot += f"\nNotes: {notes}"
    fig.text(0.01, 0.01, foot, fontsize=7.5, color=TEXT_MUTED, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    return fig


def overview_page(plt, group, items):
    n = len(items)
    ncols = 4
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.5, 3.4 * nrows + 1))
    axes = np.atleast_1d(axes).ravel()
    for ax, (feat, x, info) in zip(axes, items):
        draw_density(ax, x, info, compact=True)
        title = feat.replace("wsi_area_weighted_", "aw_").replace("wsi_", "")
        ax.set_title(title, fontsize=8.5, loc="left")
        best = curves_to_draw(info)
        ax.text(0.98, 0.95, "\n".join(c[0] + (" (AIC)" if c[3] == AIC_COLOR and c[4] == "-"
                                               and len(best) > 1 else "") for c in best),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=TEXT_MUTED)
        ax.set_xlabel("")
        ax.set_ylabel("")
    for ax in axes[n:]:
        ax.set_visible(False)

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [Patch(color=HIST_COLOR, label="Observed"),
               Line2D([], [], color=AIC_COLOR, lw=2, label="Best by AIC (and BIC if same)"),
               Line2D([], [], color=BIC_COLOR, lw=2, ls="--", label="Best by BIC (when different)")]
    fig.legend(handles=handles, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"{group}: best-fit distributions by AIC / BIC", fontsize=14, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--summary-dir", default="luad_summary_outputs/distribution_summary",
                        help="Output folder of LUAD_feature_distribution_summary.py")
    parser.add_argument("--data-dir", default="luad_summary_outputs/cleaned/tumor_core",
                        help="Folder with the per-group data CSVs the fits were made on")
    parser.add_argument("--feature", default="tumor_core", help="Feature set name in file names")
    parser.add_argument("--data-suffix", default="_{feature}_wsi_clean.csv",
                        help="Data file suffix after the group name; {feature} is substituted")
    parser.add_argument("--group", nargs="+", default=["all"],
                        help="'all' or one or more group names")
    parser.add_argument("--output-dir", default="luad_summary_outputs/distribution_plots",
                        help="Root output folder; one sub-folder per group")
    parser.add_argument("--png", action="store_true", help="Also save one PNG per feature")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    summary_dir = Path(args.summary_dir)
    summ_name = f"_{args.feature}_distribution_summary.csv"
    available = {p.parent.name: p for p in sorted(summary_dir.glob(f"*/*{summ_name}"))}
    if not available:
        raise SystemExit(f"No */*{summ_name} files in {summary_dir}")

    if [g.lower() for g in args.group] == ["all"]:
        groups = list(available)
    else:
        unknown = [g for g in args.group if g not in available]
        if unknown:
            raise SystemExit(f"Unknown group(s) {unknown}. Available: {list(available)}")
        groups = args.group

    data_suffix = args.data_suffix.format(feature=args.feature)
    for group in groups:
        summary_path = available[group]
        fits_path = summary_path.with_name(f"{group}_{args.feature}_distribution_fits.csv")
        data_path = Path(args.data_dir) / f"{group}{data_suffix}"
        if not data_path.exists():
            print(f"[skip] {group}: data file {data_path} not found")
            continue
        print(f"Group '{group}'")

        best = best_fits_for_group(summary_path, fits_path)
        df = pd.read_csv(data_path, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()

        items = []
        for feat, info in best.items():
            if feat not in df.columns:
                print(f"  [warn] {feat} not in data file, skipped")
                continue
            x = pd.to_numeric(df[feat], errors="coerce").dropna().to_numpy(dtype=float)
            if len(x) < 3:
                continue
            items.append((feat, x, info))

        out_dir = Path(args.output_dir) / group
        out_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = out_dir / f"{group}_{args.feature}_best_fit_distributions.pdf"
        with PdfPages(pdf_path) as pdf:
            fig = overview_page(plt, group, items)
            pdf.savefig(fig)
            plt.close(fig)
            for feat, x, info in items:
                fig = feature_page(plt, group, feat, x, info)
                pdf.savefig(fig)
                if args.png:
                    (out_dir / "png").mkdir(exist_ok=True)
                    fig.savefig(out_dir / "png" / f"{feat}.png", dpi=200)
                plt.close(fig)
        print(f"  wrote {pdf_path} ({len(items)} features + overview page)")


if __name__ == "__main__":
    main()
