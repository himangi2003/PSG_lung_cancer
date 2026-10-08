#!/usr/bin/env python3
"""
Univariate Cox models on subject-level (patient-level) tumor_core features.

Steps:
  1. Read the combined transformed table (LUAD_combine_transformed.py).
  2. Aggregate slides to one row per subject_id: the MEAN of each '_z' feature
     (or --agg median). Survival / clinical columns must agree within a subject;
     disagreements are reported. 'n_slides' records how many slides were averaged.
  3. Re-standardise the averaged features across subjects (mean 0, SD 1), so
     HRs are "per 1 SD of the subject-level feature" (turn off with --no-restandardize).
  4. For each feature, fit Cox models of OS_MONTHS / OS_Status:
        - unadjusted:      feature
        - stage_adjusted:  feature + stage (I / II / III-IV dummies, ref = I)
     in every scope: 'pooled' (all groups) and, with --by-group, each group alone.
  5. Report HR per SD with 95% CI, Wald and likelihood-ratio p-values,
     Benjamini-Hochberg q-values (within each scope x model), Harrell's C-index,
     and a proportional-hazards check (Grambsch-Therneau test on rank(time)).
  6. Pooled only: does the effect differ by group?  LRT of
        feature + group + feature:group  vs  feature + group.

The Cox model is implemented here with numpy (Efron ties, Newton-Raphson), so
no lifelines / statsmodels / scipy is needed. Results match R survival::coxph
and lifelines (both use Efron ties by default).

Outputs in --output-dir:
    subject_level_{feature}_features.csv     the aggregated subject table used for the models
    cox_univariate_{feature}_results.csv     one row per scope x model x feature
    cox_group_interaction_{feature}.csv      pooled feature x group interaction tests

Dependencies:
    pip install pandas numpy

Examples:
    python LUAD_cox_subject_level.py
    python LUAD_cox_subject_level.py --by-group
    python LUAD_cox_subject_level.py --group male_psg_negative male_psg_positive --censor-at 60
"""

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


SUBJECT_COLS = ["OS_MONTHS", "OS_Status", "stage", "SEX", "group"]


# --------------------------------------------------------------------------
# Distributions
# --------------------------------------------------------------------------
def norm_sf_two_sided(z):
    return math.erfc(abs(z) / math.sqrt(2))


def chi2_sf(x, df):
    """Survival function of chi-square with integer df (closed form)."""
    if x <= 0:
        return 1.0
    if df % 2 == 0:
        term = total = math.exp(-x / 2)
        for i in range(1, df // 2):
            term *= (x / 2) / i
            total += term
        return min(total, 1.0)
    total = math.erfc(math.sqrt(x / 2))
    term = math.sqrt(2 * x / math.pi) * math.exp(-x / 2)  # i = 1
    for i in range(1, (df - 1) // 2 + 1):
        total += term
        term *= x / (2 * i + 1)
    return min(total, 1.0)


def bh_qvalues(p):
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    order = np.argsort(pv)
    ranked = pv[order] * len(pv) / np.arange(1, len(pv) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(pv)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


# --------------------------------------------------------------------------
# Cox proportional hazards (Efron ties)
# --------------------------------------------------------------------------
def _efron(beta, X, time, event, event_times):
    eta = X @ beta
    w = np.exp(eta - eta.max())  # rescaling cancels in the ratios
    shift = eta.max()
    p = X.shape[1]
    ll, grad, hess = 0.0, np.zeros(p), np.zeros((p, p))
    for t in event_times:
        risk = time >= t
        dead = risk & (time == t) & (event == 1)
        d = int(dead.sum())
        wr, wd = w[risk], w[dead]
        Xr, Xd = X[risk], X[dead]
        S0, S1, S2 = wr.sum(), wr @ Xr, (Xr * wr[:, None]).T @ Xr
        D0, D1, D2 = wd.sum(), wd @ Xd, (Xd * wd[:, None]).T @ Xd
        ll += eta[dead].sum()
        grad += Xd.sum(axis=0)
        for k in range(d):
            f = k / d
            s0 = S0 - f * D0
            s1 = S1 - f * D1
            s2 = S2 - f * D2
            ll -= math.log(s0) + shift
            grad -= s1 / s0
            hess -= s2 / s0 - np.outer(s1, s1) / s0 ** 2
    return ll, grad, hess


def fit_cox(X, time, event, max_iter=50, tol=1e-9):
    """Return dict(beta, se, loglik, loglik0, cov, converged). X: n x p array."""
    X = np.asarray(X, dtype=float)
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    event_times = np.unique(time[event == 1])
    beta = np.zeros(X.shape[1])
    ll0, _, _ = _efron(beta, X, time, event, event_times)
    ll = ll0
    converged = False
    for _ in range(max_iter):
        ll, grad, hess = _efron(beta, X, time, event, event_times)
        step = np.linalg.solve(hess, -grad)
        lam = 1.0
        while True:  # step halving if the likelihood goes down
            new = beta + lam * step
            ll_new = _efron(new, X, time, event, event_times)[0]
            if ll_new >= ll - 1e-12 or lam < 1e-6:
                break
            lam /= 2
        beta = new
        if abs(ll_new - ll) < tol and np.max(np.abs(lam * step)) < 1e-6:
            ll = ll_new
            converged = True
            break
        ll = ll_new
    ll, grad, hess = _efron(beta, X, time, event, event_times)
    cov = np.linalg.inv(-hess)
    return {"beta": beta, "se": np.sqrt(np.diag(cov)), "cov": cov,
            "loglik": ll, "loglik0": ll0, "converged": converged}


def ph_test(x, beta, time, event):
    """Grambsch-Therneau test for one covariate, g(t) = rank of event time.
    Uses Breslow-style Schoenfeld residuals; adequate as a diagnostic."""
    w = np.exp(beta * x)
    ev = np.where(event == 1)[0]
    resid, g, info = [], [], 0.0
    for i in ev:
        risk = time >= time[i]
        s0 = w[risk].sum()
        xbar = (w[risk] * x[risk]).sum() / s0
        x2bar = (w[risk] * x[risk] ** 2).sum() / s0
        resid.append(x[i] - xbar)
        info += x2bar - xbar ** 2
        g.append(time[i])
    resid, g = np.array(resid), pd.Series(g).rank().to_numpy()
    d = len(ev)
    gc = g - g.mean()
    if d < 3 or info <= 0 or (gc ** 2).sum() == 0:
        return np.nan
    stat = (d / info) * (gc @ resid) ** 2 / (gc ** 2).sum()
    return chi2_sf(stat, 1)


def harrell_c(risk_score, time, event):
    conc = ties = total = 0.0
    for i in np.where(event == 1)[0]:
        comp = time > time[i]
        n = comp.sum()
        if n == 0:
            continue
        total += n
        conc += (risk_score[i] > risk_score[comp]).sum()
        ties += (risk_score[i] == risk_score[comp]).sum()
    return (conc + 0.5 * ties) / total if total else np.nan


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------
def parse_status(s):
    """0/1, or strings like '1:DECEASED' / 'LIVING'."""
    if pd.isna(s):
        return np.nan
    txt = str(s).strip().upper()
    if txt[:1] in ("0", "1"):
        return int(txt[0])
    if "DECEASED" in txt or "DEAD" in txt:
        return 1
    if "LIVING" in txt or "ALIVE" in txt:
        return 0
    return np.nan


def parse_stage(s):
    """'Stage IIB' / 'IIIA' / 'II' -> 'I', 'II', 'III-IV' (III and IV merged)."""
    if pd.isna(s):
        return np.nan
    m = re.search(r"\b(?:STAGE\s*)?(IV|III|II|I)", str(s).upper())
    if not m:
        return np.nan
    return "III-IV" if m.group(1) in ("III", "IV") else m.group(1)


def aggregate_subjects(df, features, agg):
    def first_consistent(col):
        vals = df.groupby("subject_id")[col].nunique(dropna=True)
        bad = vals[vals > 1]
        if len(bad):
            print(f"  [warn] {len(bad)} subject(s) have conflicting '{col}' across slides; "
                  f"first value kept: {bad.index.tolist()[:5]}")
        return df.groupby("subject_id")[col].first()

    feat = df.groupby("subject_id")[features].agg(agg)
    meta = pd.DataFrame({c: first_consistent(c) for c in SUBJECT_COLS if c in df.columns})
    meta["n_slides"] = df.groupby("subject_id").size()
    return meta.join(feat).reset_index()


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def stage_dummies(stage):
    return pd.DataFrame({"stage_II": (stage == "II").astype(float),
                         "stage_III_IV": (stage == "III-IV").astype(float)}, index=stage.index)


def run_feature(sub, feat, model):
    cols = ["time", "event", feat] + (["stage_cat"] if model == "stage_adjusted" else [])
    d = sub[cols].dropna()
    x = d[feat].to_numpy(float)
    X = x[:, None]
    if model == "stage_adjusted":
        X = np.column_stack([x, stage_dummies(d["stage_cat"]).to_numpy()])
        X = X[:, np.r_[True, X[:, 1:].std(axis=0) > 0]]  # drop empty stage levels
    t, e = d["time"].to_numpy(float), d["event"].to_numpy(int)
    row = {"feature": feat, "model": model, "n": len(d), "events": int(e.sum())}
    if e.sum() < 2 or x.std() == 0:
        row["note"] = "too few events or constant feature"
        return row
    try:
        fit = fit_cox(X, t, e)
    except np.linalg.LinAlgError:
        row["note"] = "model failed (singular information matrix)"
        return row

    b, se = fit["beta"][0], fit["se"][0]
    if model == "stage_adjusted":  # LRT for the feature, given stage
        base = fit_cox(X[:, 1:], t, e)["loglik"]
    else:
        base = fit["loglik0"]
    lrt = 2 * (fit["loglik"] - base)
    row.update({
        "beta": b, "se": se,
        "HR_per_SD": math.exp(b),
        "HR_CI_low": math.exp(b - 1.96 * se),
        "HR_CI_high": math.exp(b + 1.96 * se),
        "p_wald": norm_sf_two_sided(b / se),
        "p_LRT": chi2_sf(lrt, 1),
        "C_index": harrell_c(X @ fit["beta"], t, e),
        "PH_test_p": ph_test(x, b, t, e) if model == "unadjusted" else np.nan,
        "converged": fit["converged"],
    })
    events_per_var = e.sum() / X.shape[1]
    if events_per_var < 10:
        row["note"] = f"low events per variable ({events_per_var:.1f})"
    return row


def interaction_test(sub, feat, groups):
    d = sub[["time", "event", feat, "group"]].dropna()
    d = d[d["group"].isin(groups)]
    x = d[feat].to_numpy(float)
    t, e = d["time"].to_numpy(float), d["event"].to_numpy(int)
    ref = groups[0]
    G = np.column_stack([(d["group"] == g).astype(float) for g in groups[1:]])
    XG = np.column_stack([x, G])
    XI = np.column_stack([XG, G * x[:, None]])
    row = {"feature": feat, "reference_group": ref, "n": len(d), "events": int(e.sum())}
    try:
        f0, f1 = fit_cox(XG, t, e), fit_cox(XI, t, e)
    except np.linalg.LinAlgError:
        row["note"] = "model failed"
        return row
    lrt = 2 * (f1["loglik"] - f0["loglik"])
    row.update({"LRT_stat": lrt, "df": len(groups) - 1, "p_interaction": chi2_sf(lrt, len(groups) - 1)})
    # group-specific HR per SD from the interaction model
    for j, g in enumerate(groups):
        if j == 0:
            b, var = f1["beta"][0], f1["cov"][0, 0]
        else:
            k = 1 + (len(groups) - 1) + (j - 1)
            b = f1["beta"][0] + f1["beta"][k]
            var = f1["cov"][0, 0] + f1["cov"][k, k] + 2 * f1["cov"][0, k]
        row[f"HR_per_SD_{g}"] = math.exp(b)
        row[f"HR_CI_{g}"] = f"{math.exp(b - 1.96 * math.sqrt(var)):.3f}-{math.exp(b + 1.96 * math.sqrt(var)):.3f}"
    return row


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="luad_summary_outputs/transformed/tumor_core/"
                                            "all_groups_tumor_core_transformed.csv",
                        help="Combined table from LUAD_combine_transformed.py")
    parser.add_argument("--feature", default="tumor_core", help="Feature set name for output names")
    parser.add_argument("--group", nargs="+", default=["all"], help="'all' or group names to include")
    parser.add_argument("--by-group", action="store_true",
                        help="Also fit every model within each group separately")
    parser.add_argument("--agg", choices=["mean", "median"], default="mean",
                        help="How to combine slides of the same subject (default mean)")
    parser.add_argument("--no-restandardize", action="store_true",
                        help="Keep averaged features as they are instead of re-z-scoring across subjects")
    parser.add_argument("--censor-at", type=float, default=None,
                        help="Administratively censor follow-up at this many months (sensitivity analysis)")
    parser.add_argument("--output-dir", default="luad_summary_outputs/cox/tumor_core",
                        help="Folder for the result CSVs")
    args = parser.parse_args()

    df = pd.read_csv(args.input, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()
    if "group" not in df.columns:
        df["group"] = "all"
    available = list(dict.fromkeys(df["group"]))
    if [g.lower() for g in args.group] == ["all"]:
        groups = available
    else:
        unknown = [g for g in args.group if g not in available]
        if unknown:
            raise SystemExit(f"Unknown group(s) {unknown}. Available: {available}")
        groups = args.group
        df = df[df["group"].isin(groups)]

    features = [c for c in df.columns if c.endswith("_z")]
    if not features:
        raise SystemExit("No '_z' feature columns found")
    print(f"{len(df)} slides, {df['subject_id'].nunique()} subjects, groups: {groups}")

    sub = aggregate_subjects(df, features, args.agg)
    sub["time"] = pd.to_numeric(sub["OS_MONTHS"], errors="coerce")
    sub["event"] = sub["OS_Status"].map(parse_status)
    sub["stage_cat"] = sub["stage"].map(parse_stage) if "stage" in sub else np.nan
    # time 0 is kept (as in R survival::coxph / lifelines); missing or negative is dropped
    bad = sub["time"].isna() | sub["event"].isna() | (sub["time"] < 0)
    if bad.any():
        print(f"  [warn] {int(bad.sum())} subject(s) without usable OS time/status dropped: "
              f"{sub.loc[bad, 'subject_id'].tolist()[:10]}")
        sub = sub[~bad].copy()
    n_zero = int((sub["time"] == 0).sum())
    if n_zero:
        print(f"  note: {n_zero} subject(s) have OS_MONTHS = 0 (kept)")
    if args.censor_at:
        late = sub["time"] > args.censor_at
        sub.loc[late, "event"] = 0
        sub.loc[late, "time"] = args.censor_at
        print(f"  administratively censored {int(late.sum())} subject(s) at {args.censor_at} months")
    if not args.no_restandardize:
        sub[features] = (sub[features] - sub[features].mean()) / sub[features].std(ddof=1)

    print(f"Subject-level: {len(sub)} subjects, {int(sub['event'].sum())} events, "
          f"slides per subject: {sub['n_slides'].value_counts().sort_index().to_dict()}")
    print(f"  stage: {sub['stage_cat'].value_counts(dropna=False).to_dict()}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub.drop(columns=["time", "event"]).to_csv(
        out_dir / f"subject_level_{args.feature}_features.csv", index=False)

    scopes = [("pooled", sub)]
    if args.by_group and len(groups) > 1:
        scopes += [(g, sub[sub["group"] == g]) for g in groups]

    rows = []
    for scope, data in scopes:
        for model in ("unadjusted", "stage_adjusted"):
            block = [run_feature(data, f, model) for f in features]
            q = bh_qvalues([r.get("p_LRT", np.nan) for r in block])
            for r, qv in zip(block, q):
                r["q_BH"] = qv
                rows.append({"scope": scope, **r})
        print(f"  fitted scope '{scope}' ({len(data)} subjects, {int(data['event'].sum())} events)")

    res = pd.DataFrame(rows)
    res["feature"] = res["feature"].str.replace(r"__.*_z$", "", regex=True)
    first = ["scope", "model", "feature", "n", "events", "HR_per_SD", "HR_CI_low", "HR_CI_high",
             "p_LRT", "q_BH", "p_wald", "C_index", "PH_test_p"]
    res = res[[c for c in first if c in res] + [c for c in res if c not in first]]
    res_path = out_dir / f"cox_univariate_{args.feature}_results.csv"
    res.to_csv(res_path, index=False)
    print(f"Wrote {res_path}")

    if len(groups) > 1:
        inter = pd.DataFrame([interaction_test(sub, f, groups) for f in features])
        inter["feature"] = inter["feature"].str.replace(r"__.*_z$", "", regex=True)
        inter["q_BH"] = bh_qvalues(inter.get("p_interaction", pd.Series(np.nan, index=inter.index)))
        inter_path = out_dir / f"cox_group_interaction_{args.feature}.csv"
        inter.to_csv(inter_path, index=False)
        print(f"Wrote {inter_path}")

    top = res[(res["scope"] == "pooled")].sort_values(["model", "p_LRT"])
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("\nPooled results (HR per 1 SD):")
        print(top[["model", "feature", "HR_per_SD", "HR_CI_low", "HR_CI_high", "p_LRT", "q_BH",
                   "C_index", "PH_test_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
