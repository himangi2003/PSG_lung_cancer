#!/usr/bin/env python3
"""
Combine the per-group transformed feature files into one table.

Input (output of LUAD_transform_features.py):
    {input-dir}/{group}/{group}_{feature}_transformed.csv

Output:
    {output}  default: {input-dir}/all_groups_{feature}_transformed.csv
              all rows from the selected groups, with a 'group' column first.

The transformed '_z' columns were z-scored on the pooled data of all groups
(LUAD_transform_features.py --zscore pooled), so they are already comparable
across groups and can be stacked directly.

Dependencies:
    pip install pandas

Examples:
    python LUAD_combine_transformed.py
    python LUAD_combine_transformed.py --group female_psg_negative male_psg_negative
"""

import argparse
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="luad_summary_outputs/transformed/tumor_core",
                        help="Output folder of LUAD_transform_features.py (one sub-folder per group)")
    parser.add_argument("--feature", default="tumor_core", help="Feature set name in file names")
    parser.add_argument("--group", nargs="+", default=["all"],
                        help="'all' or one or more group names to combine")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: {input-dir}/all_groups_{feature}_transformed.csv)")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    name = f"_{args.feature}_transformed.csv"
    available = {p.parent.name: p for p in sorted(in_dir.glob(f"*/*{name}"))
                 if p.name == f"{p.parent.name}{name}"}
    if not available:
        raise SystemExit(f"No */{{group}}{name} files in {in_dir}")

    if [g.lower() for g in args.group] == ["all"]:
        groups = list(available)
    else:
        unknown = [g for g in args.group if g not in available]
        if unknown:
            raise SystemExit(f"Unknown group(s) {unknown}. Available: {list(available)}")
        groups = args.group

    frames = []
    for g in groups:
        df = pd.read_csv(available[g], encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        print(f"Group '{g}': {len(df)} rows, {df['subject_id'].nunique()} subjects")
        frames.append(df.assign(group=g))

    # Columns should match across groups; report any that don't
    all_cols = set().union(*(f.columns for f in frames))
    for g, f in zip(groups, frames):
        missing = sorted(all_cols - set(f.columns))
        if missing:
            print(f"  [warn] '{g}' is missing columns (filled with NaN): {missing}")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[["group"] + [c for c in frames[0].columns if c != "group"]
                        + [c for c in combined.columns if c not in frames[0].columns]]

    if "SEX" in combined.columns:
        # e.g. 'MALE' / 'Male' -> 'Male'
        combined["SEX"] = combined["SEX"].astype("string").str.strip().str.title()

    dup = combined.groupby("subject_id")["group"].nunique()
    if (dup > 1).any():
        print(f"  [warn] {int((dup > 1).sum())} subject(s) appear in more than one group: "
              f"{dup[dup > 1].index.tolist()[:10]}")

    out = Path(args.output) if args.output else in_dir / f"all_groups{name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    z_cols = [c for c in combined.columns if c.endswith("_z")]
    print(f"Wrote {out}: {len(combined)} rows, {combined['subject_id'].nunique()} subjects, "
          f"{len(groups)} groups, {len(z_cols)} transformed features")


if __name__ == "__main__":
    main()
