#!/usr/bin/env python3
"""
Clean tumor_core WSI-level summaries: keep only selected columns.

Input files follow the pattern:
    {group_name}_tumor_core_wsi_summary.csv

Groups are discovered automatically from the file names, so any set of
groups (e.g. female_psg_negative, male_psg_positive, ...) is supported.

Outputs (in --output-dir):
    {group_name}_tumor_core_wsi_clean.csv   one per group
    all_groups_tumor_core_wsi_clean.csv     all groups stacked, with a 'group' column

Dependencies:
    pip install pandas

Example:
    python LUAD_clean_tumor_core.py \
        --input-dir luad_summary_outputs/wsi_level \
        --output-dir luad_summary_outputs/cleaned/tumor_core
"""

import argparse
from pathlib import Path

import pandas as pd


FEATURE = "tumor_core"
SUFFIX = f"_{FEATURE}_wsi_summary.csv"

KEEP_COLS = [
    "wsi_name",
    "sample_id",
    "subject_id",
    "n_clusters_with_tumor",
    "wsi_tumor_n_islands",
    "wsi_cluster_area_mm2",
    "wsi_tumor_area_mm2",
    "wsi_tumor_perimeter_mm",
    "wsi_tumor_fraction_of_cluster",
    "wsi_tumor_boundary_density_per_mm",
    "wsi_tumor_patch_density_per_mm2",
    "wsi_area_weighted_tumor_largest_patch_index",
    "wsi_area_weighted_tumor_compactness_mean",
    "wsi_area_weighted_tumor_solidity_mean",
    "wsi_area_weighted_tumor_elongation_mean",
    "wsi_area_weighted_tumor_boundary_fractal_dimension",
    "wsi_area_weighted_tumor_island_nnd_median_um",
    "wsi_area_weighted_tumor_island_gap_median_um",
    "OS_MONTHS",
    "OS_Status",
    "stage",
]


def clean_file(path: Path) -> pd.DataFrame:
    # utf-8-sig strips the BOM so the first column reads as 'wsi_name'
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = df.columns.str.strip()

    present = [c for c in KEEP_COLS if c in df.columns]
    missing = [c for c in KEEP_COLS if c not in df.columns]
    if missing:
        print(f"  [warn] {path.name}: missing columns skipped: {missing}")

    return df[present]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", default="luad_summary_outputs/wsi_level",
                        help="Folder containing *_tumor_core_wsi_summary.csv files")
    parser.add_argument("--output-dir", default="luad_summary_outputs/cleaned/tumor_core",
                        help="Folder to write cleaned CSVs")
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob(f"*{SUFFIX}"))
    if not files:
        raise SystemExit(f"No files matching *{SUFFIX} in {in_dir}")

    combined = []
    for path in files:
        group = path.name[: -len(SUFFIX)]
        print(f"Processing group '{group}': {path.name}")

        df = clean_file(path)
        out_path = out_dir / f"{group}_{FEATURE}_wsi_clean.csv"
        df.to_csv(out_path, index=False)
        print(f"  wrote {out_path} ({len(df)} rows, {df.shape[1]} cols)")

        combined.append(df.assign(group=group))

    all_df = pd.concat(combined, ignore_index=True)
    all_df = all_df[["group"] + [c for c in all_df.columns if c != "group"]]
    all_path = out_dir / f"all_groups_{FEATURE}_wsi_clean.csv"
    all_df.to_csv(all_path, index=False)
    print(f"Wrote combined {all_path} ({len(all_df)} rows, {all_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
