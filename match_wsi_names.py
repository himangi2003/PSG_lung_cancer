#!/usr/bin/env python3
"""
Match clinical patients to WSI names and save the merged data as CSV.

- Automatically detects the Excel header row containing PATIENT_ID.
- Joins clinical PATIENT_ID to mapping case_id.
- Keeps every distinct patient–WSI pair.
- Repeats clinical data for patients with multiple WSIs.
- Retains unmatched patients with a blank wsi_name.

Dependencies:
    pip install pandas openpyxl

Example:
    python match_wsi_names.py \
        --clinical lung_clinical_data_patient.xlsx \
        --mapping TCGA_LUAD_WSI_case_mapping.csv \
        --output luad_clinical_with_wsi_name.csv
"""

import argparse
from pathlib import Path

import pandas as pd


def clean_columns(frame, label):
    frame.columns = frame.columns.astype(str).str.strip()
    duplicates = frame.columns[frame.columns.duplicated()].tolist()
    if duplicates:
        raise ValueError(
            f"{label} contains duplicate column names after trimming: {duplicates}"
        )
    return frame


def read_clinical(args):
    sheet = args.sheet if args.sheet is not None else 0

    if args.header_row is not None:
        header_index = args.header_row - 1
    else:
        preview = pd.read_excel(
            args.clinical,
            sheet_name=sheet,
            header=None,
            nrows=100,
            dtype=str,
            keep_default_na=False,
        )

        header_rows = [
            index
            for index, row in preview.iterrows()
            if args.patient_column in [
                str(value).strip() for value in row
            ]
        ]

        if not header_rows:
            raise ValueError(
                f"Could not find {args.patient_column!r} in the first "
                "100 rows. Check --sheet or --patient-column, or specify "
                "--header-row explicitly."
            )

        header_index = header_rows[0]

    clinical = pd.read_excel(
        args.clinical,
        sheet_name=sheet,
        header=header_index,
        keep_default_na=False,
    )
    clinical = clean_columns(clinical, "Clinical file")

    if args.patient_column not in clinical.columns:
        raise ValueError(
            f"Clinical file is missing {args.patient_column!r}. "
            f"Available columns: {list(clinical.columns)}"
        )

    # Remove entirely empty spreadsheet rows, preserving rows with clinical data.
    nonempty = clinical.apply(
        lambda column: column.astype(str).str.strip().ne("")
    ).any(axis=1)
    clinical = clinical.loc[nonempty].copy()

    print(f"Clinical header: Excel row {header_index + 1}")
    return clinical


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--clinical", required=True, type=Path,
        help="Clinical Excel workbook",
    )
    parser.add_argument(
        "--mapping", required=True, type=Path,
        help="CSV containing case IDs and WSI names",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Destination CSV filename",
    )
    parser.add_argument(
        "--sheet",
        default=None,
        help="Clinical worksheet name; default: first worksheet",
    )
    parser.add_argument(
        "--header-row",
        type=int,
        default=None,
        help="Excel header row, counting from 1; default: automatic detection",
    )
    parser.add_argument("--patient-column", default="PATIENT_ID")
    parser.add_argument("--case-column", default="case_id")
    parser.add_argument("--wsi-column", default="wsi_name")
    args = parser.parse_args()

    if args.header_row is not None and args.header_row < 1:
        parser.error("--header-row must be at least 1.")

    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv.")

    for path in (args.clinical, args.mapping):
        if not path.is_file():
            parser.error(f"Input file does not exist: {path}")

    if args.output.resolve() in {
        args.clinical.resolve(),
        args.mapping.resolve(),
    }:
        parser.error("--output must differ from the input files.")

    try:
        clinical = read_clinical(args)

        mapping = pd.read_csv(
            args.mapping,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        mapping = clean_columns(mapping, "Mapping file")

        required = [args.case_column, args.wsi_column]
        missing = [
            column for column in required
            if column not in mapping.columns
        ]
        if missing:
            raise ValueError(
                f"Mapping file is missing columns: {missing}. "
                f"Available columns: {list(mapping.columns)}"
            )

        if "wsi_name" in clinical.columns:
            raise ValueError(
                "The clinical file already contains wsi_name. "
                "Use the original clinical workbook."
            )

        # Choose a temporary key that cannot overwrite a source column.
        key = "__case_join_key"
        while key in clinical.columns or key in mapping.columns:
            key += "_"

        clinical[key] = (
            clinical[args.patient_column]
            .astype(str)
            .str.strip()
            .str.upper()
        )

        wsi_mapping = pd.DataFrame({
            key: (
                mapping[args.case_column]
                .str.strip()
                .str.upper()
            ),
            "wsi_name": mapping[args.wsi_column].str.strip(),
        })

        # Empty mapping entries must not match empty clinical IDs.
        usable = wsi_mapping[key].ne("") & wsi_mapping["wsi_name"].ne("")
        ignored_mapping_rows = int((~usable).sum())
        wsi_mapping = wsi_mapping.loc[usable].copy()

        duplicate_pairs = int(wsi_mapping.duplicated().sum())
        wsi_mapping = wsi_mapping.drop_duplicates()

        clinical_ids = set(clinical.loc[clinical[key].ne(""), key])
        mapping_ids = set(wsi_mapping[key])
        matched_ids = clinical_ids & mapping_ids
        unmatched_ids = clinical_ids - mapping_ids

        # A left join keeps clinical patients without a WSI.
        # Multiple distinct WSIs produce multiple rows for that patient.
        merged = clinical.merge(
            wsi_mapping,
            on=key,
            how="left",
            sort=False,
        ).drop(columns=key)

        merged["wsi_name"] = merged["wsi_name"].fillna("")

        # Place the new column immediately after the patient identifier.
        columns = merged.columns.tolist()
        columns.remove("wsi_name")
        columns.insert(
            columns.index(args.patient_column) + 1,
            "wsi_name",
        )
        merged = merged[columns]

        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged.to_csv(
            args.output,
            index=False,
            encoding="utf-8-sig",
        )

    except (OSError, ValueError, ImportError, KeyError) as exc:
        parser.error(str(exc))

    print(f"Clinical rows: {len(clinical)}")
    print(f"Unique clinical patients: {len(clinical_ids)}")
    print(f"Patients matched to WSI names: {len(matched_ids)}")
    print(f"Patients without WSI matches: {len(unmatched_ids)}")
    print(f"Mapping cases absent from clinical data: {len(mapping_ids - clinical_ids)}")
    print(f"Empty mapping entries ignored: {ignored_mapping_rows}")
    print(f"Duplicate case–WSI pairs removed: {duplicate_pairs}")
    print(f"Output rows: {len(merged)}")
    print(f"Rows with blank wsi_name: {merged['wsi_name'].eq('').sum()}")
    print(f"Saved: {args.output.resolve()}")


if __name__ == "__main__":
    main()