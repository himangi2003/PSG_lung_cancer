#!/usr/bin/env python3

import argparse
import re
from pathlib import Path

import pandas as pd


CASE_RE = re.compile(
    r"(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4})(?![A-Z0-9])",
    re.I,
)

SAMPLE_RE = re.compile(
    r"(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}[-_]\d{2})"
    r"[A-Z]?(?=[-_.]|$)",
    re.I,
)


def extract_id(value, pattern):
    matches = {
        match.group(1).upper().replace("_", "-")
        for match in pattern.finditer(str(value))
    }
    if len(matches) > 1:
        raise ValueError(f"Multiple conflicting identifiers in: {value!r}")
    return next(iter(matches)) if matches else ""


def find_column(df, *names, required=True):
    # Prefer exact names, particularly OS_STATUS versus derived OS_Status.
    for name in names:
        if name in df.columns:
            return name

    matches = [
        column for column in df.columns
        if column.upper() in {name.upper() for name in names}
    ]

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"Ambiguous columns for {names}: {matches}")
    if required:
        raise ValueError(
            f"Missing column: {' or '.join(names)}. "
            f"Available columns: {list(df.columns)}"
        )
    return None


def get_optional_ids(df, *names):
    column = find_column(df, *names, required=False)
    if column is None:
        return pd.Series("", index=df.index, dtype="object")
    return df[column].str.strip().str.upper().str.replace("_", "-", regex=False)


def combine_ids(primary, fallback, label):
    conflict = primary.ne("") & fallback.ne("") & primary.ne(fallback)

    if conflict.any():
        examples = pd.DataFrame({
            "existing": primary[conflict],
            "inferred": fallback[conflict],
        }).head().to_dict("records")
        raise ValueError(f"Conflicting {label} values: {examples}")

    return primary.where(primary.ne(""), fallback)


def binary_os(value):
    value = re.sub(r"\s+", "", str(value).upper())

    mapping = {
        "0": 0,
        "0.0": 0,
        "0:LIVING": 0,
        "0:ALIVE": 0,
        "LIVING": 0,
        "ALIVE": 0,
        "1": 1,
        "1.0": 1,
        "1:DECEASED": 1,
        "1:DEAD": 1,
        "DECEASED": 1,
        "DEAD": 1,
    }
    return mapping.get(value, pd.NA)


def simplify_stage(value):
    value = str(value).strip().upper()
    value = re.sub(r"^STAGE\s*", "", value)
    value = re.sub(r"\s+", "", value)

    # IA/IB -> I; IIA/IIB -> II; IIIA/IIIB/IIIC -> III; IVA/IVB -> IV.
    match = re.fullmatch(r"(IV|III|II|I)(?:[ABC](?:[123])?)?", value)

    if match:
        return match.group(1)

    return {"1": "I", "2": "II", "3": "III", "4": "IV"}.get(
        value, pd.NA
    )


def main():
    parser = argparse.ArgumentParser(
        description="Keep WSI identifiers and selected LUAD clinical columns."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    stage_report = args.output.with_name(
        args.output.stem + "_stage_unmapped.csv"
    )

    if args.output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv.")

    if args.input.resolve() in {
        args.output.resolve(),
        stage_report.resolve(),
    }:
        parser.error("Output paths must differ from the input file.")

    try:
        df = pd.read_csv(
            args.input,
            dtype=str,
            keep_default_na=False,
            encoding="utf-8-sig",
        )
        df.columns = df.columns.str.strip()

        if df.columns.duplicated().any():
            raise ValueError("Duplicate column names after trimming whitespace.")

        wsi_column = find_column(df, "wsi_name", "wsi_filename")
        stage_column = find_column(
            df,
            "AJCC_PATHOLOGIC_TUMOR_STAGE",
            "AJDCC_PATHOLOGIC_TUMOR_STAGE",
        )

        clinical_columns = {
            name: find_column(df, name)
            for name in ["SEX", "AGE", "OS_STATUS", "OS_MONTHS"]
        }

        wsi_names = df[wsi_column].str.strip()

        # Preserve supplied identifiers and fill missing ones.
        sample_ids = combine_ids(
            get_optional_ids(df, "sample_id"),
            wsi_names.map(lambda value: extract_id(value, SAMPLE_RE)),
            "sample_id",
        )

        subject_ids = combine_ids(
            get_optional_ids(df, "subject_id"),
            get_optional_ids(df, "PATIENT_ID"),
            "subject_id/PATIENT_ID",
        )

        subject_ids = combine_ids(
            subject_ids,
            wsi_names.map(lambda value: extract_id(value, CASE_RE)),
            "subject_id/WSI case",
        )

        subject_ids = combine_ids(
            subject_ids,
            sample_ids.map(lambda value: extract_id(value, CASE_RE)),
            "subject_id/sample case",
        )

        result = pd.DataFrame({
            "sample_id": sample_ids,
            "subject_id": subject_ids,
            "wsi_name": wsi_names,
            "SEX": df[clinical_columns["SEX"]],
            "AJCC_PATHOLOGIC_TUMOR_STAGE": df[stage_column],
            "AGE": df[clinical_columns["AGE"]],
            "OS_STATUS": df[clinical_columns["OS_STATUS"]],
            "OS_MONTHS": df[clinical_columns["OS_MONTHS"]],
        })

        result["OS_Status"] = (
            result["OS_STATUS"].map(binary_os).astype("Int64")
        )
        result["stage"] = result[
            "AJCC_PATHOLOGIC_TUMOR_STAGE"
        ].map(simplify_stage)

        # Keep unmatched stages blank and preserve their raw values in a report.
        unmapped = result.loc[
            result["stage"].isna(),
            [
                "sample_id",
                "subject_id",
                "wsi_name",
                "AJCC_PATHOLOGIC_TUMOR_STAGE",
            ],
        ]

        args.output.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(
            args.output,
            index=False,
            encoding="utf-8-sig",
            na_rep="",
        )
        unmapped.to_csv(
            stage_report,
            index=False,
            encoding="utf-8-sig",
        )

    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"Rows saved: {len(result)}")
    print(f"Missing sample_id: {result['sample_id'].eq('').sum()}")
    print(f"Missing subject_id: {result['subject_id'].eq('').sum()}")
    print(f"Missing/unrecognized OS status: {result['OS_Status'].isna().sum()}")
    print(f"Missing/unrecognized stage: {len(unmapped)}")

    if not unmapped.empty:
        print("\nOriginal values that could not be mapped:")
        print(
            unmapped["AJCC_PATHOLOGIC_TUMOR_STAGE"]
            .replace("", "<blank>")
            .value_counts()
            .to_string()
        )

    print(f"\nSaved: {args.output.resolve()}")
    print(f"Stage report: {stage_report.resolve()}")


if __name__ == "__main__":
    main()