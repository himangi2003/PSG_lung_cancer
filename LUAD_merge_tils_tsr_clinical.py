#!/usr/bin/env python3
"""
Combine TILs/TSR WSI summaries and optionally merge clinical data.

Clinical CSV:
    luad_clinical_with_wsi_name.csv

Required clinical columns:
    PATIENT_ID, wsi_name

Features are joined to clinical data by:
    subject_id + normalized WSI name

Multiple WSIs per patient are supported. Feature rows without matching
clinical data are retained.

Dependencies:
    pip install pandas

Examples:
    python merge_tils_tsr_clinical.py \
        --input female_psg_positive_features \
        --clinical luad_clinical_with_wsi_name.csv \
        --output tils_tsr_with_clinical.csv

    python merge_tils_tsr_clinical.py \
        --features-csv tils_tsr_features.csv \
        --clinical luad_clinical_with_wsi_name.csv \
        --output tils_tsr_with_clinical.csv
"""

import argparse
import re
from pathlib import Path

import pandas as pd


SUMMARY = "necrosis_feature_wsi_summary.csv"

CASE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.I)
SAMPLE_RE = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-\d{2}", re.I)

CASE_SEARCH = re.compile(
    r"(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4})(?![A-Z0-9])",
    re.I,
)
SAMPLE_SEARCH = re.compile(
    r"(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}[-_]\d{2})"
    r"[A-Z]?(?=[-_.]|$)",
    re.I,
)
WSI_SEARCH = re.compile(
    r"TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4}[-_]\d{2}"
    r"[A-Z]?[-_].+",
    re.I,
)

IMAGE_EXTENSION = re.compile(
    r"\.(svs|tif|tiff|ndpi|mrxs|scn|bif)$",
    re.I,
)

ISSUE_COLUMNS = ["source_file", "status", "detail"]


def basename(value):
    return str(value).strip().replace("\\", "/").rsplit("/", 1)[-1]


def normalize_wsi(value):
    """Keep the full slide identifier; remove only path and image extension."""
    return IMAGE_EXTENSION.sub("", basename(value)).upper()


def case_from_name(value):
    matches = {
        match.group(1).upper().replace("_", "-")
        for match in CASE_SEARCH.finditer(str(value))
    }
    return next(iter(matches)) if len(matches) == 1 else ""


def sample_from_name(value):
    matches = {
        match.group(1).upper().replace("_", "-")
        for match in SAMPLE_SEARCH.finditer(str(value))
    }
    return next(iter(matches)) if len(matches) == 1 else ""


def read_csv(path):
    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )
    frame.columns = frame.columns.astype(str).str.strip()

    if frame.columns.duplicated().any():
        raise ValueError(
            f"Duplicate column names after trimming whitespace: {path}"
        )

    return frame


def load_report_names(root):
    """Support extraction reports containing either case_id or sample_id."""
    report_path = root / "extraction_report.csv"
    names = {}

    if not report_path.is_file():
        return names

    report = read_csv(report_path)

    id_column = (
        "case_id" if "case_id" in report.columns
        else "sample_id" if "sample_id" in report.columns
        else None
    )

    required = {"wsi_folder", "filename", "destination"}
    if not id_column or not required.issubset(report.columns):
        raise ValueError(
            "extraction_report.csv must contain case_id or sample_id, "
            "plus wsi_folder, filename, and destination."
        )

    for _, row in report.iterrows():
        if row["filename"] != SUMMARY or not row["destination"]:
            continue

        if (
            "status" in report.columns
            and row["status"] not in {"copied", "skipped_existing"}
        ):
            continue

        case_id = case_from_name(row[id_column])
        name = basename(row["wsi_folder"])

        if not case_id or not name:
            continue

        key = (case_id, basename(row["destination"]))

        if (
            key in names
            and normalize_wsi(names[key]) != normalize_wsi(name)
        ):
            raise ValueError(
                f"Conflicting WSI names in extraction report: {key}"
            )

        names[key] = name

    return names


def names_agree(inferred, reported):
    if normalize_wsi(inferred) == normalize_wsi(reported):
        return True

    # The case-based extractor can append a 12-character hash to filenames
    # when multiple output sets have the same WSI folder name.
    without_hash = re.sub(r"__[0-9a-f]{12}$", "", inferred, flags=re.I)

    return normalize_wsi(without_hash) == normalize_wsi(reported)


def add_subject_id(frame):
    """Validate or recover sample_id and derive the corresponding case ID."""
    frame = frame.copy()

    if "wsi_name" not in frame.columns:
        raise ValueError("Feature CSV must contain wsi_name.")

    if "sample_id" not in frame.columns:
        frame["sample_id"] = ""

    samples = []
    subjects = []

    for index, row in frame.iterrows():
        supplied = str(row["sample_id"]).strip().upper().replace("_", "-")
        inferred = sample_from_name(row["wsi_name"])
        sample = supplied or inferred

        if not SAMPLE_RE.fullmatch(sample):
            raise ValueError(
                f"Cannot determine a valid sample_id for feature row {index + 1}. "
                "Keep extraction_report.csv with your extracted feature files."
            )

        if supplied and inferred and supplied != inferred:
            raise ValueError(
                f"sample_id and wsi_name disagree at feature row {index + 1}."
            )

        subject = sample.rsplit("-", 1)[0]

        if "subject_id" in frame.columns:
            existing = str(row["subject_id"]).strip().upper()
            if existing and existing != subject:
                raise ValueError(
                    f"subject_id and sample_id disagree at row {index + 1}."
                )

        samples.append(sample)
        subjects.append(subject)

    frame["sample_id"] = samples
    frame["subject_id"] = subjects

    # Repeated feature rows for the same WSI would overweight that slide.
    keys = pd.DataFrame({
        "subject_id": frame["subject_id"],
        "wsi_key": frame["wsi_name"].map(normalize_wsi),
    })

    named = keys["wsi_key"].ne("")
    if keys.loc[named].duplicated().any():
        raise ValueError(
            "Multiple feature rows map to the same patient–WSI pair. "
            "Check for duplicate or stale extracted summaries."
        )

    first = ["wsi_name", "sample_id", "subject_id"]
    return frame[first + [column for column in frame if column not in first]]


def build_dataframe(input_dir, wsi_column="wsi_name"):
    root = Path(input_dir).resolve()

    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")

    report_names = load_report_names(root)

    files = sorted(
        path for path in root.rglob("*.csv")
        if path.name == SUMMARY or path.name.endswith("__" + SUMMARY)
    )

    if not files:
        raise ValueError(
            f"No {SUMMARY} files found inside {root}. Check --input."
        )

    records = []
    issues = []

    for source in files:
        try:
            parent_names = (
                root.name,
                *source.relative_to(root).parts[:-1],
            )

            folder_case = ""
            folder_sample = ""
            folder_wsi = ""

            for name in reversed(parent_names):
                if not folder_case:
                    folder_case = case_from_name(name)
                if not folder_sample:
                    folder_sample = sample_from_name(name)
                if not folder_wsi and WSI_SEARCH.search(name):
                    folder_wsi = name

            prefix = (
                source.name[:-len("__" + SUMMARY)]
                if source.name.endswith("__" + SUMMARY)
                else ""
            )

            inferred_name = prefix or folder_wsi
            inferred_case = case_from_name(inferred_name)

            if folder_case and inferred_case and folder_case != inferred_case:
                raise ValueError("Case folder and WSI filename disagree.")

            lookup_case = folder_case or inferred_case
            reported_name = report_names.get(
                (lookup_case, source.name), ""
            )

            if (
                reported_name
                and inferred_name
                and not names_agree(inferred_name, reported_name)
            ):
                raise ValueError(
                    "Extraction report and filename/folder WSI names disagree."
                )

            file_identity = reported_name or inferred_name
            frame = read_csv(source)

            if frame.empty:
                raise ValueError("Summary contains no data rows.")

            if len(frame) > 1:
                if wsi_column not in frame.columns:
                    raise ValueError(
                        f"Multi-row summary requires a {wsi_column!r} column."
                    )

                row_keys = frame[wsi_column].map(normalize_wsi)

                if row_keys.eq("").any() or row_keys.duplicated().any():
                    raise ValueError(
                        "Multi-row summary requires distinct, nonblank WSI names."
                    )

            pending = []

            for _, values in frame.iterrows():
                csv_name = basename(values.get(wsi_column, ""))

                if (
                    file_identity
                    and csv_name
                    and normalize_wsi(file_identity) != normalize_wsi(csv_name)
                ):
                    raise ValueError(
                        "Summary WSI name disagrees with its filename, "
                        "folder, or extraction report."
                    )

                if len(frame) > 1 and file_identity:
                    raise ValueError(
                        "A summary identified as one WSI contains multiple WSI rows."
                    )

                name = file_identity or csv_name

                from_name = sample_from_name(name)
                from_column = str(
                    values.get("sample_id", values.get("Sample ID", ""))
                ).strip().upper().replace("_", "-")

                available_samples = [
                    value for value in
                    (from_name, folder_sample, from_column)
                    if value
                ]

                if len(set(available_samples)) > 1:
                    raise ValueError(
                        "Sample identifiers disagree between the WSI name, "
                        "folder, and summary columns."
                    )

                sample = available_samples[0] if available_samples else ""

                if not SAMPLE_RE.fullmatch(sample):
                    raise ValueError(
                        "Cannot determine sample_id. Keep extraction_report.csv "
                        "at the root of the extracted feature directory."
                    )

                subject = sample.rsplit("-", 1)[0]

                if folder_case and subject != folder_case:
                    raise ValueError("WSI identity disagrees with its case folder.")

                record = {
                    "wsi_name": name,
                    "sample_id": sample,
                    "subject_id": subject,
                }

                reserved = {
                    "wsi_name", "sample_id", "subject_id", "source_file"
                }

                for column, value in values.items():
                    target = (
                        f"feature__{column}" if column in reserved else column
                    )

                    if target in record:
                        raise ValueError(
                            f"Feature column name collision: {target}"
                        )

                    record[target] = value

                record["source_file"] = str(source)
                pending.append(record)

            records.extend(pending)

            if any(not record["wsi_name"] for record in pending):
                issues.append([
                    str(source),
                    "missing_wsi_name",
                    "Feature row retained but cannot receive a WSI-level clinical match.",
                ])

        except (OSError, ValueError) as exc:
            issues.append([str(source), "invalid_summary", str(exc)])

    result = pd.DataFrame(records)

    if not result.empty:
        result = add_subject_id(result)

    return (
        result,
        pd.DataFrame(issues, columns=ISSUE_COLUMNS),
        len(files),
    )


def merge_clinical(features, clinical_path, patient_column="PATIENT_ID"):
    features = add_subject_id(features)
    clinical = read_csv(clinical_path)

    required = {patient_column, "wsi_name"}
    missing = required - set(clinical.columns)

    if missing:
        raise ValueError(
            f"Clinical CSV is missing columns: {sorted(missing)}. "
            f"Available columns: {list(clinical.columns)}"
        )

    reserved = {"__case_key", "__wsi_key", "clinical_match"}
    conflicts = reserved & (set(features.columns) | set(clinical.columns))

    if conflicts:
        raise ValueError(
            f"Inputs already contain reserved columns: {sorted(conflicts)}. "
            "Use the feature-only CSV and the original clinical CSV."
        )

    features = features.copy()
    clinical = clinical.copy()

    features["__case_key"] = features["subject_id"].str.strip().str.upper()
    features["__wsi_key"] = features["wsi_name"].map(normalize_wsi)

    clinical[patient_column] = (
        clinical[patient_column].str.strip().str.upper()
    )
    clinical["__case_key"] = clinical[patient_column]
    clinical["__wsi_key"] = clinical["wsi_name"].map(normalize_wsi)

    valid_cases = clinical["__case_key"].str.fullmatch(CASE_RE)

    if not valid_cases.all():
        examples = (
            clinical.loc[~valid_cases, patient_column].head().tolist()
        )
        raise ValueError(
            f"Invalid or blank clinical patient IDs: {examples}"
        )

    # Reject known barcode disagreements instead of attaching the wrong patient.
    named_cases = clinical["wsi_name"].map(case_from_name)
    disagreements = (
        named_cases.ne("")
        & named_cases.ne(clinical["__case_key"])
    )
    if disagreements.any():
        raise ValueError(
            "Some clinical WSI names contain a case barcode that disagrees "
            f"with {patient_column}."
        )

    # A clinical row with no WSI cannot match a specific feature slide.
    clinical = clinical.loc[clinical["__wsi_key"].ne("")].copy()

    # Retain the feature-side spelling of wsi_name in the final output.
    clinical = clinical.drop(columns="wsi_name").drop_duplicates()

    keys = ["__case_key", "__wsi_key"]
    duplicates = clinical.duplicated(keys, keep=False)

    if duplicates.any():
        examples = (
            clinical.loc[duplicates, keys]
            .drop_duplicates()
            .head()
            .to_dict("records")
        )
        raise ValueError(
            "Conflicting clinical rows for the same patient–WSI pair: "
            f"{examples}"
        )

    # Preserve overlapping clinical fields without replacing feature values.
    overlaps = (
        set(features.columns) & set(clinical.columns)
    ) - set(keys)

    rename = {column: f"clinical__{column}" for column in overlaps}

    for old_name, new_name in rename.items():
        if new_name in features.columns or new_name in clinical.columns:
            raise ValueError(
                f"Cannot rename {old_name!r}: {new_name!r} already exists."
            )

    clinical = clinical.rename(columns=rename)

    merged = features.merge(
        clinical,
        on=keys,
        how="left",
        validate="many_to_one",
        indicator="clinical_match",
        sort=False,
    )

    merged["clinical_match"] = merged["clinical_match"].map({
        "both": "matched",
        "left_only": "missing_clinical",
        "right_only": "unused",
    })

    merged = merged.drop(columns=keys)

    if len(merged) != len(features):
        raise ValueError("Clinical merge changed the feature-row count.")

    return merged


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument(
        "--input",
        type=Path,
        help="Directory containing extracted TILs/TSR summary CSVs",
    )
    inputs.add_argument(
        "--features-csv",
        type=Path,
        help="Previously created TILs/TSR feature CSV",
    )

    parser.add_argument(
        "--clinical", "--survival",
        dest="clinical",
        type=Path,
        help="Clinical CSV containing PATIENT_ID and wsi_name",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--wsi-column",
        default="wsi_name",
        help="WSI-name column inside source summary CSVs",
    )
    parser.add_argument(
        "--patient-column",
        default="PATIENT_ID",
        help="Patient-ID column in the clinical CSV",
    )

    args = parser.parse_args()

    output = args.output.resolve()
    issues_path = output.with_name(output.stem + "_issues.csv")

    if output.suffix.lower() != ".csv":
        parser.error("--output must end in .csv.")

    sources = {
        path.resolve()
        for path in (args.features_csv, args.clinical)
        if path is not None
    }

    if output in sources or issues_path in sources:
        parser.error("Output files must not overwrite an input CSV.")

    if args.input:
        input_root = args.input.resolve()
        if output == input_root or input_root in output.parents:
            parser.error(
                "Save the output outside the extracted input directory."
            )

    try:
        if args.features_csv:
            features = add_subject_id(read_csv(args.features_csv))
            issues = pd.DataFrame(columns=ISSUE_COLUMNS)
        else:
            features, issues, file_count = build_dataframe(
                args.input,
                args.wsi_column,
            )
            print(f"Summary files found: {file_count}")

        output.parent.mkdir(parents=True, exist_ok=True)

        if args.input:
            issues.to_csv(
                issues_path,
                index=False,
                encoding="utf-8-sig",
            )

        if features.empty:
            raise ValueError(
                f"No feature rows available. Check {issues_path} "
                "when extracting from a folder."
            )

        if args.clinical:
            features = merge_clinical(
                features,
                args.clinical,
                args.patient_column,
            )

        features.to_csv(
            output,
            index=False,
            encoding="utf-8-sig",
        )

    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    print(f"Saved {len(features)} WSI feature rows: {output}")

    if args.clinical:
        print(features["clinical_match"].value_counts().to_string())

    if args.input:
        print(f"Reported issues: {len(issues)}")
        print(f"Issues file: {issues_path}")


if __name__ == "__main__":
    main()