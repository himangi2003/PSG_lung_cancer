#!/usr/bin/env python3
"""Build four separate WSI-level feature CSVs and optionally add clinical data.

Requires: pip install pandas

Run:
    python LUAD_merge_four_summaries.py --input female_psg_positive_features \
        --clinical luad_clinical_with_wsi_name.csv --output-dir luad_summary_outputs

Output filenames start with the input folder name after removing a trailing
_features. For female_psg_positive_features, the TILs/TSR output is
female_psg_positive_tils_tsr_wsi_summary.csv. The run report uses the same
group prefix. Per-summary issues CSVs are not written.

All four families are processed independently. Each output retains every valid
WSI, including multiple WSIs belonging to the same patient. Clinical rows join
by patient ID AND full normalized WSI name, not by patient ID alone. Different
stages assigned to different WSIs in the clinical CSV remain distinct. No stage
is inferred, averaged, or selected from another WSI. Conflicting clinical rows
for the same patient/WSI cause a reported error rather than row multiplication.

Keep extraction_report.csv at the input root to recover WSI identities for
unprefixed summary filenames. Missing/invalid summaries are printed in the
terminal and summarized in the group-prefixed run report. All four output
CSVs are written; a family with no usable data has headers only and is marked
in the run report.
A nonzero exit code means at least one family is missing, partial, or failed;
successful families are still saved. Existing named outputs are replaced on rerun.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


SUMMARIES = {
    "tils_tsr": "tils_tsr_wsi_summary.csv",
    "tumor_core": "tumor_core_wsi_summary.csv",
    "immune_proximity": "immune_proximity_wsi_summary.csv",
    "necrosis_feature": "necrosis_feature_wsi_summary.csv",
}

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


def load_report_names(root, summary_filename):
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
        if row["filename"] != summary_filename or not row["destination"]:
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


def build_dataframe(input_dir, summary_filename, wsi_column="wsi_name", files=None):
    root = Path(input_dir).resolve()

    if not root.is_dir():
        raise ValueError(f"Input folder does not exist: {root}")

    report_names = load_report_names(root, summary_filename)

    if files is None:
        files = sorted(
            path for path in root.rglob("*.csv")
            if path.name == summary_filename or path.name.endswith("__" + summary_filename)
        )

    if not files:
        raise ValueError(
            f"No {summary_filename} files found inside {root}. Check --input."
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
                source.name[:-len("__" + summary_filename)]
                if source.name.endswith("__" + summary_filename)
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--input', required=True, type=Path,
                        help='Extracted feature directory; all four summary types are searched')
    parser.add_argument('--clinical', '--survival', dest='clinical', type=Path,
                        help='Optional clinical CSV with PATIENT_ID and wsi_name')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Directory for four summary CSVs and one run report')
    parser.add_argument('--wsi-column', default='wsi_name',
                        help='WSI-name column in source summary CSVs')
    parser.add_argument('--patient-column', default='PATIENT_ID',
                        help='Patient-ID column in the clinical CSV')
    args = parser.parse_args(argv)

    root = args.input.resolve()
    output_dir = args.output_dir.resolve()
    group_name = args.input.name
    if group_name.endswith('_features'):
        group_name = group_name[:-len('_features')]
    if not group_name or group_name in {'.', '..'}:
        group_name = root.name
        if group_name.endswith('_features'):
            group_name = group_name[:-len('_features')]
    if not group_name:
        parser.error('Cannot determine group name from the input folder.')
    if not root.is_dir():
        parser.error(f'Input directory does not exist: {root}')
    if output_dir == root or root in output_dir.parents:
        parser.error('Choose --output-dir outside the extracted input directory.')
    if args.clinical and not args.clinical.is_file():
        parser.error(f'Clinical CSV does not exist: {args.clinical}')

    output_paths = [output_dir / f'{group_name}_{filename}' for filename in SUMMARIES.values()]
    run_report_path = output_dir / f'{group_name}_run_report.csv'
    if args.clinical and args.clinical.resolve() in {
        path.resolve() for path in output_paths + [run_report_path]
    }:
        parser.error('An output path would overwrite the clinical input CSV.')

    # One recursive inventory, then independent processing for each feature family.
    files_by_family = {family: [] for family in SUMMARIES}
    try:
        for path in sorted(root.rglob('*.csv')):
            for family, filename in SUMMARIES.items():
                if path.name == filename or path.name.endswith('__' + filename):
                    files_by_family[family].append(path)
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        parser.error(str(exc))

    run_rows = []
    any_problem = False
    identifier_columns = ['wsi_name', 'sample_id', 'subject_id', 'source_file']
    for family, filename in SUMMARIES.items():
        output_path = output_dir / f'{group_name}_{filename}'
        files = files_by_family[family]
        features = pd.DataFrame(columns=identifier_columns)
        issues = pd.DataFrame(columns=ISSUE_COLUMNS)
        status = 'ok'
        detail = ''
        matched_rows = 0
        missing_clinical_rows = 0

        try:
            if not files:
                status = 'no_files'
                detail = f'No {filename} or WSI-prefixed versions were found.'
                issues = pd.DataFrame(
                    [[str(root), status, detail]], columns=ISSUE_COLUMNS)
            else:
                features, issues, _ = build_dataframe(
                    root, filename, args.wsi_column, files=files)
                if features.empty:
                    features = pd.DataFrame(columns=identifier_columns)
                    status = 'no_valid_rows'
                    detail = 'All discovered summaries were invalid; see terminal details below.'
                else:
                    if args.clinical:
                        features = merge_clinical(
                            features, args.clinical, args.patient_column)
                        matched_rows = int(features['clinical_match'].eq('matched').sum())
                        missing_clinical_rows = int(
                            features['clinical_match'].eq('missing_clinical').sum())
                    if not issues.empty:
                        status = 'partial'
                        detail = 'Some summary files or identifiers have issues; see terminal details below.'
        except (OSError, ValueError, KeyError) as exc:
            status = 'failed'
            detail = str(exc)
            features = pd.DataFrame(columns=identifier_columns)
            issues = pd.concat([
                issues,
                pd.DataFrame([[str(root), status, detail]], columns=ISSUE_COLUMNS),
            ], ignore_index=True)

        # Header-only outputs on failure prevent stale prior results from looking current.
        try:
            features.drop(columns=['source_file'], errors='ignore').to_csv(
                output_path, index=False, encoding='utf-8-sig')
        except OSError as exc:
            parser.error(f'Cannot save {family} outputs: {exc}')

        any_problem = any_problem or status != 'ok'
        patients = features['subject_id'].nunique() if len(features) else 0
        named_wsis = int(features['wsi_name'].ne('').sum()) if len(features) else 0
        run_rows.append({
            'group_name': group_name,
            'feature_family': family,
            'source_summary': filename,
            'source_files': len(files),
            'output_rows': len(features),
            'unique_patients': patients,
            'named_wsi_rows': named_wsis,
            'clinical_matched_rows': matched_rows if args.clinical else '',
            'clinical_unmatched_rows': missing_clinical_rows if args.clinical else '',
            'issue_count': len(issues),
            'status': status,
            'detail': detail,
            'output_file': str(output_path),
        })
        print(f'{family}: {len(files)} files -> {len(features)} WSI rows; '
              f'{patients} patients; status={status}')
        if args.clinical:
            print(f'  Clinical matched={matched_rows}; unmatched={missing_clinical_rows}')
        if detail:
            print(f'  {detail}')
        for issue in issues.itertuples(index=False):
            print(f'  {issue.status}: {issue.source_file}: {issue.detail}')
        print(f'  Saved: {output_path}')

    try:
        pd.DataFrame(run_rows).to_csv(
            run_report_path, index=False, encoding='utf-8-sig')
    except OSError as exc:
        parser.error(f'Cannot save run report: {exc}')
    print(f'Run report: {run_report_path}')
    return 1 if any_problem else 0


if __name__ == '__main__':
    raise SystemExit(main())
