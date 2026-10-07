#!/usr/bin/env python3
"""Combine four per-cluster feature families across TCGA WSIs.

Requires: pip install pandas

Input can be extracted CASE_ID folders (keep extraction_report.csv at the root)
or original WSI-named folders with nested output directories. Every source row
is retained as one patient/WSI/cluster row; no patient or WSI averaging occurs.
Cluster IDs are read from the CSV, never invented from row numbers. Original
cluster labels, including zero and negative values, are preserved as text.

Outputs use the input-folder name with a trailing _features removed. Four feature
CSVs and a run report are written; no issues CSVs are created. An error in one
family is printed and recorded in the run report while other families continue.
Failed or missing families produce header-only outputs and a nonzero exit code.
Existing named outputs are replaced on rerun.

Optional clinical data joins by case ID and full normalized WSI name, so each
cluster receives the clinical data for its own WSI. Different WSIs for a patient
remain separate. Stage is copied from the clinical source and is never inferred.
"""

import argparse
import re
from pathlib import Path

import pandas as pd

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

def names_agree(inferred, reported):
    if normalize_wsi(inferred) == normalize_wsi(reported):
        return True

    # The case-based extractor can append a 12-character hash to filenames
    # when multiple output sets have the same WSI folder name.
    without_hash = re.sub(r"__[0-9a-f]{12}$", "", inferred, flags=re.I)

    return normalize_wsi(without_hash) == normalize_wsi(reported)


CLUSTER_FILES = {
    'tils_tsr': 'tils_tsr_by_cluster.csv',
    'tumor_core': 'tumor_core_features_by_cluster.csv',
    'immune_proximity': 'immune_proximity_by_cluster.csv',
    'necrosis_feature': 'necrosis_feature_by_cluster.csv',
}
ID_COLUMNS = ['patient_id', 'wsi_name', 'cluster_id', 'sample_id']
BASE_COLUMNS = ID_COLUMNS + ['source_file', 'source_row']
CLUSTER_ALIASES = {
    'cluster', 'clusterid', 'clusterlabel', 'clusternumber',
    'clusterno', 'clusternum', 'clusterindex',
}


def normalized_column(name):
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def choose_cluster_column(frame, requested=None):
    if requested:
        if requested not in frame.columns:
            raise ValueError(f'Cluster column {requested!r} is missing. '
                             f'Available columns: {list(frame.columns)}')
        return requested
    if 'cluster_id' in frame.columns:
        return 'cluster_id'
    matches = [c for c in frame.columns if normalized_column(c) in CLUSTER_ALIASES]
    if len(matches) != 1:
        raise ValueError('Cannot choose a unique cluster-ID column. '
                         f'Candidates: {matches}; columns: {list(frame.columns)}. '
                         'Specify --cluster-column with the actual column name.')
    return matches[0]


def report_lookup(root):
    names = {}
    report_path = root / 'extraction_report.csv'
    if not report_path.is_file():
        return names
    report = read_csv(report_path)
    id_col = next((c for c in ['case_id', 'sample_id'] if c in report), None)
    if not id_col or not {'wsi_folder', 'filename', 'destination'}.issubset(report.columns):
        raise ValueError('extraction_report.csv requires case_id or sample_id, '
                         'and wsi_folder, filename, destination.')
    for _, row in report.iterrows():
        if row['filename'] not in CLUSTER_FILES.values() or not row['destination']:
            continue
        if 'status' in report and row['status'] not in {'copied', 'skipped_existing'}:
            continue
        case_id = case_from_name(row[id_col])
        wsi = basename(row['wsi_folder'])
        if not case_id or not wsi:
            raise ValueError('A cluster entry in extraction_report.csv has no valid case or WSI name.')
        key = (case_id, basename(row['destination']))
        if key in names and normalize_wsi(names[key]) != normalize_wsi(wsi):
            raise ValueError(f'Ambiguous WSI names in extraction report for {key}.')
        names[key] = wsi
    return names


def source_identity(source, root, filename, reports):
    parent_names = (root.name, *source.relative_to(root).parts[:-1])
    folder_case = folder_sample = folder_wsi = ''
    for name in reversed(parent_names):
        folder_case = folder_case or case_from_name(name)
        folder_sample = folder_sample or sample_from_name(name)
        if not folder_wsi and WSI_SEARCH.search(name):
            folder_wsi = name
    prefix = source.name[:-len('__' + filename)] if source.name.endswith('__' + filename) else ''
    inferred = prefix or folder_wsi
    inferred_case = case_from_name(inferred)
    if folder_case and inferred_case and folder_case != inferred_case:
        raise ValueError(f'Folder and filename case IDs disagree: {source}')
    lookup_case = folder_case or inferred_case
    reported = reports.get((lookup_case, source.name), '')
    if not reported and not lookup_case:
        candidates = {value for (case, name), value in reports.items() if name == source.name}
        if len(candidates) == 1:
            reported = next(iter(candidates))
    if reported and inferred and not names_agree(inferred, reported):
        raise ValueError(f'Extraction report and filename/folder disagree: {source}')
    return reported or inferred, folder_case, folder_sample


def combine_cluster_files(root, filename, files, reports, cluster_column, wsi_column):
    records = []
    reserved = set(BASE_COLUMNS) | {'subject_id', 'PATIENT_ID'}
    for source in files:
        frame = read_csv(source)
        if frame.empty:
            raise ValueError(f'Cluster file has no data rows: {source}')
        cluster_col = choose_cluster_column(frame, cluster_column)
        clusters = frame[cluster_col].str.strip()
        if clusters.str.lower().isin({'', 'nan', 'none', 'null', 'na', 'n/a'}).any():
            raise ValueError(f'Missing cluster IDs in {source}; IDs will not be invented.')
        file_wsi, folder_case, folder_sample = source_identity(source, root, filename, reports)
        for row_number, (_, row) in enumerate(frame.iterrows(), start=2):
            row_wsi = basename(row.get(wsi_column, ''))
            if file_wsi and row_wsi and not names_agree(file_wsi, row_wsi):
                raise ValueError(f'WSI column and file identity disagree: {source}, row {row_number}')
            wsi = file_wsi or row_wsi
            if not wsi:
                raise ValueError(f'Cannot recover WSI name: {source}, row {row_number}. '
                                 'Keep extraction_report.csv, use WSI-named folders, '
                                 'or provide a WSI column in the CSV.')

            case_candidates = [folder_case, case_from_name(wsi)]
            for column in ['patient_id', 'PATIENT_ID', 'case_id', 'subject_id']:
                value = str(row.get(column, '')).strip().upper().replace('_', '-')
                if CASE_RE.fullmatch(value):
                    case_candidates.append(value)
            samples = [folder_sample, sample_from_name(wsi)]
            for column in ['sample_id', 'Sample ID']:
                value = str(row.get(column, '')).strip().upper().replace('_', '-')
                if SAMPLE_RE.fullmatch(value):
                    samples.append(value)
            samples = {value for value in samples if value}
            if len(samples) > 1:
                raise ValueError(f'Conflicting sample IDs: {source}, row {row_number}')
            sample = next(iter(samples)) if samples else ''
            if sample:
                case_candidates.append(sample.rsplit('-', 1)[0])
            cases = {value for value in case_candidates if value}
            if len(cases) != 1:
                raise ValueError(f'Missing or conflicting patient ID: {source}, row {row_number}')
            case_id = next(iter(cases))
            record = {
                'patient_id': case_id,
                'wsi_name': wsi,
                'cluster_id': str(row[cluster_col]).strip(),
                'sample_id': sample,
            }
            for column, value in row.items():
                # cluster_id already preserves this column's actual value.
                if column == 'cluster_id' and cluster_col == 'cluster_id':
                    continue
                target = 'feature__' + column if column in reserved else column
                if target in record:
                    raise ValueError(f'Feature column collision for {target!r}: {source}')
                record[target] = value
            record['source_file'] = str(source)
            record['source_row'] = row_number
            records.append(record)

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)
    keys = pd.DataFrame({
        'patient_id': result['patient_id'],
        'wsi': result['wsi_name'].map(normalize_wsi),
        'cluster_id': result['cluster_id'],
    })
    duplicates = keys.duplicated(keep=False)
    if duplicates.any():
        examples = result.loc[duplicates, ID_COLUMNS + ['source_file']].head().to_dict('records')
        raise ValueError('Repeated patient/WSI/cluster keys within this feature family. '
                         'Check duplicate files, the selected cluster column, or whether '
                         f'rows have an additional measurement dimension. Examples: {examples}')
    metric_columns = [c for c in result if c not in BASE_COLUMNS]
    return result[ID_COLUMNS + metric_columns + ['source_file', 'source_row']]


def prepare_clinical(path, patient_column):
    clinical = read_csv(path)
    if patient_column is None:
        patient_column = next((c for c in ['subject_id', 'PATIENT_ID', 'patient_id', 'case_id']
                               if c in clinical), None)
    if not patient_column or patient_column not in clinical or 'wsi_name' not in clinical:
        raise ValueError('Clinical CSV requires a patient-ID column and wsi_name. '
                         'Set --patient-column if necessary.')
    reserved = {'__case_key', '__wsi_key', 'clinical_match'}
    if reserved.intersection(clinical.columns):
        raise ValueError('Clinical CSV already has merge-helper columns; use the original clinical data.')
    clinical['__case_key'] = clinical[patient_column].str.strip().str.upper()
    clinical['__wsi_key'] = clinical['wsi_name'].map(normalize_wsi)
    if not clinical['__case_key'].str.fullmatch(CASE_RE).all():
        raise ValueError(f'Invalid clinical case IDs in {patient_column}.')
    wsi_cases = clinical['wsi_name'].map(case_from_name)
    if (wsi_cases.ne('') & wsi_cases.ne(clinical['__case_key'])).any():
        raise ValueError('Clinical case IDs disagree with the case barcodes in their WSI names.')
    clinical = clinical.loc[clinical['__wsi_key'].ne('')].drop(columns='wsi_name').drop_duplicates()
    if clinical.duplicated(['__case_key', '__wsi_key']).any():
        raise ValueError('Conflicting clinical rows for the same patient/WSI. '
                         'Distinct WSIs may have different stages; one WSI must have '
                         'one unambiguous clinical record for this merge.')
    return clinical


def merge_clinical(features, clinical):
    keys = ['__case_key', '__wsi_key']
    if (set(keys) | {'clinical_match'}).intersection(features.columns):
        raise ValueError('Feature columns conflict with reserved clinical-merge columns.')
    left = features.copy()
    left['__case_key'] = left['patient_id']
    left['__wsi_key'] = left['wsi_name'].map(normalize_wsi)
    overlaps = (set(left.columns) & set(clinical.columns)) - set(keys)
    rename = {c: 'clinical__' + c for c in overlaps}
    if any(name in left or name in clinical for name in rename.values()):
        raise ValueError('Cannot preserve colliding clinical columns under clinical__ names.')
    merged = left.merge(clinical.rename(columns=rename), on=keys, how='left',
                        validate='many_to_one', indicator='clinical_match', sort=False)
    merged['clinical_match'] = merged['clinical_match'].map({
        'both': 'matched', 'left_only': 'missing_clinical', 'right_only': 'unused'})
    if len(merged) != len(features):
        raise ValueError('Clinical merge changed the number of cluster rows.')
    return merged.drop(columns=keys)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--clinical', type=Path, help='Optional WSI-mapped clinical CSV')
    parser.add_argument('--patient-column', help='Clinical case-ID column; detected when omitted')
    parser.add_argument('--cluster-column', help='Source cluster-ID column; detected when omitted')
    parser.add_argument('--wsi-column', default='wsi_name', help='WSI column inside source cluster CSVs')
    parser.add_argument('--group-name', help='Override input-folder-derived output prefix')
    args = parser.parse_args(argv)
    root = args.input.resolve()
    out = args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f'Input directory does not exist: {root}')
    if out == root or root in out.parents:
        parser.error('Choose an output directory outside the input directory.')
    group = args.group_name or args.input.name
    if not group or group in {'.', '..'}:
        group = root.name
    if not args.group_name and group.endswith('_features'):
        group = group[:-len('_features')]
    if not group or group in {'.', '..'} or '/' in group or '\\' in group:
        parser.error('Group name must be a nonempty filename component.')
    outputs = {family: out / f'{group}_{name}' for family, name in CLUSTER_FILES.items()}
    run_path = out / f'{group}_cluster_run_report.csv'
    if args.clinical and args.clinical.resolve() in set(outputs.values()) | {run_path}:
        parser.error('An output would overwrite the clinical input.')

    try:
        clinical = prepare_clinical(args.clinical, args.patient_column) if args.clinical else None
        reports = report_lookup(root)
        inventory = {family: [] for family in CLUSTER_FILES}
        for source in sorted(root.rglob('*.csv')):
            for family, filename in CLUSTER_FILES.items():
                if source.name == filename or source.name.endswith('__' + filename):
                    inventory[family].append(source)
        out.mkdir(parents=True, exist_ok=True)
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))

    run_rows = []
    failed = False
    for family, filename in CLUSTER_FILES.items():
        files = inventory[family]
        status, detail = 'ok', ''
        frame = pd.DataFrame(columns=BASE_COLUMNS)
        try:
            if not files:
                status, detail = 'no_files', f'No {filename} files were found.'
            else:
                frame = combine_cluster_files(root, filename, files, reports,
                                              args.cluster_column, args.wsi_column)
                if clinical is not None:
                    frame = merge_clinical(frame, clinical)
        except (OSError, ValueError, KeyError) as exc:
            status, detail = 'failed', str(exc)
            frame = pd.DataFrame(columns=BASE_COLUMNS)
        try:
            frame.drop(columns=['source_file'], errors='ignore').to_csv(
                outputs[family], index=False, encoding='utf-8-sig')
        except OSError as exc:
            parser.error(f'Cannot save {outputs[family]}: {exc}')
        patients = frame['patient_id'].nunique()
        wsis = len(frame[['patient_id', 'wsi_name']].assign(
            wsi_name=frame['wsi_name'].map(normalize_wsi)).drop_duplicates())
        matched = int(frame['clinical_match'].eq('matched').sum()) if 'clinical_match' in frame else 0
        run_rows.append({
            'group_name': group, 'feature_family': family,
            'source_files': len(files), 'cluster_rows': len(frame),
            'unique_patients': patients, 'unique_wsis': wsis,
            'clinical_matched_cluster_rows': matched if clinical is not None else '',
            'status': status, 'detail': detail, 'output_file': str(outputs[family]),
        })
        print(f'{family}: {len(files)} files -> {len(frame)} cluster rows; '
              f'{wsis} WSIs; {patients} patients; status={status}')
        if detail:
            print(f'  {detail}')
        print(f'  Saved: {outputs[family]}')
        failed = failed or status != 'ok'
    try:
        pd.DataFrame(run_rows).to_csv(run_path, index=False, encoding='utf-8-sig')
    except OSError as exc:
        parser.error(f'Cannot save run report: {exc}')
    print(f'Run report: {run_path}')
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

