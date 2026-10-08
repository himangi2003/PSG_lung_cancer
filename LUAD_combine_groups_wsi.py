#!/usr/bin/env python3
"""Combine group-level WSI feature tables separately for each feature type.

Requires: pip install pandas

Example:
    python LUAD_combine_groups_wsi.py --input luad_summary_outputs/wsi_level \
        --output-dir luad_all_groups_wsi

Searches --input recursively for filenames such as:
    female_psg_positive_tils_tsr_wsi_summary.csv
    male_psg_negative_tumor_core_wsi_summary.csv

Writes exactly four all_groups_<feature>_wsi_summary.csv files. Rows are
concatenated across groups within each feature type; no feature types are
joined and no patient/WSI averages are calculated. Missing columns across
groups are retained with blank values. Multiple WSIs per patient are kept.

group_name is taken from the filename prefix, patient_id is obtained from
existing case-ID columns or the WSI barcode, and wsi_name remains the original
full name. All source clinical/stage fields remain attached to their own WSI.
The same WSI in different groups is retained separately. Repeated normalized
group/patient/WSI keys within one feature type cause an error rather than being
silently dropped or counted twice. Names are normalized only for validation.

source_file is excluded from saved tables. No issues or report CSVs are saved.
Header-only input files are skipped with a terminal message. Missing feature
types get header-only output files and a nonzero exit code. Existing outputs
are replaced on rerun; use an output directory outside the input directory.
"""

import argparse
import re
from pathlib import Path

import pandas as pd


FEATURE_FILES = {
    'tils_tsr': 'tils_tsr_wsi_summary.csv',
    'tumor_core': 'tumor_core_wsi_summary.csv',
    'immune_proximity': 'immune_proximity_wsi_summary.csv',
    'necrosis_feature': 'necrosis_feature_wsi_summary.csv',
}
CASE = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', re.I)
CASE_IN_NAME = re.compile(r'(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4})(?![A-Z0-9])', re.I)
IMAGE_EXTENSION = re.compile(r'\.(svs|tif|tiff|ndpi|mrxs|scn|bif)$', re.I)
FIRST_COLUMNS = ['group_name', 'patient_id', 'wsi_name', 'sample_id', 'subject_id']


def normalize_wsi(value):
    name = str(value).strip().replace('\\', '/').rsplit('/', 1)[-1]
    return IMAGE_EXTENSION.sub('', name).upper()


def case_from_wsi(value):
    cases = {m.group(1).upper().replace('_', '-') for m in CASE_IN_NAME.finditer(value)}
    if len(cases) > 1:
        raise ValueError(f'Multiple case barcodes in WSI name: {value!r}')
    return next(iter(cases)) if cases else ''


def read_group_file(path, group):
    try:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False, encoding='utf-8-sig')
    except pd.errors.EmptyDataError:
        print(f'Skipping empty file: {path}')
        return None
    frame.columns = frame.columns.astype(str).str.strip()
    if frame.columns.duplicated().any():
        raise ValueError(f'Duplicate column names after trimming: {path}')
    if frame.empty:
        print(f'Skipping header-only file: {path}')
        return None
    if 'wsi_name' not in frame:
        raise ValueError(f'{path}: missing wsi_name. Use the merged WSI-level outputs.')
    if frame['wsi_name'].str.strip().eq('').any():
        rows = (frame.index[frame['wsi_name'].str.strip().eq('')] + 2).tolist()[:5]
        raise ValueError(f'{path}: blank WSI names at CSV rows {rows}. Recover the names before combining.')

    # Do not overwrite an existing, conflicting group label.
    if 'group_name' in frame:
        supplied = frame['group_name'].str.strip()
        if (supplied.ne('') & supplied.ne(group)).any():
            raise ValueError(f'{path}: group_name column disagrees with filename prefix {group!r}.')
    frame['group_name'] = group

    patient = pd.Series('', index=frame.index, dtype='object')
    for column in ['patient_id', 'subject_id', 'PATIENT_ID', 'case_id']:
        if column not in frame:
            continue
        candidate = frame[column].str.strip().str.upper().str.replace('_', '-', regex=False)
        nonblank = candidate.ne('')
        if (nonblank & ~candidate.str.fullmatch(CASE)).any():
            raise ValueError(f'{path}: {column} contains values that are not TCGA case barcodes.')
        if (patient.ne('') & nonblank & patient.ne(candidate)).any():
            raise ValueError(f'{path}: patient identifier columns disagree.')
        patient = patient.where(patient.ne(''), candidate)

    inferred = frame['wsi_name'].map(case_from_wsi)
    if (patient.ne('') & inferred.ne('') & patient.ne(inferred)).any():
        raise ValueError(f'{path}: patient ID disagrees with the WSI barcode.')
    patient = patient.where(patient.ne(''), inferred)
    if patient.eq('').any():
        raise ValueError(f'{path}: cannot determine patient_id for every WSI.')
    frame['patient_id'] = patient

    # Internal source paths from older scripts are not exported.
    return frame.drop(columns=['source_file'], errors='ignore')


def concatenate_family(parts, family):
    if not parts:
        return pd.DataFrame(columns=FIRST_COLUMNS)
    combined = pd.concat(parts, ignore_index=True, sort=False).fillna('')
    keys = pd.DataFrame({
        'group_name': combined['group_name'],
        'patient_id': combined['patient_id'],
        'wsi_name': combined['wsi_name'].map(normalize_wsi),
    })
    duplicate = keys.duplicated(keep=False)
    if duplicate.any():
        examples = keys.loc[duplicate].drop_duplicates().head().to_dict('records')
        raise ValueError(f'{family}: repeated group/patient/WSI keys. '
                         f'Check duplicate input files or rows. Examples: {examples}')
    order = [column for column in FIRST_COLUMNS if column in combined]
    order += [column for column in combined if column not in order]
    return combined[order].sort_values(
        ['group_name', 'patient_id', 'wsi_name'], kind='stable').reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='Folder containing group-prefixed WSI CSVs, including nested folders')
    parser.add_argument('--output-dir', required=True, type=Path,
                        help='Directory for the four all-groups feature tables')
    args = parser.parse_args(argv)
    root, out = args.input.resolve(), args.output_dir.resolve()
    if not root.is_dir():
        parser.error(f'Input directory does not exist: {root}')
    if out == root or root in out.parents:
        parser.error('Choose --output-dir outside --input to avoid re-reading generated outputs.')

    try:
        parts = {family: [] for family in FEATURE_FILES}
        file_counts = {family: 0 for family in FEATURE_FILES}
        for path in sorted(root.rglob('*.csv')):
            for family, suffix in FEATURE_FILES.items():
                ending = '_' + suffix
                if not path.name.endswith(ending):
                    continue
                group = path.name[:-len(ending)]
                if not group or group == 'all_groups':
                    continue
                # Extraction copies use WSI__filename; this script needs merged group files.
                if group.endswith('_') or CASE_IN_NAME.search(group):
                    continue
                frame = read_group_file(path, group)
                if frame is not None:
                    parts[family].append(frame)
                    file_counts[family] += 1

        if not any(parts.values()):
            raise ValueError('No nonempty group-level WSI tables found. Expected filenames such as '
                             'female_psg_negative_tils_tsr_wsi_summary.csv.')

        # Validate every family before replacing any output files.
        outputs = {family: concatenate_family(parts[family], family) for family in FEATURE_FILES}
        out.mkdir(parents=True, exist_ok=True)
        incomplete = False
        for family, suffix in FEATURE_FILES.items():
            frame = outputs[family]
            destination = out / ('all_groups_' + suffix)
            frame.to_csv(destination, index=False, encoding='utf-8-sig')
            groups = frame['group_name'].nunique()
            patients = frame['patient_id'].nunique()
            print(f'{family}: {file_counts[family]} files; {groups} groups; '
                  f'{patients} unique patients; {len(frame)} group/WSI rows')
            if frame.empty:
                incomplete = True
                print('  No data for this feature type; saved headers only.')
            print(f'  Saved: {destination}')
        return 1 if incomplete else 0
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    raise SystemExit(main())
