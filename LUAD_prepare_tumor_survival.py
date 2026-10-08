#!/usr/bin/env python3
"""Select tumor-core WSI features and prepare one record per patient.

Dependencies: pip install pandas numpy

Example:
  python LUAD_prepare_tumor_survival.py \
      --input all_groups_tumor_core_wsi_summary.csv \
      --output tumor_core_patient_survival.csv

Outputs: the requested columns at WSI level, plus a separate patient-level CSV
containing raw aggregated features and their transformations. Numeric features
use the median across distinct WSIs by default; --aggregation mean is available.
Aggregation happens BEFORE transformation. Counts become median/mean WSI counts,
not total counts across a patient's slides. Precomputed area-weighted WSI metrics
are aggregated as WSI measurements, not recomputed as pooled tissue metrics.

Clinical values are not averaged: nonmissing SEX, stage, OS_Status and OS_MONTHS
must agree within a patient. Resolve conflicts to the intended baseline cohort
before rerunning. Do not pool different clinical timepoints into a baseline
survival model. No stage or survival endpoint is inferred.

Positive-valued distance/size/density aggregates receive log2 columns. Zero
aggregates retain their raw value but receive a missing log2 value; no arbitrary
pseudocount is added. Count aggregates receive natural-log log1p columns as an
alternative to their raw values. Other requested features remain raw.

Spacing is preserved as metadata, not assumed to be a numeric model predictor.
SEX_male and stage_II/III/IV are optional model inputs with female and stage I
as references; unknown categories stay missing. OS_Status is 1=death, 0=censored.
survival_outcome_valid requires positive OS_MONTHS and a known binary status.
All patients are retained; missing predictors are not imputed and invalid
survival outcomes are flagged, not silently removed. No model is fitted.

Duplicate copies of the same patient/WSI with identical retained values count
once in aggregation. Conflicting copies cause an error. The filtered WSI output
retains every original row and the exact requested columns. No issues files are
written. No standardization is performed; learn any scaling/imputation inside
training folds rather than across the full dataset.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


KEEP_COLS = [
    'sample_id', 'subject_id', 'wsi_name',
    'n_clusters_with_tumor',
    'wsi_tumor_n_islands',
    'wsi_cluster_area_mm2',
    'wsi_tumor_area_mm2',
    'wsi_tumor_perimeter_mm',
    'wsi_tumor_fraction_of_cluster',
    'wsi_tumor_boundary_density_per_mm',
    'wsi_tumor_patch_density_per_mm2',
    'wsi_area_weighted_tumor_largest_patch_index',
    'wsi_area_weighted_tumor_compactness_mean',
    'wsi_area_weighted_tumor_solidity_mean',
    'wsi_area_weighted_tumor_elongation_mean',
    'wsi_area_weighted_tumor_boundary_fractal_dimension',
    'Spacing',
    'wsi_area_weighted_tumor_island_nnd_median_um',
    'wsi_area_weighted_tumor_island_gap_median_um',
    'SEX', 'OS_MONTHS', 'OS_Status', 'stage',
]
LOG2_FEATURES = [
    'wsi_area_weighted_tumor_island_nnd_median_um',
    'wsi_area_weighted_tumor_island_gap_median_um',
    'wsi_tumor_area_mm2',
    'wsi_cluster_area_mm2',
    'wsi_tumor_perimeter_mm',
    'wsi_tumor_boundary_density_per_mm',
]
COUNT_FEATURES = ['n_clusters_with_tumor', 'wsi_tumor_n_islands']
NONFEATURES = {'sample_id', 'subject_id', 'wsi_name', 'Spacing',
               'SEX', 'OS_MONTHS', 'OS_Status', 'stage'}
FEATURES = [column for column in KEEP_COLS if column not in NONFEATURES]
MISSING = {'', 'na', 'n/a', 'nan', 'none', 'null', 'unknown', 'not reported',
           'not available', '[not available]', '[not applicable]', '[unknown]',
           '[not reported]', 'not applicable', '--', '-'}
CASE = re.compile(r'TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}', re.I)
CASE_IN_NAME = re.compile(r'(TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4})(?![A-Z0-9])', re.I)


def missing_value(value):
    return str(value).strip().lower() in MISSING


def normalize_wsi(value):
    name = str(value).strip().replace('\\', '/').rsplit('/', 1)[-1]
    return re.sub(r'\.(svs|tif|tiff|ndpi|mrxs|scn|bif)$', '', name, flags=re.I).upper()


def numeric_column(series, column):
    text = series.astype(str).str.strip()
    absent = text.map(missing_value)
    numeric = pd.to_numeric(text.where(~absent, np.nan), errors='coerce')
    invalid = ~absent & (numeric.isna() | ~np.isfinite(numeric))
    if invalid.any():
        raise ValueError(f'{column}: unrecognized numeric values: '
                         f'{text[invalid].drop_duplicates().head().tolist()}')
    if numeric.lt(0).any():
        raise ValueError(f'{column}: negative values found. Resolve these values before modelling.')
    return numeric


def normalize_sex(value):
    text = str(value).strip().upper()
    if missing_value(text):
        return np.nan
    mapping = {'F': 'FEMALE', 'FEMALE': 'FEMALE', 'M': 'MALE', 'MALE': 'MALE'}
    if text not in mapping:
        raise ValueError(f'Unrecognized SEX value: {value!r}')
    return mapping[text]


def normalize_stage(value):
    if missing_value(value):
        return np.nan
    text = re.sub(r'^STAGE\s*', '', str(value).strip().upper())
    text = re.sub(r'\s+', '', text)
    match = re.fullmatch(r'(IV|III|II|I)(?:[ABC](?:[123])?)?', text)
    if match:
        return match.group(1)
    if text in {'1', '2', '3', '4'}:
        return {'1': 'I', '2': 'II', '3': 'III', '4': 'IV'}[text]
    raise ValueError(f'Unrecognized stage value: {value!r}; expected I, II, III or IV.')


def prepare(raw, aggregation):
    missing = [column for column in KEEP_COLS if column not in raw]
    if missing:
        raise ValueError(f'Missing required columns: {missing}. Available: {list(raw.columns)}')
    filtered = raw[KEEP_COLS].copy()
    if filtered.empty:
        raise ValueError('Input contains no WSI rows.')
    work = filtered.copy()
    work['subject_id'] = work['subject_id'].str.strip().str.upper()
    work['sample_id'] = work['sample_id'].str.strip().str.upper()
    if not work['subject_id'].str.fullmatch(CASE).all():
        raise ValueError('subject_id must contain a valid TCGA case barcode for every row.')
    work['__wsi_key'] = work['wsi_name'].map(normalize_wsi)
    if work['__wsi_key'].eq('').any():
        raise ValueError('Blank wsi_name prevents identifying unique WSIs.')
    for _, row in work.iterrows():
        encoded = {m.group(1).upper().replace('_', '-')
                   for m in CASE_IN_NAME.finditer(row['wsi_name'])}
        if encoded and encoded != {row['subject_id']}:
            raise ValueError(f'WSI barcode disagrees with subject_id: {row["wsi_name"]}')
        sample_case = CASE_IN_NAME.search(row['sample_id'])
        if sample_case and sample_case.group(1).upper().replace('_', '-') != row['subject_id']:
            raise ValueError(f'sample_id disagrees with subject_id: {row["sample_id"]}')

    for column in FEATURES + ['OS_MONTHS', 'OS_Status']:
        work[column] = numeric_column(work[column], column)
    if not work['OS_Status'].dropna().isin([0, 1]).all():
        raise ValueError('OS_Status must be 0 (censored/alive) or 1 (death), or missing.')
    for column in COUNT_FEATURES:
        values = work[column].dropna()
        if not np.isclose(values, np.round(values)).all():
            raise ValueError(f'{column}: source WSI counts must be whole numbers.')
    work['SEX'] = work['SEX'].map(normalize_sex)
    work['stage'] = work['stage'].map(normalize_stage)

    # Repeated group membership must not give the same WSI additional weight.
    comparison = [column for column in KEEP_COLS if column != 'wsi_name']
    retained = []
    duplicates_removed = 0
    for key, group in work.groupby(['subject_id', '__wsi_key'], sort=False):
        if len(group[comparison].drop_duplicates()) > 1:
            raise ValueError(f'Conflicting copies of the same patient/WSI: {key}. '
                             'Resolve duplicate source records before aggregation.')
        retained.append(group.index[0])
        duplicates_removed += len(group) - 1
    work = work.loc[retained].copy()

    # Select the one known clinical value, never average stage or outcomes.
    clinical_cols = ['SEX', 'stage', 'OS_Status', 'OS_MONTHS']
    conflicts = []
    for subject, group in work.groupby('subject_id', sort=True):
        for column in clinical_cols:
            values = group[column].dropna().unique()
            if len(values) > 1:
                conflicts.append(f'{subject}: {column}={values.tolist()}')
    if conflicts:
        raise ValueError('Conflicting patient-level clinical records. Select/correct the '
                         'intended baseline records; no value was chosen automatically. '
                         + '; '.join(conflicts[:10]))

    records = []
    for subject, group in work.groupby('subject_id', sort=True):
        record = {
            'sample_id': ' | '.join(sorted(set(group['sample_id']) - {''})),
            'subject_id': subject,
            'wsi_name': ' | '.join(group['wsi_name'].astype(str)),
            'n_wsi': len(group),
            'feature_aggregation': aggregation,
        }
        for column in FEATURES:
            values = group[column].dropna()
            record[column] = getattr(values, aggregation)() if len(values) else np.nan
        spacing = [str(v).strip() for v in group['Spacing'] if not missing_value(v)]
        record['Spacing'] = ' | '.join(sorted(set(spacing)))
        for column in clinical_cols:
            known = group[column].dropna()
            record[column] = known.iloc[0] if len(known) else np.nan
        records.append(record)
    patient = pd.DataFrame(records)
    patient = patient[KEEP_COLS + ['n_wsi', 'feature_aggregation']]
    for column in LOG2_FEATURES:
        # Mask nonpositive values before log2; zero stays visible in raw columns.
        positive = patient[column].where(patient[column].gt(0))
        patient['log2_' + column] = np.log2(positive)
    for column in COUNT_FEATURES:
        patient['log1p_' + column] = np.log1p(patient[column])
    patient['OS_Status'] = patient['OS_Status'].astype('Int64')
    patient['SEX_male'] = patient['SEX'].map({'FEMALE': 0, 'MALE': 1}).astype('Int64')
    for stage in ['II', 'III', 'IV']:
        patient['stage_' + stage] = patient['stage'].map(
            lambda value: pd.NA if pd.isna(value) else int(value == stage)).astype('Int64')
    patient['survival_outcome_valid'] = (
        patient['OS_MONTHS'].gt(0) & patient['OS_Status'].isin([0, 1]))
    assert patient['subject_id'].is_unique
    return filtered, patient, duplicates_removed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path,
                        help='Patient-level transformed CSV')
    parser.add_argument('--wsi-output', type=Path,
                        help='Optional path for filtered WSI rows; default: OUTPUT_STEM_wsi_filtered.csv')
    parser.add_argument('--aggregation', choices=['median', 'mean'], default='median')
    args = parser.parse_args(argv)
    wsi_output = args.wsi_output or args.output.with_name(args.output.stem + '_wsi_filtered.csv')
    paths = [args.input.resolve(), args.output.resolve(), wsi_output.resolve()]
    if len(set(paths)) != 3:
        parser.error('Input, patient output and WSI output paths must be different.')
    if any(path.suffix.lower() != '.csv' for path in [args.output, wsi_output]):
        parser.error('Both output paths must end in .csv.')
    try:
        raw = pd.read_csv(args.input, dtype=str, keep_default_na=False, encoding='utf-8-sig')
        raw.columns = raw.columns.str.strip()
        if raw.columns.duplicated().any():
            raise ValueError('Duplicate input column names after trimming whitespace.')
        filtered, patient, dropped = prepare(raw, args.aggregation)
        for path in [wsi_output, args.output]:
            path.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_csv(wsi_output, index=False, encoding='utf-8-sig')
        patient.to_csv(args.output, index=False, encoding='utf-8-sig', na_rep='')
    except (OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(f'WSI input rows retained in filtered output: {len(filtered)}')
    print(f'Identical duplicate WSI copies removed for patient aggregation: {dropped}')
    print(f'Patient rows: {len(patient)}; feature aggregation: {args.aggregation}')
    print(f'Patients with a valid positive survival time and known event: '
          f'{int(patient["survival_outcome_valid"].sum())}')
    print('Log2 columns with missing values (raw zero values are not changed):')
    for column in LOG2_FEATURES:
        print(f'  {column}: raw zeros={int(patient[column].eq(0).sum())}; '
              f'missing log2={int(patient["log2_" + column].isna().sum())}')
    print('Missing clinical values: ' + ', '.join(
        f'{column}={int(patient[column].isna().sum())}'
        for column in ['SEX', 'stage', 'OS_MONTHS', 'OS_Status']))
    print(f'Filtered WSI data: {wsi_output.resolve()}')
    print(f'Patient survival data: {args.output.resolve()}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
