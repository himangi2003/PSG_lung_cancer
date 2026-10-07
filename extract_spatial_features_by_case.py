#!/usr/bin/env python3
"""Copy eight selected ViSpace spatial-feature CSVs for TCGA CASE IDs.

Python 3, standard library only.

Example:
  python extract_spatial_features_by_case.py --input WSI_or_ViSpace_output \
      --ids female_psg_positive.txt --output female_psg_positive_features

The ID file contains one CASE ID per line, such as TCGA-05-4250.
A case matches WSI FOLDER names such as TCGA-05-4250-01Z-00-DX1.<uuid>.svs.
The case ID can occur anywhere in a name; hyphens and underscores are supported.
All matching sample codes and slides are included by default. To select only
sample code 01, add --sample-type 01. The slide's vial letter is ignored.

The immediate subfolders of --input are the WSI folders. For example:
  WSI_or_ViSpace_output/<WSI_NAME>/spatial_feature_results/<category>/<file.csv>
Match the case ID against each WSI folder's name, then search inside that folder
for its outputs. A WSI folder can have a .svs suffix; it is still a directory.
Nested output directories, category folders directly inside the WSI folder,
and the selected CSVs directly inside an output directory are supported.
Always writes case_wsi_matches.csv covering EVERY requested case ID and ALL
matching WSI folders. Use --match-only to create this mapping without copying features.
Copies the same eight CSVs into OUTPUT/CASE_ID. When several slides match a
case, each copied filename is prefixed with its WSI folder name.
Detects spatial_feature_results or spatial_features_results automatically.
If both exist for a slide, choose one explicitly with --results-folder.
Existing destinations are skipped; source files are never moved or changed.
Writes extraction_report.csv and missing_case_ids.txt in OUTPUT.
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path


CASE_PATTERN = re.compile(r"TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}", re.I)
WSI_PATTERN = re.compile(
    r"(?P<case_id>TCGA[-_][A-Z0-9]{2}[-_][A-Z0-9]{4})(?![A-Z0-9])"
    r"(?:[-_](?P<sample_code>\d{2})[A-Z]?(?=[-_.]|$))?",
    re.I,
)
RESULTS_FOLDERS = ('spatial_feature_results', 'spatial_features_results')
SELECTED_FILES = {
    'cluster_tils_tsr_score': (
        'tils_tsr_by_cluster.csv', 'tils_tsr_wsi_summary.csv'),
    'immune_proximity': (
        'immune_proximity_by_cluster.csv', 'immune_proximity_wsi_summary.csv'),
    'necrosis_feature': (
        'necrosis_feature_by_cluster.csv', 'necrosis_feature_wsi_summary.csv'),
    'tumor_morphology': (
        'tumor_core_features_by_cluster.csv', 'tumor_core_wsi_summary.csv'),
}


def read_case_ids(path):
    """Validate case IDs, preserve order, and remove duplicates."""
    ids, seen = [], set()
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        value = line.strip().upper()
        if not value or value.startswith('#') or value in {'CASE_ID', 'PATIENT_ID'}:
            continue
        if not CASE_PATTERN.fullmatch(value):
            raise ValueError(
                f'Invalid case ID at line {number}: {line!r}; '
                'use one case ID per line, for example TCGA-05-4250.')
        if value not in seen:
            ids.append(value)
            seen.add(value)
    if not ids:
        raise ValueError('No case IDs found in the text file.')
    return ids


def name_identity(name):
    """Read the case barcode from a name, without requiring a sample suffix."""
    matches = list(WSI_PATTERN.finditer(name))
    identities = {(m.group('case_id').upper().replace('_', '-'),
                   m.group('sample_code') or '') for m in matches}
    # Never guess when a name contains conflicting barcodes.
    return next(iter(identities)) if len(identities) == 1 else None


def discover(root, requested, sample_type, results_folder, parser, match_only=False):
    """Match immediate WSI folders, then discover outputs inside each matched WSI."""
    matches, slides, result_links = defaultdict(list), defaultdict(list), defaultdict(list)
    results, no_results, examples = [], [], []
    selected_names = {name for names in SELECTED_FILES.values() for name in names}

    def walk_error(error):
        parser.error(f'Cannot scan input directory: {error}')

    try:
        wsi_folders = sorted(path for path in root.iterdir() if path.is_dir())
    except OSError as exc:
        parser.error(f'Cannot list WSI folders: {exc}')
    examples = [folder.name for folder in wsi_folders[:12]]
    for folder in wsi_folders:
        identity = name_identity(folder.name)
        if not identity:
            continue
        case_id, sample_code = identity
        if case_id not in requested or (sample_type is not None and sample_code != sample_type):
            continue
        matches[case_id].append((folder, 'wsi_folder', sample_code))

        candidates = []
        for current, dirnames, filenames in os.walk(folder, onerror=walk_error):
            directory = Path(current)
            dirnames.sort()
            if results_folder != 'auto':
                # An explicit spelling selects that result tree, including its descendants.
                dirnames[:] = [name for name in dirnames
                               if name not in RESULTS_FOLDERS or name == results_folder]
            if (directory.name in RESULTS_FOLDERS
                    or any(category in dirnames for category in SELECTED_FILES)
                    or selected_names.intersection(filenames)):
                candidates.append(directory)
            # Category contents belong to this result set, not to a second WSI.
            dirnames[:] = [name for name in dirnames if name not in SELECTED_FILES]

        # Discard empty wrapper containers when their actual output roots are deeper.
        candidate_set = set(candidates)
        containers = {parent for result in candidates for parent in result.parents
                      if parent in candidate_set
                      and not any((parent / category).is_dir() for category in SELECTED_FILES)
                      and not any((parent / name).is_file() for name in selected_names)}
        candidates = [result for result in candidates if result not in containers]
        groups = defaultdict(list)
        for result in candidates:
            if result.name in RESULTS_FOLDERS:
                groups[result.parent].append(result)
        if not match_only and results_folder == 'auto':
            for parent, paths in groups.items():
                if len(paths) > 1:
                    parser.error(f'Both result-folder spellings exist in {parent}; '
                                 'use --results-folder to select one.')
        if not candidates:
            no_results.append(folder)
            slides[case_id].append((folder, sample_code, None, folder.name))
        for result in candidates:
            slides[case_id].append((folder, sample_code, result, folder.name))
            result_links[folder].append(result)
            results.append(result)
    return matches, slides, result_links, results, no_results, examples


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', required=True, type=Path,
                        help='Parent directory whose immediate subfolders are WSI names')
    parser.add_argument('--ids', required=True, type=Path,
                        help='Text file with one TCGA case ID per line')
    parser.add_argument('--output', required=True, type=Path,
                        help='Destination directory')
    parser.add_argument('--sample-type', metavar='NN',
                        help='Optional two-digit sample code, e.g. 01; default: all codes')
    parser.add_argument('--results-folder', default='auto',
                        choices=('auto',) + RESULTS_FOLDERS)
    parser.add_argument('--match-only', action='store_true',
                        help='Write the complete case-to-WSI-name mapping without copying files')
    args = parser.parse_args(argv)

    root, output = args.input.resolve(), args.output.resolve()
    if not root.is_dir():
        parser.error(f'Input directory does not exist: {root}')
    if output == root or root in output.parents or output in root.parents:
        parser.error('Input and output must be separate, non-overlapping directories.')
    if args.sample_type is not None and not re.fullmatch(r'\d{2}', args.sample_type):
        parser.error('--sample-type must be a two-digit code, for example 01.')
    try:
        case_ids = read_case_ids(args.ids)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))

    matches_by_case, slides, result_links, results, no_results, examples = discover(
        root, set(case_ids), args.sample_type, args.results_folder, parser, args.match_only)
    output.mkdir(parents=True, exist_ok=True)
    mapping_rows = []
    for case_id in case_ids:
        for path, kind, sample_code in matches_by_case[case_id]:
            mapping_rows.append([case_id, path.name, str(path), kind, sample_code,
                                 'matched', ' | '.join(map(str, result_links[path]))])
        if not matches_by_case[case_id]:
            mapping_rows.append([case_id, '', '', '', '', 'no_matching_wsi_folder', ''])
    with (output / 'case_wsi_matches.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'wsi_name', 'wsi_path', 'source_type',
                         'sample_code', 'match_status', 'feature_result_directories'])
        writer.writerows(mapping_rows)
    unmatched = [case_id for case_id in case_ids if not matches_by_case[case_id]]
    (output / 'unmatched_case_ids.txt').write_text(
        ''.join(f'{case_id}\n' for case_id in unmatched), encoding='utf-8')
    print(f'Requested cases: {len(case_ids)}')
    print(f'Cases with matching WSI folders: {len(case_ids) - len(unmatched)}')
    print(f'Matching WSI folders: {sum(len(v) for v in matches_by_case.values())}')
    print(f'Cases with linked feature results: {sum(any(s[2] is not None for s in slides[c]) for c in case_ids)}')
    print(f'Sample-code filter: {args.sample_type or "all"}')
    print(f'Complete case-to-WSI mapping: {output / "case_wsi_matches.csv"}')
    diagnostics = [f'Input: {root}', f'Feature result directories found: {len(results)}',
                   f'Matched WSI folders without feature results: {len(no_results)}',
                   'Example WSI folder names immediately inside input:'] + examples
    diagnostics += ['Matched WSI folders without feature results:'] + [str(p) for p in no_results]
    (output / 'discovery_summary.txt').write_text('\n'.join(diagnostics) + '\n', encoding='utf-8')
    if len(unmatched) == len(case_ids):
        print('No requested case barcode was found in the WSI folder names. '
              'Set --input to the parent directory containing your WSI-named folders. '
              'See discovery_summary.txt for the folder names checked. '
              'UUID-only WSI names require a separate barcode-to-UUID mapping.')
    if args.match_only:
        return 0

    rows, missing = [], []
    for case_id in case_ids:
        incomplete = False
        matches = slides[case_id]
        case_output = output / case_id
        case_output.mkdir(parents=True, exist_ok=True)
        if not matches:
            incomplete = True
            rows.append([case_id, '', '', '', 'missing_wsi_folder', '', '', ''])
        name_counts = Counter(item[3] for item in matches)
        for folder, sample_code, result_directory, wsi_name in matches:
            if result_directory is None:
                incomplete = True
                rows.append([case_id, sample_code, wsi_name, '',
                             'missing_feature_results', str(folder), '', ''])
                continue
            prefix = wsi_name
            if name_counts[wsi_name] > 1:
                # Identical slide names in different batches must not collide.
                relative = str(result_directory.relative_to(root))
                prefix += '__' + hashlib.sha256(relative.encode()).hexdigest()[:12]
            for subfolder, filenames in SELECTED_FILES.items():
                for filename in filenames:
                    source = result_directory / subfolder / filename
                    if not source.is_file() and (result_directory / filename).is_file():
                        source = result_directory / filename
                    output_name = f'{prefix}__{filename}' if len(matches) > 1 else filename
                    destination = case_output / output_name
                    error = ''
                    if not source.is_file():
                        status = 'missing_file'
                        incomplete = True
                    elif destination.exists() or destination.is_symlink():
                        status = 'skipped_existing'
                    else:
                        try:
                            # Stage each file so a failed copy leaves no partial destination.
                            with tempfile.TemporaryDirectory(prefix='.copy-', dir=case_output) as temporary:
                                staged = Path(temporary) / filename
                                shutil.copy2(source, staged)
                                staged.rename(destination)
                            status = 'copied'
                        except (OSError, shutil.Error) as exc:
                            status, error = 'copy_error', str(exc)
                            incomplete = True
                    rows.append([case_id, sample_code, wsi_name, filename, status,
                                 str(source), str(destination), error])
        if incomplete:
            missing.append(case_id)

    with (output / 'extraction_report.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['case_id', 'sample_code', 'wsi_folder', 'filename',
                         'status', 'source', 'destination', 'error'])
        writer.writerows(rows)
    (output / 'missing_case_ids.txt').write_text(
        ''.join(f'{case_id}\n' for case_id in missing), encoding='utf-8')

    counts = Counter(row[4] for row in rows)
    print(f'Matched feature result sets: {len(results)}')
    print(f'Cases with missing sources or copy errors: {len(missing)}')
    for status, count in sorted(counts.items()):
        print(f'{status}: {count}')
    print(f'Reports saved in: {output}')
    return 1 if counts['copy_error'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
