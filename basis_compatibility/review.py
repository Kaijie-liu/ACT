"""Inspect the sealed 30 obligations and four native records without new solves."""
import argparse
from collections import Counter
from pathlib import Path
import time
from single_check_portable.execution import ROOT, read, save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity
from exact_basis.propose import LIMITS
from exact_primal.propose import POLICY

REPORT = ROOT / 'docs/basis_compatibility_v1.json'
ANALYSIS = ROOT / 'docs/nonpositive_v1_analysis.json'
ARCHIVE = ROOT / 'docs/lp_diagnostic_v1_execution_results.json'


def assess(lp):
    n, na, ne = len(lp['c']), len(lp['b']), len(lp['h'])
    if len(lp['lower']) != n or len(lp['upper']) != n:
        raise ValueError('box dimensions')
    for name, rows in (('A', na), ('E', ne)):
        mat = lp[name]
        if (mat['shape'] != [rows, n] or len(mat['indptr']) != rows + 1 or
                mat['indptr'][0] != 0 or mat['indptr'][-1] != len(mat['data']) or
                len(mat['indices']) != len(mat['data']) or
                any(a > b for a, b in zip(mat['indptr'], mat['indptr'][1:])) or
                any(type(j) is not int or not 0 <= j < n for j in mat['indices'])):
            raise ValueError('CSR dimensions')
    nnz = len(lp['A']['data']) + len(lp['E']['data'])
    # These are necessary lower bounds from the actual constructor loops, not
    # predictions of elimination fill-in, rank or rational bit growth.
    dimensions = {'variables': n, 'A_rows': na, 'E_rows': ne,
                  'augmented_equations': na + ne, 'A_stored_nnz': len(lp['A']['data']),
                  'E_stored_nnz': len(lp['E']['data']), 'input_nnz': nnz,
                  'initial_scalar_visits_lower_bound': 3*n + na + ne + 1 + nnz,
                  'full_rank_pivot_nnz_lower_bound': na + ne}
    caps = {'variables': LIMITS['variables'], 'augmented_equations': LIMITS['equations'],
            'input_nnz': LIMITS['input_nnz'],
            'initial_scalar_visits_lower_bound': POLICY['max_operations'],
            'full_rank_pivot_nnz_lower_bound': LIMITS['live_elimination_nnz']}
    failures = {k: {'observed_or_necessary_minimum': dimensions[k], 'cap': cap}
                for k, cap in caps.items() if dimensions[k] > cap}
    return {'dimensions': dimensions, 'static_blockers': failures,
            'status': 'INCOMPATIBLE_CURRENT_LIMITS' if failures else 'NO_STATIC_BLOCKER_NOT_RUNTIME_APPROVAL',
            'rational_bits': 'NOT_SCANNED', 'actual_elimination_fill_in': 'NOT_COMPUTED',
            'rank_and_basis_mapping': 'NOT_INFERRED_FROM_DIMENSIONS',
            'freeze_eligible': False}


def bound_read(path, sha):
    raw = path.read_bytes()
    if digest(raw) != sha:
        raise ValueError('archived file drift: ' + str(path))
    return read(path)


def collect():
    from basis_supervised.controls import old
    from basis_supervised.flow import sources as supervised_sources
    old()
    controls = read(ROOT / 'docs/basis_supervised_controls_attempt002.json')
    if controls['status'] != 'PASS' or controls['sources'] != supervised_sources():
        raise ValueError('supervision freeze drift')
    analysis, archive = read(ANALYSIS), read(ARCHIVE)
    parent = ROOT / 'docs/reuse_supervised_v1_execution_results.json'
    if analysis['archive_sha256'] != digest(parent.read_bytes()):
        raise ValueError('analysis parent drift')
    parent_inventory = read(parent)['artifact_sha256']
    selected = [r for r in analysis['obligations'] if not r['positive']]
    if len(selected) != 30 or analysis['nonpositive'] != 30:
        raise ValueError('sealed nonpositive denominator')
    rows, refs = [], {}
    for ob in selected:
        paths = [name for name, sha in analysis['artifact_sha256'].items()
                 if name.endswith('_weighted.export.json') and sha == ob['weighted_export_sha256']]
        if len(paths) != 1:
            raise ValueError('unique original export binding')
        name = paths[0]
        if parent_inventory.get(name) != ob['weighted_export_sha256']:
            raise ValueError('parent export binding')
        path = (ROOT / name).resolve()
        if not path.is_relative_to(ROOT):
            raise ValueError('archive path escapes checkout')
        ex = bound_read(path, ob['weighted_export_sha256'])
        refs[name] = ob['weighted_export_sha256']
        rows.append({'dataset_index': ob['dataset_index'], 'pair': ob['pair'],
                     'property_index': ob['property_index'], 'request_id': ob['request_id'],
                     'export_path': name, 'export_sha256': ob['weighted_export_sha256'],
                     'lp_sha256': identity(ex['lp']), **assess(ex['lp'])})
    freeze_path = ROOT / 'docs/lp_diagnostic_v1_freeze.json'
    freeze = bound_read(freeze_path, archive['artifact_sha256'][str(freeze_path.relative_to(ROOT))])
    four = []
    for job, old_row in zip(freeze['jobs'], archive['rows'], strict=True):
        if job['job_id'] != old_row['job_id']:
            raise ValueError('archived four-job order')
        match = [r for r in rows if (r['dataset_index'], r['pair'], r['property_index']) ==
                 (job['dataset_index'], job['statement']['pair'], job['statement']['property_index'])]
        if len(match) != 1 or match[0]['lp_sha256'] != job['statement']['lp_sha256']:
            raise ValueError('original LP identity')
        d = match[0]['dimensions']
        if {k:d[k] for k in ('variables', 'A_rows', 'E_rows')} != old_row['dimensions']:
            raise ValueError('archived dimensions disagree')
        name = f'data/moe/results/lp_diagnostic_20260919_v1/{job["job_id"]}/proposal/native.json'
        sha = archive['artifact_sha256'][name]
        native = bound_read(ROOT / name, sha)
        refs[name] = sha
        if (native['lp_sha256'], native['statement_sha256']) != (match[0]['lp_sha256'], job['statement_sha256']):
            raise ValueError('native record binding')
        required = ('basis_valid', 'column_status', 'row_status', 'version', 'options',
                    'submitted', 'readback_before', 'readback_after')
        missing = [k for k in required if k not in native]
        if not missing:
            raise ValueError('new native schema requires separate review')
        four.append({'job_id': job['job_id'], **match[0], 'native_sha256': sha,
                     'saved_native_fields': sorted(native), 'missing_adapter_fields': missing,
                     'mapping_status': 'UNDETERMINED_NO_BOUND_BASIS_CAPTURE',
                     'saved_native_success_NOT_MAPPING_PROOF': native['success'],
                     'native_configuration_comparable': False,
                     'reason': 'legacy scipy linprog(method=highs,time_limit) is not the fixed native adapter capture protocol'})
    counts = Counter(r['status'] for r in rows)
    if counts != {'INCOMPATIBLE_CURRENT_LIMITS': 30}:
        # Fail closed: passing a shape check would not authorize auto-selection.
        decision = 'NOT_FROZEN_REQUIRES_SEPARATE_MAPPING_AND_SELECTION_REVIEW'
    else:
        decision = 'NOT_FROZEN_INCOMPATIBLE_CURRENT_LIMITS'
    docs = [ANALYSIS, ARCHIVE, parent, freeze_path,
            ROOT / 'docs/basis_supervised_controls_attempt002.json']
    refs.update({str(p.relative_to(ROOT)): digest(p.read_bytes()) for p in docs})
    return {'schema': 'BASIS_REAL_LP_COMPATIBILITY_V1', 'status': 'COMPATIBILITY_REVIEW_COMPLETED',
            'decision': decision, 'scope': 'sealed 30 nonpositive obligations; four prior diagnostics identified, no new selection',
            'denominator': 30, 'status_counts': dict(counts), 'obligations': rows, 'four_prior_diagnostics': four,
            'current_limits': dict(LIMITS), 'arithmetic_policy': dict(POLICY),
            'range': {k: {'min': min(r['dimensions'][k] for r in rows), 'max': max(r['dimensions'][k] for r in rows)}
                      for k in rows[0]['dimensions']},
            'artifact_sha256': refs,
            'sources': {**supervised_sources(), **{str(p.relative_to(ROOT)): digest(p.read_bytes())
                         for p in Path(__file__).parent.glob('*.py')}},
            'new_solver_calls': 0, 'new_reconstructions': 0, 'new_bound_checks': 0,
            'diagnostic_frozen': False, 'diagnostic_launched': False,
            'old_outcomes_unchanged': True,
            'interpretation': 'static incompatibility is an implementation limit, not LP infeasibility or model UNSAFE; missing basis is unknown, not an observed unsupported E-row status'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    result = collect()
    if args.audit:
        if result != read(REPORT):
            raise ValueError('saved compatibility report differs')
        save_new(ROOT / 'docs/basis_compatibility_v1_audit.json', {
            'status': 'PASS', 'issues': [], 'report_sha256': digest(REPORT.read_bytes()),
            'seconds': time.monotonic()-start, 'independent_process': True,
            'new_solver_or_reconstruction_or_bound_check_calls': 0,
            'scope': 'fresh-process original-file identity and size reconstruction; not mathematical proof'})
    else:
        save_new(REPORT, result)
    print(result['decision'], result['status_counts'], result['range'], flush=True)
