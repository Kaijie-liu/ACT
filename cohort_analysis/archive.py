"""Read-only re-review and cost analysis of the frozen general-evidence cohort.

No solver, generation, retry or terminal promotion is introduced here. Fresh
per-request reviews use the original auditor; local prechecks are observations
whose file identities and arithmetic counts are checked, not fresh LP proofs.
"""
import argparse
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data/moe/results/general_evidence_cohort_20260916_v1'
OUT = ROOT / 'data/moe/results/general_evidence_archive_20260916_v1'
ACT = '/data1/Kane/miniconda3/envs/act-py312/bin/python'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_new(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def local_precheck(value, request, manifest):
    """Validate saved result accounting without claiming to reprove its bounds."""
    expected = {(tuple(p), i) for p in manifest['routes']['feasible']
                for i in range(len(request['properties']))}
    rows = value['obligations']
    keys = [(tuple(r['pair']), r['property_index']) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError('precheck obligations missing, duplicated or replaced')
    counts = Counter()
    for row in rows:
        bound = row.get('lower_bound')
        if bound is None:
            if row['state'] != 'MISSING_EVIDENCE':
                raise ValueError('missing bound called checked')
            counts['missing'] += 1
        elif Fraction(bound) > Fraction(manifest['positive_threshold']):
            if row['state'] not in ('CHECKED_RATIONAL_POSITIVE', 'CHECKED_REUSED_POSITIVE'):
                raise ValueError('positive state differs')
            counts['positive'] += 1
        else:
            if row['state'] != 'CHECKED_NONPOSITIVE_OR_BELOW_THRESHOLD':
                raise ValueError('nonpositive state differs')
            counts['nonpositive'] += 1
    if value['required_obligations'] != len(expected) or any(
        value[k + '_obligations'] != counts[k] for k in ('positive', 'missing', 'nonpositive')
    ):
        raise ValueError('precheck accounting differs')
    if value['route_pairs'] != manifest['routes']['feasible']:
        raise ValueError('precheck routes differ')
    status = ('UNKNOWN_MISSING_EVIDENCE' if counts['missing'] or not manifest['generation_complete'] else
              'UNKNOWN_NONPOSITIVE' if counts['nonpositive'] else 'CHECKED_CONDITIONAL')
    if value['status'] != status:
        raise ValueError('precheck status differs')
    return {'status': status, 'required': len(expected), **{k: counts[k] for k in
            ('positive', 'nonpositive', 'missing')},
            'nonpositive_rows': [r for r in rows if r.get('lower_bound') is not None
                                 and Fraction(r['lower_bound']) <= Fraction(manifest['positive_threshold'])],
            'scope': 'saved local precheck; hash/accounting checked, not freshly re-proved or terminal-promoted'}


def cost_record(directory, terminal, detail):
    phases = terminal['stages']
    completed = {k: v['elapsed_seconds'] for k, v in phases.items() if v['state'] == 'COMPLETED'}
    ended = {k: v['elapsed_seconds'] for k, v in phases.items() if v['state'] != 'COMPLETED'}
    censored = terminal['censored_phase']
    window = censored['observed_phase_window_seconds'] if censored else 0
    accounted = sum(completed.values()) + sum(ended.values()) + window
    overhead = terminal['wall_seconds'] - accounted
    if not math.isfinite(overhead) or overhead < -.02:
        raise ValueError('phase time overlaps or exceeds request')
    calls = read(directory/'query_log.json') if (directory/'query_log.json').exists() else []
    call_times = [c['seconds'] for c in calls if 'seconds' in c]
    if any(not math.isfinite(t) or t < 0 for t in call_times):
        raise ValueError('invalid proposal wrapper duration')
    propose_elapsed = phases.get('propose', {}).get('elapsed_seconds')
    residual = propose_elapsed - sum(call_times) if propose_elapsed is not None else None
    if residual is not None and residual < -.02:
        raise ValueError('proposal wrappers exceed phase')
    inventory = terminal['artifact_sha256']; sizes = {p: (directory/p).stat().st_size for p in inventory}
    unique = {}
    for p, h in inventory.items():
        if h in unique and unique[h] != sizes[p]:
            raise ValueError('same hash different byte size')
        unique[h] = sizes[p]
    stop = censored['phase'] if censored else next(
        (k for k, v in phases.items() if v['state'] != 'COMPLETED'), 'complete')
    return {'dataset_index': terminal['dataset_index'], 'terminal': terminal['status'],
            'wall_seconds': terminal['wall_seconds'], 'stop_phase': stop,
            'outer_timeout': terminal['outer_timeout'], 'completed_phase_seconds': completed,
            'noncompleted_phase_elapsed_seconds': ended, 'censored_phase_window': censored,
            'unassigned_outer_seconds': overhead,
            'proposal_calls': len(calls), 'proposal_states': dict(Counter(c['status'] for c in calls)),
            'proposal_wrapper_seconds': sum(call_times) if calls else None,
            'propose_phase_minus_recorded_wrappers_seconds': residual,
            'native_solver_exclusive_seconds': None, 'serialization_exclusive_seconds': None,
            'propagation_exclusive_seconds': None,
            'cost_scope': 'phase/wrapper elapsed time; residual is not exclusively serialization or checking',
            'artifact_files': len(sizes), 'artifact_logical_bytes': sum(sizes.values()),
            'identical_file_content_bytes': sum(unique.values()), 'packing': detail['proof_size'],
            'saved_precheck': None}


def one(job_id, output):
    from evidence_cohort.audit import check_one
    from evidence_cohort.contract import verify_freeze, selection, request_for
    verify_freeze(); selected = selection(); rt = read(RAW/'runtime.json')
    rows = [json.loads(s) for s in (RAW/'rows.jsonl').read_text().splitlines()]
    row = next(r for r in rows if r['job_id'] == job_id)
    job = next(j for j in selected['jobs'] if j['job_id'] == job_id)
    req = request_for(selected, job, rt['git_head'])
    if read(RAW/job_id/'request.json') != req:
        raise ValueError('frozen request mismatch')
    detail = check_one(RAW, row, req)
    if detail['status'] != 'PASS' or detail['issues']:
        raise ValueError('request re-review failed')
    write_new(output, detail)


def archive():
    from evidence_cohort.audit import roster, summarize
    from evidence_cohort.contract import verify_freeze, selection
    verify_freeze(); selected = selection(); rt = read(RAW/'runtime.json')
    from evidence_cohort.run import resource, resource_ok
    if not resource_ok(resource()):
        raise RuntimeError('resources unavailable for archival review; no run started')
    rows = [json.loads(s) for s in (RAW/'rows.jsonl').read_text().splitlines()]
    final = read(RAW/'audit.final.json'); end = read(RAW/'run_terminal.json')
    roster(rows, selected['jobs'], end)
    if final['status'] != 'PASS' or final['issues'] or end['state'] != 'EXECUTION_COMPLETED':
        raise ValueError('no completed audited cohort')
    if final['rows_sha256'] != sha(RAW/'rows.jsonl') or final['run_terminal_sha256'] != sha(RAW/'run_terminal.json'):
        raise ValueError('final audit identity differs')
    OUT.mkdir(exist_ok=False)
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='')
    before = {name: sha(RAW/name) for name in ('audit.final.json', 'rows.jsonl', 'runtime.json', 'run_terminal.json', 'audit_execution.json')}
    started = time.monotonic(); details = []
    for row in rows:
        job = row['job_id']; dest = OUT/(job+'.json')
        with (OUT/(job+'.log')).open('x') as log:
            subprocess.run([ACT, '-m', 'cohort_analysis.archive', '--one', job, '--output', str(dest)],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
        details.append(read(dest))
        print(f"REVIEW {len(details)}/{len(rows)} {job} PASS", flush=True)
    summary = summarize(rows, details, selected['jobs'], True)
    if summary != final['summary']:
        raise ValueError('recomputed summary differs from frozen final audit')
    if details != final['details']:
        raise ValueError('fresh detail review differs from original audit')
    evidence = []; bindings = []
    for row, detail in zip(rows, details):
        directory = RAW/row['job_id']; terminal = read(directory/'terminal.json')
        bindings.append({'job_id': row['job_id'], 'dataset_index': row['dataset_index'],
                         'arm': row['arm'], 'status': row['status'],
                         'terminal_sha256': sha(directory/'terminal.json'),
                         'request_sha256': sha(directory/'request.json'),
                         'fresh_review_sha256': sha(OUT/(row['job_id']+'.json'))})
        if row['arm'] != 'evidence':
            continue
        cost = cost_record(directory, terminal, detail)
        manifest = read(directory/'manifest.json'); request = read(directory/'request.json')['evidence_request']
        cost['pair_count'] = len(manifest['routes']['feasible'])
        cost['required_obligations'] = len(manifest['routes']['feasible'])*len(request['properties'])
        cost['manifest_obligation_kinds'] = dict(Counter(r['kind'] for r in manifest['obligations']))
        cost['manifest_weighted_states'] = dict(Counter(r.get('weighted_status', 'not_residual') for r in manifest['obligations']))
        cost['saved_source_sha256'] = {f: sha(directory/f) for f in
            ('manifest.json', 'query_log.json', 'independent.json', 'packing.json', 'check.log') if (directory/f).exists()}
        if (directory/'independent.json').exists():
            cost['saved_precheck'] = local_precheck(read(directory/'independent.json'), request, manifest)
        evidence.append(cost)
    observed = [r['saved_precheck'] for r in evidence if r['saved_precheck'] is not None]
    union = sorted({r['dataset_index'] for r in rows if r['status'] == 'UNSAFE'})
    unsafe_sets = {a: {r['dataset_index'] for r in rows if r['arm'] == a and r['status'] == 'UNSAFE'} for a in ('matched', 'crown')}
    phases = {}
    for phase in ('capture', 'propose', 'precheck', 'package', 'check'):
        times = [r['completed_phase_seconds'][phase] for r in evidence if phase in r['completed_phase_seconds']]
        phases[phase] = {'completed_count': len(times), 'completed_sum_seconds': sum(times),
                         'completed_mean_seconds': mean(times) if times else None,
                         'completed_median_seconds': median(times) if times else None,
                         'scope': 'completion-conditioned; missing/censored executions are not zero-cost'}
    analysis = {'stop_phase_counts': dict(Counter(r['stop_phase'] for r in evidence)),
                'saved_prechecks': len(observed), 'saved_precheck_statuses': dict(Counter(r['status'] for r in observed)),
                'no_saved_precheck': len(evidence)-len(observed),
                'saved_precheck_obligations': {k: sum(r[k] for r in observed) for k in ('required', 'positive', 'nonpositive', 'missing')},
                'phases': phases, 'replayed_unsafe_input_union': union,
                'unsafe_intersection': sorted(unsafe_sets['matched'] & unsafe_sets['crown']),
                'matched_only_unsafe': sorted(unsafe_sets['matched']-unsafe_sets['crown']),
                'crown_only_unsafe': sorted(unsafe_sets['crown']-unsafe_sets['matched']),
                'potentially_safe_inputs_at_most': 20-len(union),
                'no_free_cross_arm_reuse': True,
                'precheck_scope': 'saved observations only; no fresh bound proposal or complete request promotion',
                'single_pair_inputs': sum(r['pair_count'] == 1 for r in evidence),
                'multi_pair_inputs': sum(r['pair_count'] > 1 for r in evidence)}
    after = {name: sha(RAW/name) for name in before}
    if before != after:
        raise ValueError('original cohort records changed')
    result = {'schema': 'GENERAL_EVIDENCE_ARCHIVE_REVIEW_V1', 'status': 'PASS', 'issues': [],
              'execution_head': rt['git_head'], 'execution_freeze_sha256': rt['freeze_sha256'],
              'raw_root': str(RAW.relative_to(ROOT)), 'raw_identity': before,
              'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'selection_sha256': selected['protocol'].get('selection_sha256', sha(ROOT/'docs/general_evidence_v1_selection.json')),
              'new_solver_queries': 0, 'original_terminals_unchanged': True,
              'fresh_request_reviews': len(details), 'replayed_unsafe_runs': sum(d['replayed'] for d in details),
              'summary': summary, 'analysis': analysis, 'evidence_rows': evidence,
              'terminal_bindings': bindings, 're_review_seconds': time.monotonic()-started,
              'scope': 'fresh source/terminal/witness audit plus saved-log accounting; not reproof of partial LP results'}
    write_new(OUT/'review.json', result)
    write_new(ROOT/'docs/general_evidence_execution_v1_results.json', result)
    print(json.dumps({'status': result['status'], 'analysis': analysis}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--one'); p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.one:
        if args.output is None or args.output.parent.resolve() != OUT.resolve():
            raise ValueError('review output must be in new archive directory')
        one(args.one, args.output)
    else:
        archive()
