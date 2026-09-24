"""Saved-only archive of the single frozen 4088 request; never launches work.

Run with act-py312 python -S. This receipt/identity audit does not manufacture
missing source constructions, candidates, or lower-bound certificates.
"""
import argparse
import copy
import itertools
import json
import math
from pathlib import Path
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def forbid(event, args):
    if event == 'import' and args[0].split('.')[0] in (
            'torch', 'numpy', 'scipy', 'highspy', 'gurobipy', 'act'):
        raise ImportError('saved-only archive must not load model or solver')
    if event.startswith(('subprocess.', 'socket.')) or event in ('os.system', 'os.fork', 'os.exec'):
        raise PermissionError('saved-only archive cannot execute external work')


sys.addaudithook(forbid)
from scoped_proof.audit import audit
from scoped_proof.io import load, save, sha
from source_enclosure.format import identity

CONFIG = ROOT / 'configs/backend_controls/scoped_proof_execution_r1.json'
REVIEW = ROOT / 'docs/scoped_proof_freeze_review_20260924_r1.json'
AUDIT = ROOT / 'docs/scoped_proof_execution_audit_20260924_r1.json'
ARCHIVE = ROOT / 'docs/scoped_proof_execution_archive_20260924_r1.json'
LAUNCH_HEAD = 'bc1716e706cb0ded60d6c941bc3aadabe3946520'


def stable_audit(value):
    return {k: v for k, v in value.items() if k != 'separate_audit_seconds'}


def collect():
    started = time.monotonic()
    review = load(REVIEW)
    cfg = load(CONFIG, review['config_sha256'])
    root = Path(cfg['output'])
    if load(root / 'spec.json') != cfg:
        raise ValueError('executed spec differs from frozen config')
    if sha(root / 'spec.json') != sha(CONFIG):
        raise ValueError('executed spec byte identity')
    for name, digest in cfg['sources'].items():
        if sha(ROOT / name) != digest:
            raise ValueError('frozen execution source changed: ' + name)
    parent = load(ROOT / cfg['source_protocol']['path'], cfg['source_protocol']['sha256'])
    scope = cfg['scope']
    obligations = [{'pair': list(p), 'label': scope['label'], 'competitor': k}
        for p in itertools.combinations(range(scope['experts']), 2)
        for k in range(scope['classes']) if k != scope['label']]
    if obligations != parent['output_obligations'] or len(obligations) != 252:
        raise ValueError('complete frozen obligation inventory changed')
    fresh = audit(root)
    terminal = load(root / 'terminal.json')
    cost = load(root / 'cost.json')
    stages = terminal['stages']
    observations = []
    for stage in stages:
        p = root / (stage['phase'] + '_events.jsonl')
        events, open_operation = [], None
        for line in p.read_text().splitlines() if p.exists() else []:
            e = json.loads(line)
            if (not math.isfinite(e['elapsed']) or
                    not stage['start_seconds'] <= e['elapsed'] <= stage['end_seconds']):
                raise ValueError('event outside charged phase')
            if e['event'] == 'ENTER':
                if open_operation is not None:
                    raise ValueError('overlapping event operations')
                open_operation = e
            elif e['event'] in ('EXIT', 'EXIT_ERROR'):
                if open_operation is None or open_operation['operation'] != e['operation']:
                    raise ValueError('unbound operation exit')
                open_operation = None
            else:
                raise ValueError('unknown event')
            events.append(e)
        observations.append({'phase': stage['phase'], 'events': events,
            'open_operation_at_stop': open_operation,
            'open_operation_observed_seconds': None if open_operation is None else
                stage['end_seconds'] - open_operation['elapsed']})
    files = {}
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError('raw artifact symlink')
        if p.is_file():
            files[str(p.relative_to(root))] = {'sha256': sha(p), 'bytes': p.stat().st_size}
    candidates = sorted(str(p.relative_to(root)) for p in (root / 'candidates').glob('*.json'))
    source_checked = load(root / 'source_check.json') if (root / 'source_check.json').exists() else None
    evidence = load(root / 'evidence_check.json') if (root / 'evidence_check.json').exists() else None
    checked = 0 if evidence is None else evidence['checked_bounds']
    positive = 0 if evidence is None else evidence['positive_bounds']
    native_calls = [e for s in observations for e in s['events']
                    if e['event'] == 'ENTER' and e['operation'].startswith('native_lp_')]
    if not any(s['phase'] == 'propose' for s in stages) and (candidates or native_calls or evidence):
        raise ValueError('evidence exists without proposal stage')
    result = {'schema': 'SCOPED_PROOF_EXECUTION_ARCHIVE_V1', 'audit': 'PASS', 'issues': 0,
        'raw_root': str(root), 'config_sha256': sha(CONFIG),
        'implementation_commit': cfg['implementation_commit'],
        'launch_head_operator_observed_clean': LAUNCH_HEAD,
        'frozen_source_files_rechecked': len(cfg['sources']),
        'scope_sha256': identity(scope), 'dataset_index': parent['sample']['dataset_index'],
        'required_output_obligations': len(obligations), 'request_count': 1,
        'effective_status': fresh['effective_status'], 'cost': cost, 'stages': stages,
        'operation_observations': observations, 'source_check': source_checked,
        'evidence': {'source_captured': (root / 'source.json').exists(),
            'construction_published': (root / 'construction.json').exists(),
            'source_independently_checked': source_checked is not None,
            'native_lp_calls_started': len(native_calls), 'candidate_files': candidates,
            'exact_bounds_checked': checked, 'positive_bounds': positive,
            'obligations_without_checked_bound': len(obligations) - checked,
            'complete_output_positive_proof': fresh['complete_output_positive_proof'],
            'native_float_proof': False, 'route_changing_established': False},
        'files': files, 'total_raw_bytes': sum(v['bytes'] for v in files.values()),
        'frozen_audit': stable_audit(fresh), 'archive_source_sha256': sha(__file__),
        'saved_only_archive_seconds': time.monotonic() - started,
        'new_solves_during_archive': 0,
        'scope': 'saved identity, phase/cost/receipt and available exact evidence; no recovery of missing proof',
        'decision': 'ONE_FROZEN_ATTEMPT_SEALED; no retry, added budget, sample/row change or historical proof reuse'}
    return result, fresh


def receipt_mutation_controls(root):
    """Small receipt copies only: no checkpoint, matrix, solve, or propagation."""
    names = ['invocation.json', 'spec.json', 'cost.json', 'terminal.json', 'receipt.json']
    names += [p.name for p in root.glob('*_stage.json')]
    original = {name: load(root / name) for name in names}
    if original['cost.json']['complete_output_positive_proof']:
        raise ValueError('these receipt-prefix controls are for this incomplete run')
    mutations = {
        'budget': ('cost.json', 'budget_seconds', 301),
        'invocation': ('cost.json', 'invocation', 'wrong-invocation'),
        'cost': ('cost.json', 'stage_seconds', -1),
        'terminal_hash': ('cost.json', 'terminal_sha256', '0' * 64),
        'receipt_hash': ('cost.json', 'receipt_sha256', '0' * 64),
        'false_positive': ('cost.json', 'complete_output_positive_proof', True),
    }
    for name, mutation in [('unchanged', None), *mutations.items()]:
        with tempfile.TemporaryDirectory(prefix='receipt_audit_', dir=root.parent) as folder:
            copied = copy.deepcopy(original)
            if mutation is not None:
                filename, key, value = mutation
                copied[filename][key] = value
            for filename, value in copied.items():
                save(Path(folder) / filename, value)
            try:
                audit(folder, recheck=False)
            except ValueError:
                if mutation is None:
                    raise
            else:
                if mutation is not None:
                    raise AssertionError('accepted corruption: ' + name)
    return {'unchanged_receipt_accepted': 1, 'corruptions_rejected': len(mutations),
        'new_solves': 0, 'scope': 'receipt integrity, not new mathematical controls'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--check', action='store_true')
    p.add_argument('--receipt-controls', action='store_true')
    args = p.parse_args()
    if not sys.flags.no_site:
        p.error('use python -S for a saved-only archive')
    if args.receipt_controls:
        print(json.dumps(receipt_mutation_controls(Path(load(CONFIG)['output']))))
    else:
        value, fresh = collect()
        if args.check:
            if stable_audit(fresh) != stable_audit(load(AUDIT)):
                raise ValueError('saved audit differs')
            old = load(ARCHIVE)
            normalize = lambda v: {k: x for k, x in v.items() if k != 'saved_only_archive_seconds'}
            if normalize(value) != normalize(old):
                raise ValueError('archive drift')
            print('PASS: saved-only archive and independent receipt audit reproduce; zero new solves')
        else:
            if AUDIT.exists() or ARCHIVE.exists():
                raise FileExistsError('archive already exists; use --check')
            save(AUDIT, fresh)
            save(ARCHIVE, value)
            print(json.dumps({'effective_status': value['effective_status'], 'evidence': value['evidence']}))
