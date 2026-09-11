"""Frozen new-input three-arm confirmation, with a separate old-input smoke.

All methods recompute their own obligations under a hard process budget. No
resume, overwritten runs, performance-based stopping or automatic Git writes.
"""
import argparse
from collections import Counter
import fcntl
import json
import math
import os
from pathlib import Path
from statistics import mean, median
import subprocess
import sys
import time

import numpy as np
import scipy
import torch

from act.pipeline.moe.experiment1 import PROJECT_ROOT, WRITE_ROOT, _inside, _sha256, _git_value
from act.pipeline.moe.paired_followup import save, source_identity
from act.pipeline.moe.audit_staged_evidence import audit_evidence_package
from act.pipeline.moe.common_fact_snapshot import check_snapshot, fact_view, digest
from act.pipeline.moe.route_complexity_paired import counters
from act.pipeline.moe.schedule_confirmation_selection import verify_exclusions

DEFAULT = PROJECT_ROOT/'act/pipeline/moe/configs/schedule_confirmation_r1.json'
ARMS = ('adaptive', 'matched', 'legacy')
METHOD_HASHES = {
    'adaptive': '208e9ea19ada608ec72498a1768b5c56a2c5f3c3f745c89ecc99289ea37c39c0',
    'matched': '7c726d497e2eec4fe32b14d4a2e4acc725fd7a9b407e0c3b05eb2e4f5418dd1f',
    'legacy': 'bb05702ebcd61ae47db28f2d35fe03d2e84803b23fa3e2cc4328a954e00ff756',
}


def jobs(selection, smoke):
    result = []; models = sorted(selection['models'])
    samples = selection['smoke_samples' if smoke else 'samples']
    for rank, sample in enumerate(samples):
        for offset in range(len(models)):
            model = models[(rank + offset) % len(models)]
            for position in range(3):
                arm = ARMS[(rank + models.index(model) + position) % 3]
                result.append({'rank': rank, 'model': model, 'method': arm, 'position': position,
                               'dataset_index': sample['dataset_index'],
                               'job_id': f'rank{rank}_{model}_{arm}'})
    return result


def artifacts(config, deep=False):
    if config['budget_seconds'] != 300 or config['sample_count'] != 30 or set(config['methods']) != set(ARMS):
        raise ValueError('registered size/arms/budget changed')
    if config['primary_comparator'] != 'matched' or config['secondary_comparator'] != 'legacy':
        raise ValueError('comparator changed')
    if config['bootstrap'] != {'seed': 20260912, 'replicates': 10000, 'confidence': 0.95}:
        raise ValueError('statistics changed')
    selection_path = _inside(Path(config['selection']), PROJECT_ROOT)
    if _sha256(selection_path) != config['selection_sha256']:
        raise ValueError('selection drift')
    selection = json.loads(selection_path.read_text())
    review_path = _inside(Path(config['selection_audit']), PROJECT_ROOT)
    if _sha256(review_path) != config['selection_audit_sha256']:
        raise ValueError('selection audit drift')
    review = json.loads(review_path.read_text())
    if review['status'] != 'PASS' or review['issues'] or review['selection_sha256'] != config['selection_sha256']:
        raise ValueError('selection not audited')
    if (len(selection['samples']) != 30 or len(selection['smoke_samples']) != 1
            or selection['request'] != {'epsilon': 2/255, 'boundary_search': False, 'route_instability_prefilter': False}):
        raise ValueError('task size or semantics drift')
    if deep:
        verify_exclusions(selection)
        data = selection['dataset']
        if _sha256(_inside(Path(data['raw_test_batch']), WRITE_ROOT)) != data['raw_test_batch_sha256']:
            raise ValueError('raw dataset drift')
        for subject in selection['models'].values():
            if _sha256(_inside(Path(subject['checkpoint']), WRITE_ROOT)) != subject['checkpoint_sha256']:
                raise ValueError('checkpoint drift')
    configs = {}
    for arm in ARMS:
        entry = config['methods'][arm]; path = _inside(Path(entry['path']), PROJECT_ROOT)
        if entry['sha256'] != METHOD_HASHES[arm] or _sha256(path) != METHOD_HASHES[arm]:
            raise ValueError('frozen solver configuration changed')
        configs[arm] = json.loads(path.read_text())
    return selection, configs


def expected_identity(selection, job, cfg, smoke):
    model = selection['models'][job['model']]
    sample = selection['smoke_samples' if smoke else 'samples'][job['rank']]
    if sample['dataset_index'] != job['dataset_index']:
        raise ValueError('job/index mismatch')
    return {'model_state': model['model_state'],
            'checkpoint': {'path': model['checkpoint'], 'sha256': model['checkpoint_sha256']},
            **{key: sample[key] for key in ('center', 'lower', 'upper')},
            'property': {'kind': 'TOP1_ROBUST', 'clean_prediction': sample['label'], 'classes': 10},
            'epsilon': selection['request']['epsilon'], 'config_sha256': digest(cfg)}


def inspect_row(root, row, runtime, selection, configs):
    """Structural package/snapshot checks, including killed requests."""
    arm = row['method']; directory = root/row['job_id']; cfg = configs[arm]
    if (row['status'] not in {'SAFE', 'UNSAFE', 'UNKNOWN', 'TIMEOUT'} or row['budget_seconds'] != 300
            or not math.isfinite(row['wall_seconds']) or row['wall_seconds'] < 0):
        raise ValueError('invalid terminal or timing')
    expected = expected_identity(selection, row, cfg, runtime['smoke'])
    package = row.get('package'); e = None
    if row['outer_timeout']:
        if row['status'] != 'TIMEOUT' or package is not None:
            raise ValueError('outer timeout promoted')
    else:
        if not package or row['return_code'] != 0:
            raise ValueError('missing successful package')
        p = _inside(Path(package), root)
        if p != directory/'package' or _sha256(p/'manifest.json') != row['manifest_sha256']:
            raise ValueError('package hash/path mismatch')
        checked = audit_evidence_package(p, replay_unsafe=True)
        if checked['status'] != 'PASS': raise ValueError(f'package failed: {checked}')
        e = json.loads((p/'evidence.json').read_text())
        if (e['identity'] != expected or e['verdict']['status'] != row['status']
                or e['execution']['git_head'] != runtime['git_head']
                or e['execution']['dataset_index'] != row['dataset_index']
                or e['execution']['config_sha256'] != runtime['config']['methods'][arm]['sha256']
                or e['numerical_safety'] != cfg['numerical_safety']):
            raise ValueError('executed identity/semantics differ from frozen request')
    path = directory/'common_facts.json'; recorded = row.get('snapshot_sha256')
    facts = None; pair_count = None
    if path.exists() != (recorded is not None):
        raise ValueError('snapshot omitted from terminal or missing on disk')
    if recorded is not None:
        if arm == 'legacy' or _sha256(path) != recorded:
            raise ValueError('snapshot hash/arm mismatch')
        value = json.loads(path.read_text())
        check_snapshot(value, expected_identity=expected, expected_config=cfg, evidence=e)
        if value['payload']['completion_elapsed_seconds'] > row['wall_seconds']:
            raise ValueError('snapshot completion after request termination')
        facts = fact_view(value); pair_count = len(facts['pairs'])
    if e is not None:
        if arm != 'legacy' and e['route_complexity_schedule']['common_fact_prelude_complete'] and facts is None:
            raise ValueError('completed prelude lacks durable snapshot')
        if e['route_coverage']['route_sets_exact']:
            pairs = e['route_coverage']['feasible_route_sets']
            if facts is not None and pairs != facts['pairs']:
                raise ValueError('snapshot and final route coverage differ')
            pair_count = len(pairs)
    return {'facts': facts, 'pair_count': pair_count, 'package': e is not None,
            'replayed': row['status'] == 'UNSAFE', 'counters': counters(e) if e is not None else None}


def summarize(rows, details, selection, smoke):
    models = sorted(selection['models']); n = 1 if smoke else 30
    by = {(r['model'], r['rank'], r['method']): r for r in rows}
    results = {}; snapshots_equal = snapshots_unavailable = 0
    for model in models:
        contrasts = {}; methods = {}
        for rank in range(n):
            if {'SAFE', 'UNSAFE'} <= {by[model, rank, arm]['status'] for arm in ARMS}:
                raise ValueError('SAFE/UNSAFE conflict across arms')
            a, b = [details[model, rank, arm]['facts'] for arm in ('adaptive', 'matched')]
            if a is not None and b is not None:
                if a != b: raise ValueError('charged common facts differ')
                snapshots_equal += 1
            else: snapshots_unavailable += 1
        for arm in ARMS:
            values = [by[model, i, arm] for i in range(n)]
            methods[arm] = {'states': dict(Counter(r['status'] for r in values)),
                'mean_observed_seconds': mean(r['wall_seconds'] for r in values),
                'route_changing_safe_ranks': [i for i in range(n) if by[model,i,arm]['status']=='SAFE'
                    and (details[model,i,arm]['pair_count'] or 0)>1],
                'snapshot_available': sum(details[model,i,arm]['facts'] is not None for i in range(n)),
                'missing_packages': sum(not details[model,i,arm]['package'] for i in range(n))}
        for arm in ('matched', 'legacy'):
            record = {}
            for label, statuses in [('SAFE', {'SAFE'}), ('solved', {'SAFE', 'UNSAFE'})]:
                gain = [i for i in range(n) if by[model,i,'adaptive']['status'] in statuses and by[model,i,arm]['status'] not in statuses]
                loss = [i for i in range(n) if by[model,i,arm]['status'] in statuses and by[model,i,'adaptive']['status'] not in statuses]
                record[label] = {'gained': gain, 'lost': loss, 'net': len(gain)-len(loss), 'denominator': n}
            delta = [by[model,i,'adaptive']['wall_seconds']-by[model,i,arm]['wall_seconds'] for i in range(n)]
            record['mean_paired_observed_seconds'] = mean(delta)
            record['median_paired_observed_seconds'] = median(delta)
            # Legal-route strata are descriptive, never an alternative primary metric.
            strata = {}
            for kind in ('single', 'multiple', 'unavailable'):
                ranks = []
                for i in range(n):
                    counts = {details[model,i,a]['pair_count'] for a in ARMS} - {None}
                    if len(counts)>1: raise ValueError('exact pair counts disagree')
                    count = next(iter(counts)) if counts else None
                    if ('unavailable' if count is None else 'single' if count==1 else 'multiple') == kind: ranks.append(i)
                strata[kind] = {'ranks': ranks, 'adaptive_safe': sum(by[model,i,'adaptive']['status']=='SAFE' for i in ranks),
                                'baseline_safe': sum(by[model,i,arm]['status']=='SAFE' for i in ranks)}
            record['explanatory_strata'] = strata; contrasts[arm] = record
        results[model] = {'methods': methods, 'contrasts': contrasts}
    clustered = {}
    if not smoke:
        # The same resampled input carries all models, never 90 independent pairs.
        rng = np.random.default_rng(20260912)
        indices = rng.integers(0, n, size=(10000, n))
        for arm in ('matched', 'legacy'):
            clustered[arm] = {}
            for label, statuses in [('SAFE', {'SAFE'}), ('solved', {'SAFE','UNSAFE'})]:
                d = np.asarray([mean(int(by[m,i,'adaptive']['status'] in statuses)-int(by[m,i,arm]['status'] in statuses)
                                     for m in models) for i in range(n)])
                ci = np.quantile(d[indices].mean(1), [.025,.975]).tolist()
                clustered[arm][label] = {'mean_input_cluster_difference': float(d.mean()),
                                        'percentile_interval_95': ci, 'degenerate_observed_differences': bool(np.ptp(d)==0)}
    return {'models': results, 'input_clustered_contrasts': clustered,
            'common_fact_pairs_equal': snapshots_equal, 'common_fact_pairs_unavailable': snapshots_unavailable,
            'scope': '30 shared clean-correct inputs, 3 fixed same-family models; observed bounded costs, not uncensored speedup. Structural audits are not independent SAFE proofs. Degenerate bootstrap intervals are not equivalence evidence.'}


def audit(root):
    root = _inside(Path(root), WRITE_ROOT)
    runtime = json.loads((root/'runtime.json').read_text()); config = runtime['config']
    path = _inside(Path(runtime['config_path']), PROJECT_ROOT)
    if _sha256(path) != runtime['config_sha256'] or json.loads(path.read_text()) != config:
        raise ValueError('experiment config drift')
    selection, configs = artifacts(config, deep=True)
    expected = jobs(selection, runtime['smoke'])
    rows = [json.loads(line) for line in (root/'rows.jsonl').read_text().splitlines()]
    if len(rows) != len(expected) or any(any(row[k]!=job[k] for k in job) for row,job in zip(rows,expected)):
        raise ValueError('incomplete, duplicate or reordered jobs')
    details = {(r['model'],r['rank'],r['method']): inspect_row(root,r,runtime,selection,configs) for r in rows}
    result = summarize(rows, details, selection, runtime['smoke'])
    result.update(status='PASS', issues=[], rows=len(rows),
        packages=sum(d['package'] for d in details.values()), unsafe_replayed=sum(d['replayed'] for d in details.values()),
        snapshots_on_outer_timeout=sum(r['outer_timeout'] and r.get('snapshot_sha256') is not None for r in rows),
        rows_sha256=_sha256(root/'rows.jsonl'), runtime_sha256=_sha256(root/'runtime.json'))
    return result


def smoke_gate(config_path, config, source):
    root = _inside(Path(config['smoke_output']), WRITE_ROOT)
    if not (root/'runtime.json').exists() or not (root/'audit.final.json').exists():
        raise ValueError('full entry needs audited old-input smoke')
    rt = json.loads((root/'runtime.json').read_text())
    if not rt['smoke'] or rt['config_sha256'] != _sha256(config_path) or rt['source_sha256'] != source:
        raise ValueError('smoke source/config identity differs')
    checked = audit(root)
    if checked != json.loads((root/'audit.final.json').read_text()): raise ValueError('smoke audit differs')
    for arm in ARMS:
        if not any(v['methods'][arm]['missing_packages']==0 for v in checked['models'].values()):
            raise ValueError('smoke lacks complete evidence for an arm')
    if checked['common_fact_pairs_equal'] == 0: raise ValueError('smoke lacks common-fact conformance')
    return {'root': str(root), 'audit_sha256': _sha256(root/'audit.final.json')}


def run(config_path, smoke=False):
    if _git_value('branch','--show-current') != 'feat/moe-route-verification' or _git_value('status','--porcelain'):
        raise ValueError('clean feature checkout required')
    config_path = _inside(Path(config_path), PROJECT_ROOT); config = json.loads(config_path.read_text())
    selection, configs = artifacts(config, deep=True)
    if Path(sys.executable).resolve() != Path(config['python']).resolve(): raise ValueError('act-py312 required')
    source = source_identity(); gate = None if smoke else smoke_gate(config_path, config, source)
    root = _inside(Path(config['smoke_output' if smoke else 'output']), WRITE_ROOT)
    root.mkdir(exist_ok=False)
    runtime = {'config': config, 'config_path': str(config_path), 'config_sha256': _sha256(config_path),
               'git_head': _git_value('rev-parse','HEAD'), 'source_sha256': source, 'smoke': smoke, 'smoke_gate': gate,
               'started_unix': time.time(), 'python': sys.version, 'load_average_before': list(os.getloadavg())}
    runtime['versions'] = {'torch': torch.__version__, 'numpy': np.__version__, 'scipy': scipy.__version__}
    save(root/'runtime.json', runtime)
    env = {**os.environ, 'ACT_TORCHVISION_DATA_ROOT': str(PROJECT_ROOT/'data/torchvision'),
           'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
    schedule = jobs(selection, smoke)
    for number, job in enumerate(schedule, 1):
        if _git_value('status','--porcelain') or source_identity()!=source: raise ValueError('source/worktree drift')
        artifacts(config)
        directory = root/job['job_id']; directory.mkdir()
        command = [config['python'],'-m','act.pipeline.moe.staged_verifier',
                   '--checkpoint',selection['models'][job['model']]['checkpoint'],
                   '--dataset-index',str(job['dataset_index']),'--epsilon',repr(selection['request']['epsilon']),
                   '--config',config['methods'][job['method']]['path'], '--output-dir',str(directory/'package'),
                   '--progress-path',str(directory/'progress.json')]
        if job['method']!='legacy': command.extend(['--common-fact-snapshot',str(directory/'common_facts.json')])
        started = time.monotonic(); expired = False; code = None
        with (directory/'worker.log').open('x') as log:
            try:
                code = subprocess.run(command,cwd=PROJECT_ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
            except subprocess.TimeoutExpired: expired = True
        elapsed = time.monotonic()-started
        expired = expired or elapsed>300
        row = {**job, 'budget_seconds':300, 'wall_seconds':elapsed,'outer_timeout':expired,'return_code':code,
               'status':'TIMEOUT' if expired else 'ERROR','package':None,'snapshot_sha256':None,
               'load_average_after':list(os.getloadavg())}
        if (directory/'common_facts.json').exists(): row['snapshot_sha256'] = _sha256(directory/'common_facts.json')
        try:
            if not expired and code==0:
                p = directory/'package'
                e = json.loads((p/'evidence.json').read_text())
                row.update(status=e['verdict']['status'],package=str(p),manifest_sha256=_sha256(p/'manifest.json'))
            if row['status']!='ERROR': inspect_row(root,row,runtime,selection,configs)
        except Exception as exc:
            row.update(status='ERROR',error=f'{type(exc).__name__}: {exc}')
        save(directory/'terminal.json',row)
        with (root/'rows.jsonl').open('a') as f:
            f.write(json.dumps(row,sort_keys=True)+'\n'); f.flush(); os.fsync(f.fileno())
        print(f"{'smoke' if smoke else 'full'} {number}/{len(schedule)} {job['job_id']} {row['status']}",flush=True)
        if row['status']=='ERROR': raise RuntimeError('worker/audit failure retained; no replacement')
    save(root/'audit.final.json',audit(root))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=DEFAULT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--pipeline',action='store_true'); mode.add_argument('--smoke',action='store_true')
    mode.add_argument('--audit',type=Path)
    args = parser.parse_args()
    if args.audit:
        print(json.dumps(audit(args.audit),indent=2)); return
    with (PROJECT_ROOT/'data/moe/results/route_complexity_pairing.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.pipeline:
            run(args.config,True);run(args.config,False)
        else: run(args.config,args.smoke)


if __name__ == '__main__': main()
