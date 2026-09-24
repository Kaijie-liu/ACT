"""Saved-only timing/identity review plus unchanged exact construction checks."""
import argparse
from collections import defaultdict
import math
from pathlib import Path
import statistics
import time
from scoped_proof.io import ROOT, load, save, sha
from scoped_source.check import check
from source_enclosure.format import identity

CONFIG = ROOT/'configs/backend_controls/source_construction_parse_r1.json'
RECORD = ROOT/'docs/source_construction_parse_results_20260924_r1.json'


def review():
    started = time.monotonic(); cfg = load(CONFIG); root = Path(cfg['output'])
    launch = load(root/'launch.json')
    if launch['config'] != cfg or launch['config_sha256'] != sha(CONFIG): raise ValueError('launch/config identity')
    for name, digest in cfg['sources'].items():
        if sha(ROOT/name) != digest: raise ValueError('implementation drift')
    execution = load(root/'execution.json')['rows']
    if [{k: r[k] for k in ('id','fixture','round','mode')} for r in execution] != cfg['calls']:
        raise ValueError('missing/reordered timing calls')
    rows = []; expected = {}; checked_once = set(); groups = defaultdict(list)
    for item, runtime in zip(cfg['calls'], execution):
        path = root/item['id']; plan = load(path/'plan.json'); c = load(path/'cost.json')
        t = load(path/'terminal.json', c['terminal_sha256'])
        if t['plan_sha256'] != sha(path/'plan.json') or plan['fixture'] != cfg['fixtures'][item['fixture']] or plan['mode'] != item['mode']:
            raise ValueError('call/fixture identity')
        if (c['budget_seconds'] != cfg['budget_seconds'] or
            abs(c['end_to_end_seconds'] - c['stage_seconds'] - c['overhead_seconds']) > 1e-8 or
            abs(c['stage_seconds'] - sum(p['seconds'] for p in t['phases'])) > 1e-8 or
            t['positive_certificate'] is not False): raise ValueError('budget/cost/proof grade')
        status = 'TIMEOUT' if (path/'publication_timeout.json').exists() else c['status']
        if runtime['status'] != status or runtime['seconds'] < c['end_to_end_seconds']:
            raise ValueError('execution/cost terminal mismatch')
        if status not in ('COMPLETED_CONSTRUCTION_CHECK_ONLY', 'TIMEOUT', 'ERROR', 'RESOURCE_LIMIT'):
            raise ValueError('unexpected construction terminal')
        if any(not math.isfinite(c[k]) or c[k] < 0 for k in ('end_to_end_seconds','stage_seconds','overhead_seconds')):
            raise ValueError('invalid costs')
        if c['end_to_end_seconds'] >= cfg['budget_seconds'] and status != 'TIMEOUT':
            raise ValueError('late result accepted')
        if [p['phase'] for p in t['phases']] != ['build','check'][:len(t['phases'])]:
            raise ValueError('missing or reordered phases')
        previous = 0.
        for phase in t['phases']:
            if load(path/(phase['phase']+'_stage.json')) != phase or not previous <= phase['start_seconds'] <= phase['end_seconds'] <= c['end_to_end_seconds']:
                raise ValueError('stage identity/time')
            previous = phase['end_seconds']
        report = None; mathematical = None
        if status == 'COMPLETED_CONSTRUCTION_CHECK_ONLY':
            if [p['phase'] for p in t['phases']] != ['build','check'] or any(p['status'] != 'COMPLETED' for p in t['phases']):
                raise ValueError('incomplete successful process coverage')
            b = load(path/'build_receipt.json'); checked = load(path/'check_receipt.json'); report = b['report']
            signature = (sha(path/'source.json'), sha(path/'construction.json'), identity(checked['result']))
            if signature[:2] != (checked['source_sha256'], checked['construction_sha256']): raise ValueError('artifact identity')
            if b['source']['sha256'] != signature[0] or b['construction']['sha256'] != signature[1] or t['source_hash'] != signature[0] or t['construction_hash'] != signature[1]:
                raise ValueError('unbound source/bundle')
            if item['fixture'] in expected and expected[item['fixture']] != signature:
                raise ValueError('modes do not construct exactly the same problem')
            expected[item['fixture']] = signature
            if report['parser'] is not None and any(report['parser'][k] != 0 for k in ('live_entries','live_bytes','live_cells')):
                raise ValueError('cache not cleared')
            if signature not in checked_once:
                doc, bundle = load(path/'source.json'), load(path/'construction.json')
                mathematical = check(doc,bundle,expected_source_sha256=identity(doc),deadline=time.monotonic()+30)
                if mathematical != checked['result']: raise ValueError('fresh exact construction recheck differs')
                checked_once.add(signature)
        seconds = {p['phase']: p['seconds'] for p in t['phases']}
        row = {**item, 'status': status, 'phase_seconds': seconds, 'complete_seconds': runtime['seconds'],
            'cost_receipt_seconds': c['end_to_end_seconds'], 'peak_rss': c['sampled_peak_rss'],
            'report': report, 'fresh_mathematical_recheck': mathematical,
            'raw_hashes': {p.name: sha(p) for p in path.iterdir() if p.is_file()}}
        rows.append(row); groups[(item['fixture'],item['mode'])].append(row)
    summary = []
    for (fixture, mode), values in groups.items():
        summary.append({'fixture': fixture, 'mode': mode, 'calls': len(values),
            'completed': sum(r['status'] == 'COMPLETED_CONSTRUCTION_CHECK_ONLY' for r in values),
            'median_full_seconds': statistics.median(r['complete_seconds'] for r in values),
            'median_build_seconds': statistics.median(r['phase_seconds']['build'] for r in values) if all('build' in r['phase_seconds'] for r in values) else None,
            'median_check_seconds': statistics.median(r['phase_seconds']['check'] for r in values) if all('check' in r['phase_seconds'] for r in values) else None,
            'max_sampled_rss': max(r['peak_rss'] for r in values)})
    return {'audit':'PASS','issues':0,'config_sha256':sha(CONFIG),'launch_head':launch['head'],
        'rows':rows,'summary':summary,'new_real_requests':0,'new_solver_calls':0,'positive_certificates':0,
        'exact_construction_rechecks':len(checked_once), 'review_source_sha256':sha(__file__),
        'separate_review_seconds':time.monotonic()-started,
        'scope':'fixed synthetic construction/check timing; not real-request latency or new certificates'}


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--check',action='store_true');a=p.parse_args();r=review()
    if a.check:
        omit=lambda v:{k:x for k,x in v.items() if k!='separate_review_seconds'}
        if omit(r)!=omit(load(RECORD)): raise ValueError('saved review drift')
        print('PASS: saved-only archive and exact construction rechecks reproduce; zero solves')
    else:
        save(RECORD,r);print(r['summary'])
