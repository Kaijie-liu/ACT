"""Read-only derivation of synthetic H1 supervision results; never solves."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load, save, sha
from scoped_source.sparse_supervised import audit, POSITIVE

CASES = {'success':POSITIVE,'full':POSITIVE,'ties':POSITIVE,
    'negative':'UNKNOWN_NONPOSITIVE','missing-cert':'UNKNOWN_MISSING_EVIDENCE',
    **{n:'ERROR' for n in ('source-mismatch','omit_property','exception_after_bundle','partial_output','wrong_invocation','descendant','rebind-checker-context')},
    **{n:'TIMEOUT' for n in ('produce_delay','check_delay','receive_delay','late_publish','actual-publication-overrun','prelaunch-expiry')},
    'rss':'RESOURCE_LIMIT'}


def derive(root):
    bindings = load(root/'implementation.json')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name) != digest or sha(ROOT/name) != digest:
            raise ValueError('use the bound implementation to reproduce this derivation')
    records = []
    for name,expected in CASES.items():
        path = root/name; observed = load(path/'caller_observation.json')
        if observed['status'] != expected: raise ValueError('unexpected control terminal: '+name)
        if name == 'rebind-checker-context':
            try: audit(path,observed)
            except ValueError as error:
                if str(error) != 'invocation identity': raise
                checked = {'status':'EXPECTED_REJECTION','reason':str(error),'positive_execution_accepted':False}
            else: raise ValueError('changed trust anchor accepted')
        else: checked = audit(path,observed)
        terminal = load(path/'terminal.json')
        record = {'case':name,'status':observed['status'],'seconds':observed['seconds'],
            'audit':checked, 'receipt_sha256':sha(path/'receipt.json'),
            'caller_observation_sha256':sha(path/'caller_observation.json'),
            'terminal_sha256':sha(path/'terminal.json'),
            'stage_seconds':{s['phase']:s['seconds'] for s in terminal['stages']},
            'publication_and_overhead_seconds':observed['seconds']-terminal['stage_seconds']}
        if expected in (POSITIVE,'UNKNOWN_NONPOSITIVE','UNKNOWN_MISSING_EVIDENCE'):
            accepted = load(path/'accepted.json'); result = accepted['result']
            record.update(required=accepted['required'],positive=accepted['positive'],missing=accepted['missing'],
                reused=result['reused'], source_blocks=result['source_blocks_checked'],
                source_nodes=result['source_nodes'], lp_rows=result['lp_rows_reconstructed'],
                bounds=[r['lower_bound'] for r in result['obligations']],
                bundle_bytes=load(path/'built.json')['bundle_bytes'],
                source_sha256=accepted['source_sha256'])
        records.append(record)
    by_name = {r['case']:r for r in records}
    if by_name['success']['bounds'] != by_name['full']['bounds']: raise ValueError('paired bound differential')
    return {'schema':'H1_SUPERVISION_CONTROL_ARCHIVE_V1','root':str(root),'cases':records,
        'terminal_counts':dict(Counter(r['status'] for r in records)), 'source_bindings':bindings,
        'unittest_controls_reported':22, 'real_requests_started':0, 'new_solves_during_archive':0,
        'performance_claim':False, 'paired_bounds_equal':True,
        'scope':'Synthetic declared real Linear/ReLU top2; controlled failures; not native FP or real-model comparison.',
        'cost_scope':'API entry through observed return; includes workers/imports/source/LP/pack/check/receive/publication/cleanup; external test-driver logging and later audit separate.',
        'failed_history':[
            {'root':'/data1/Kane/MOE/baseline_runs/h1_supervised_controls_20260930_r1',
             'tests':15,'failed':1,'reason':'negative control falsely ERROR: checker child appeared in stale pre-poll process snapshot; fixed with same-PID exec; old artifacts retained'},
            {'root':'/data1/Kane/MOE/baseline_runs/h1_supervised_controls_20260930_r2',
             'tests':18,'failed':0,'disposition':'development only; further publication/cost controls and trust-anchor audit followed'},
            {'root':'/data1/Kane/MOE/baseline_runs/h1_supervised_controls_20260930_r3',
             'tests':21,'failed':0,'disposition':'development only; review found producer-writable checker context, fixed and negative-tested in final version'}]}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--output',type=Path); group.add_argument('--check',type=Path)
    a = p.parse_args(); report = derive(a.root)
    if a.output: save(a.output,report)
    elif load(a.check) != report: raise ValueError('archive changed')
    print({'status':'PASS','cases':len(report['cases']),'terminal_counts':report['terminal_counts'],
           'real_requests_started':0,'new_solves':0})
