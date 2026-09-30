"""Read-only H2 execution accounting; use archived implementation, never solve."""
import argparse
from collections import Counter
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scoped_proof.io import load, save, sha
from scoped_source.endpoint_supervised import audit, POSITIVE

CASES = {'success':POSITIVE,'weighted_sign-mccormick':'UNKNOWN_NONPOSITIVE',
    'tied_partial_reuse-endpoints':POSITIVE,'tied_partial_reuse-mccormick':POSITIVE,
    'unsafe_tied-endpoints':'UNKNOWN_NONPOSITIVE','unsafe_tied-mccormick':'UNKNOWN_NONPOSITIVE',
    'unresolved_sign-endpoints':POSITIVE,'unresolved_sign-mccormick':POSITIVE,
    **{n:'UNKNOWN_MISSING_EVIDENCE' for n in ('endpoints-missing_certificate','endpoints-missing_both',
                                            'mccormick-missing_certificate','proposal-exception')},
    **{n:'ERROR' for n in ('wrong-source','omit_property','missing_endpoint','exception_after_bundle',
                          'partial_output','wrong_invocation','wrong_mode','descendant','rebind-checker-context',
                          'self-consistent-stdout')},
    **{n:'TIMEOUT' for n in ('produce_delay','proposal_delay','serialization_delay','check_delay',
                            'receive_delay','late_publish','publication-overrun','prelaunch-expiry')},
    'rss':'RESOURCE_LIMIT'}


def derive(root):
    bindings = load(root/'implementation.json')
    for name,digest in bindings.items():
        if sha(root/'implementation'/name) != digest or sha(ROOT/name) != digest:
            raise ValueError('use archived implementation for this accounting')
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
        record = {'case':name,'execution_status':observed['status'],'seconds':observed['seconds'],
            'audit':checked,'receipt_sha256':sha(path/'receipt.json'),
            'caller_observation_sha256':sha(path/'caller_observation.json'),'terminal_sha256':sha(path/'terminal.json'),
            'stage_seconds':{s['phase']:s['seconds'] for s in terminal['stages']},
            'publication_and_overhead_seconds':observed['seconds']-terminal['stage_seconds']}
        if expected in (POSITIVE,'UNKNOWN_NONPOSITIVE','UNKNOWN_MISSING_EVIDENCE'):
            accepted = load(path/'accepted.json'); result = accepted['result']
            record.update(required=accepted['required'],positive=accepted['positive'],missing=accepted['missing'],
                lp_bounds_checked=result['lp_bounds_checked'], source_blocks=result['source_blocks_checked'],
                bounds=[r['lower_bound'] for r in result['duties']],origins=result['origins'],
                bundle_bytes=load(path/'built.json')['bundle_bytes'],source_sha256=accepted['source_sha256'],
                generation_recorded=load(path/'generation.json'))
        records.append(record)
    for case in ('weighted_sign','tied_partial_reuse','unsafe_tied','unresolved_sign'):
        left = root/('success' if case=='weighted_sign' else case+'-endpoints')
        a,b = [load(p/'bundle/proof.json') for p in (left,root/(case+'-mccormick'))]
        if a['request'] != b['request'] or a['reuse_requested'] != b['reuse_requested']:
            raise ValueError('paired source/P/gate/facts differ')
    return {'schema':'H2_SUPERVISION_CONTROL_ARCHIVE_V1','root':str(root),'cases':records,
        'terminal_counts':dict(Counter(r['execution_status'] for r in records)), 'source_bindings':bindings,
        'real_requests_started':0,'new_solves_during_archive':0,'performance_claim':False,
        'scope':'Fixed synthetic declared real graphs. Controlled failures, not native FP or real-model results.',
        'cost_scope':'API entry through observed return; imports, source, LPs, pack, check, receive, publication and cleanup charged. Test-driver setup/logging and offline audit separate.'}


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('root',type=Path)
    group=p.add_mutually_exclusive_group(required=True); group.add_argument('--output',type=Path); group.add_argument('--check',type=Path)
    a=p.parse_args(); result=derive(a.root)
    if a.output: save(a.output,result)
    elif load(a.check)!=result: raise ValueError('archive differs')
    print({'status':'PASS','cases':len(result['cases']),'counts':result['terminal_counts'],'new_solves':0})
