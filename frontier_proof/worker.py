"""Seven-phase checked-frontier work, sharing real intake and native LP proposer."""
import argparse
import math
from pathlib import Path
import sys
import time

from frontier_proof.contract import (accepted_routes, file_record, output_inputs,
                                    policy, retained_roster, route_inputs, route_roster)
from scoped_proof.evidence import bind_source, context, roster
from scoped_proof.io import Events, load, save, tick
from source_enclosure.format import identity


def work(phase, root, deadline):
    inv = load(root/'invocation.json')
    spec = load(root/'spec.json', inv['spec_file_sha256'])
    if policy(spec)['mode'] != 'checked_frontier':
        raise ValueError('frontier worker requires opt-in')
    tick(deadline)
    events = Events(root, phase, inv['started_monotonic'])
    scope, token = spec['scope'], inv['invocation']
    if phase == 'route_propose':
        from checked_route_frontier.build import prefix, propose_final_affine
        doc = events.call('read_source', lambda: load(root/'source.json'))
        digest = bind_source(doc, scope)
        pre = events.call('router_source_construction', lambda: prefix(doc, expected_source_sha256=digest, deadline=deadline))
        events.call('serialize_router_prefix', lambda: save(root/'router_prefix.json', pre))
        rows = events.call('fixed_final_affine_duals', lambda: propose_final_affine(doc, pre, deadline=deadline))
        expected = route_roster(scope)
        prefix_hash = identity(pre)
        if len(rows) != len(expected):
            raise ValueError('route proposal inventory')
        (root/'route_candidates').mkdir(exist_ok=False)
        files = {}
        for i, (row, obligation) in enumerate(zip(rows, expected)):
            tick(deadline)
            record = {'schema': 'FRONTIER_ROUTE_FILE_V1', 'candidate': row,
                'context': {'invocation': token, 'request_sha256': identity(scope), 'source_sha256': digest,
                            'prefix_sha256': prefix_hash, 'index': i, **obligation}}
            name = f'{i:04d}.json'
            files[name] = events.call('serialize_route_candidate_'+str(i),
                                     lambda: save(root/'route_candidates'/name, record))
        tick(deadline)
        events.call('serialize_route_completion', lambda: save(root/'route_complete.json',
            {'schema': 'FRONTIER_ROUTE_COMPLETE_V1', 'invocation': token,
             'request_sha256': identity(scope), 'source': file_record(root/'source.json'),
             'prefix': file_record(root/'router_prefix.json'), 'files': files, 'required': len(expected)}))
    elif phase == 'route_check':
        from checked_route_frontier.check import check_frontier
        doc, pre, rows, receipt_context = events.call('receive_bound_route_files', lambda: route_inputs(root, deadline))
        result = events.call('independent_router_source_and_bounds', lambda: check_frontier(doc, pre, rows,
            expected_source_sha256=bind_source(doc, scope), deadline=deadline))
        events.call('serialize_route_check', lambda: save(root/'route_check.json',
            {'schema': 'FRONTIER_ROUTE_CHECK_V1', 'context': receipt_context, 'frontier': result}))
    elif phase == 'construct':
        from checked_route_frontier.build import finish
        doc, pre, rows, receipt = events.call('receive_checked_routes', lambda: accepted_routes(root, deadline))
        bundle = events.call('checked_lazy_source_construction', lambda: finish(doc, pre, rows,
            expected_source_sha256=bind_source(doc, scope), deadline=deadline))
        if bundle['frontier'] != receipt['frontier']:
            raise ValueError('checked route plan changed')
        record = events.call('serialize_construction', lambda: save(root/'construction.json', bundle))
        events.call('serialize_construction_receipt', lambda: save(root/'construction_receipt.json',
            {'invocation': token, 'request_sha256': identity(scope), 'construction': record,
             'route_check': file_record(root/'route_check.json')}))
    elif phase in ('source_check', 'aggregate'):
        from checked_route_frontier.check import check
        from checked_route_frontier.evidence import aggregate
        doc, pre, route_rows, receipt = events.call('receive_checked_routes', lambda: accepted_routes(root, deadline))
        bundle = events.call('read_construction', lambda: load(root/'construction.json'))
        expected_receipt = {'invocation': token, 'request_sha256': identity(scope),
            'construction': file_record(root/'construction.json'), 'route_check': file_record(root/'route_check.json')}
        if (load(root/'construction_receipt.json') != expected_receipt or bundle['prefix'] != pre or
                bundle['route_candidates'] != route_rows or bundle['frontier'] != receipt['frontier']):
            raise ValueError('construction/route reception mismatch')
        if phase == 'source_check':
            result = events.call('independent_retained_source_and_output_construction', lambda: check(doc, bundle,
                expected_source_sha256=bind_source(doc, scope), deadline=deadline))
            events.call('serialize_source_check', lambda: save(root/'source_check.json', result))
        else:
            candidates, complete = events.call('receive_all_output_candidates', lambda: output_inputs(root, scope, bundle, token, deadline))
            result = events.call('fresh_full_source_routes_and_all_bounds', lambda: aggregate(scope, doc, bundle,
                candidates, invocation=token, proposal_complete=complete, deadline=deadline))
            record = events.call('serialize_evidence', lambda: save(root/'evidence_check.json', result))
            keys = ('status','request_sha256','invocation','required','checked_bounds','positive_bounds',
                    'proposal_complete','complete_output_positive_proof','native_float_proof','route_changing_established',
                    'discharged_by_exclusion','required_output_bounds')
            small = {k: result[k] for k in keys}
            small.update(evidence_check=record, missing=len(result['missing']), nonpositive=len(result['nonpositive']))
            events.call('serialize_terminal_candidate', lambda: save(root/'result_candidate.json', small))
    elif phase == 'propose':
        from scoped_proof.worker import propose
        from full_source.obligations import materialize
        doc, bundle = events.call('read_source_and_construction', lambda:
            (load(root/'source.json'), load(root/'construction.json')))
        digest, bundle_hash = bind_source(doc, scope), identity(bundle)
        expected = retained_roster(scope, bundle)
        by_pair = {tuple(p['pair']): p for p in bundle['pairs']}
        (root/'candidates').mkdir(exist_ok=False)
        files = {}
        for position, (index, obligation) in enumerate(expected):
            tick(deadline)
            pair = by_pair[tuple(obligation['pair'])]
            row = next(r for r in pair['obligations']['rows'] if r['competitor'] == obligation['competitor'])
            lp = events.call('materialize_'+str(index), lambda: materialize(pair['base'], row))
            allowance = (deadline-time.monotonic())/(len(expected)-position)
            if not math.isfinite(allowance) or allowance <= 0:
                raise TimeoutError('no proposal budget left')
            cert, native = events.call('native_lp_'+str(index), lambda: propose(lp, allowance))
            candidate = {'schema': 'SCOPED_LP_CANDIDATE_V1', 'context': context(scope, digest, bundle_hash,
                token, index, obligation, lp), 'certificate': cert,
                'status': 'CANDIDATE' if cert is not None else 'NO_CANDIDATE', 'native': native}
            name = f'{index:04d}.json'
            files[name] = events.call('serialize_candidate_'+str(index), lambda: save(root/'candidates'/name, candidate))
            tick(deadline)
        events.call('serialize_proposal_completion', lambda: save(root/'proposal_complete.json',
            {'schema': 'FRONTIER_OUTPUT_COMPLETE_V1', 'invocation': token,
             'request_sha256': identity(scope), 'bundle_sha256': bundle_hash,
             'original_required': len(roster(scope)), 'retained_indices': [i for i, _ in expected], 'files': files}))
    else:
        raise ValueError('unknown frontier phase')
    tick(deadline)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase'); p.add_argument('root', type=Path); p.add_argument('--deadline', type=float, required=True)
    a = p.parse_args()
    if a.phase in ('route_check','source_check','aggregate'):
        if not sys.flags.no_site:
            raise ValueError('independent check requires python -S')
        def forbid(event, args):
            if event == 'import' and (args[0].split('.')[0] in ('torch','numpy','scipy','highspy','act','gurobipy') or
                    args[0] in ('checked_route_frontier.build','scoped_source.build','full_source.obligations')):
                raise ImportError('checker cannot import producer/model/solver')
            if event.startswith(('subprocess.', 'socket.')) or event in ('os.system','os.fork','os.exec'):
                raise PermissionError('no external execution from checker')
        sys.addaudithook(forbid)
    work(a.phase, a.root.resolve(), a.deadline)
